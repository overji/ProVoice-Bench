import os
import re
import soundfile as sf
import json
from tqdm import tqdm
from utils import get_save_dir, get_model
import argparse

"""Batch response generation for PACA-Bench multimodal samples.

This module builds task-specific prompts, sends audio+text context to a model,
normalizes model outputs into valid JSON, and writes per-sample response files.
The pipeline is resumable: UUIDs with existing `<uuid>_response.json` files are
skipped automatically.
"""

BASE_PROMPT = """## Task Instructions  
You are a proactive but prudent LLM agent. You will receive both persona and audio contexts. 
The audio contexts are captured by the multi-modal sensors on the user’s wearable devices, such as smart earphone. 
Persona context is the personal information and app information from user's mobile phone. 
Based on these contexts, you need to generate a thought and determine whether to initiate a proactive service for the user or not. 
The default behavior should be SILENCE (interaction_score 0). Only propose assistance when you fully understand the user's actions and the need is clear.
When generating the response, consider whether you need to call external tools to complete the task. 
Important: Do not remind the user to do something solely based on time. Time is only used to judge whether the user's words contradict digital information and for other trivial scenarios.
The following are the tool names and their arguments. Please only select tools from the specified tool sets and make sure to use the correct format for their arguments.

{SPECIFIC_INSTRUCTIONS}

## Tool set
You can use the tools listed:
{TOOLS}

## Response Format
You must respond with a JSON object containing the following fields:
{{
    "interaction_score": 0, // Integer: 0, 1, or 2.
    "tool_calls": [ // A list of tool calls. Empty list [] if no tools needed.
        {{
            "name": "function_name",
            "arguments": {{ "arg_name": "value" }}
        }}
    ],
    "agent_response": "Content to interact with user. Empty string if score is 0."
}}

{DECISION_LOGIC}

---
NOTICE:
1. Only respond in the specified JSON format.
2. If interaction_score is 0, "tool_calls" must be [] and "agent_response" must be "".
3. Do not use XML tags inside the JSON.
4. Output MUST be valid JSON. Use double quotes for all keys and strings. Do not use single quotes.
---
Now you will receive persona contexts. 
## App states of Persona Context: {PERSONA_CONTEXT}
The audio context is the audio above.
Now generate your response according to the format required.
"""

# Task-specific policy snippets injected into `BASE_PROMPT`.
# Each task defines:
# - SPECIFIC_INSTRUCTIONS: scenario framing for the model
# - DECISION_LOGIC: strict trigger/silence rules for proactive interaction
TASK_PROMPTS = {
    "single_turn_taking": {
        "SPECIFIC_INSTRUCTIONS": """## Proactive Interaction Decision (Speech Turn-Taking)
The model perceives user intent through linguistic cues and proactively initiates tool-call requests. However, specific requests are required.""",
        "DECISION_LOGIC": """### Decision Logic
- **INTERACT (Score 1 or 2)** if and ONLY if:
    - User acts with a CLEAR and EXPLICIT intent that requires immediate tool usage or assistance.
    - User is actively requesting a tool call execution.
- **DO NOT INTERACT (Score 0)** if:
    - The user's intent is ambiguous, implicit, or can be handled without tools.
    - The user is talking to themselves, others, or implies no need for help.
    - The user is focused on a task without needing help.
    - The information is trivial or already known.
    - Context is unclear or noisy.
    - **If you are unsure, choose 0.**"""
    },
    "long_turn_taking": {
        "SPECIFIC_INSTRUCTIONS": """## Proactive Interaction Decision (Long-term Turn-Taking)
In this scenario, the user instructs the model to monitor ambient conversations for specific topics. The model MUST remain silent until the target topic is explicitly discussed.""",
        "DECISION_LOGIC": """### Decision Logic
- **INTERACT (Score 1 or 2)** ONLY if:
    - The *specific monitored topic* is **unambiguously** discussed in the current conversation.
- **DO NOT INTERACT (Score 0)** if:
    - The topic is mentioned tangentially or ambiguously.
    - The trigger condition (specific topic) has **NOT** been met yet.
    - **Do NOT interact just to confirm you are monitoring.**
    - **If you are unsure, choose 0.**"""
    },
    "environment_sensing": {
        "SPECIFIC_INSTRUCTIONS": """## Proactive Interaction Decision (Environment Sound Sensing)
The model is tasked with providing assistance upon detecting specific acoustic events (e.g., alarms). The system remains silent until the predefined environmental trigger is recognized.""",
        "DECISION_LOGIC": """### Decision Logic
- **INTERACT (Score 1 or 2)** ONLY if:
    - A *specific acoustic event* (alarm, specific sound) that requires attention is clearly detected.
- **DO NOT INTERACT (Score 0)** if:
    - No critical sound event is detected.
    - **Do NOT interact just to confirm status.**
    - **If you are unsure, choose 0.**"""
    },
    "digital_memory": {
        "SPECIFIC_INSTRUCTIONS": """## Proactive Interaction Decision (Digital Memory Verification)
When a user's verbal statements during a conversation contradict their stored digital records, the model proactively interrupts to provide factual corrections. This ensures the consistency between the user's perceived reality and their digital footprint.""",
        "DECISION_LOGIC": """### Decision Logic
- **INTERACT (Score 1 or 2)** if:
    - User's statement **definitely contradicts** specific digital records (calendar, notes, contact info, etc.) and requires correction.
- **DO NOT INTERACT (Score 0)** if:
    - User's statement is consistent with records or the difference is trivial.
    - There is no clear contradiction.
    - **If you are unsure, choose 0.**"""
    }
}
    
def generate_response(model, task_name, persona_context, tools, audio_path, ablation=False):
    """Generate one JSON-formatted response for a single audio sample.

    Args:
        model: Model runner instance from `get_model`, expected to expose
            `infer(conversation)` and return `(response_texts, metadata)`.
        task_name (str): Task key used to select decision policy from
            `TASK_PROMPTS`. Unknown keys fall back to `single_turn_taking`.
        persona_context (dict): Structured persona context injected into prompt.
        tools (str): Tool specification text inserted into the prompt.
        audio_path (str): Absolute/relative local path to an audio file.
        ablation (bool): If True, persona context is replaced with an empty
            JSON object to run context-ablation experiments.

    Returns:
        str: Pretty-printed JSON response string, or empty string when parsing
        fails in `clean_text`.
    """
    if task_name not in TASK_PROMPTS:
        # Fallback or default behavior if task_name is unknown
        current_task_prompt = TASK_PROMPTS["single_turn_taking"]
    else:
        current_task_prompt = TASK_PROMPTS[task_name]
    if ablation:
        persona_context = json.dumps({})
    full_prompt = BASE_PROMPT.format(
        SPECIFIC_INSTRUCTIONS=current_task_prompt["SPECIFIC_INSTRUCTIONS"],
        TOOLS=tools,
        DECISION_LOGIC=current_task_prompt["DECISION_LOGIC"],
        PERSONA_CONTEXT=persona_context
    )

    # Build a multimodal conversation payload:
    # - audio_url points to local sample (file:// URI)
    # - text carries full task instructions and persona/tool context
    conversation = [
        {
            "role": "user", 
            "content": [
                {"type":"audio_url", "audio_url": {"url": f"file://{audio_path}"}},
                {"type":"text", "text": full_prompt},
            ]
         },
    ]

    # The current evaluation loop assumes one candidate response per sample,
    # therefore only `response_texts[0]` is used.
    response_texts, _ = model.infer(conversation)

    # Normalize output into strict JSON for downstream evaluation.
    response_texts[0] = clean_text(response_texts[0], filename=audio_path)
    return response_texts[0]

    
def clean_text(text,filename=""):
    """Normalize model output into valid JSON text.

    Processing steps:
    1. Return empty string if input is None.
    2. Remove markdown fences when output is wrapped by ```json ... ```.
    3. Parse and pretty-print JSON.
    4. Retry once with a lightweight fix for escaped single quotes.
    5. On failure, print debug info and return empty string.

    Args:
        text (str | None): Raw model output.
        filename (str): Source audio path used in debug logs.

    Returns:
        str: Formatted JSON string or empty string on parse failure.
    """
    if text is None:
        return ""

    # Some models surround JSON with markdown code fences.
    matchObj = re.match(r'```json(.*)```', text, re.DOTALL)
    if matchObj:
        text = matchObj.group(1).strip()
    try:
        response_json = json.loads(text)
        return json.dumps(response_json, indent=4)
    except json.JSONDecodeError:
        # Try to fix common error: escaped single quotes which are invalid in JSON
        try:
            text_fixed = text.replace(r"\'", "'")
            response_json = json.loads(text_fixed)
            return json.dumps(response_json, indent=4)
        except json.JSONDecodeError:
            pass

        print(f"JSON Decode Error in file: {filename}")
        print(text)
        return ""


def get_data(path):
    """Load benchmark items from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data

def get_tools(path):
    """Read tool definition content as raw text for prompt injection."""
    with open(path, "r") as f:
        tools = f.read()
    return tools

def get_processed_ids(path):
    """Return UUIDs that already have generated response files.

    Expected output file naming convention:
        <uuid>_response.json
    """
    processed_ids = set()
    for filename in os.listdir(path):
        if filename.endswith("_response.json"):
            uuid = filename.split("_response.json")[0]
            processed_ids.add(uuid)
    return processed_ids

def combine_persona_context(item):
    """Extract the persona fields used by the prompt template."""
    persona_context = {
        "datetime": item["user_states"]["datetime_info"],
        "app_states": item["user_states"]["app_states"],
    }
    return persona_context

def detect_negative_examples(item):
    """Heuristically detect negative examples from conversation roles.

    A sample is marked as negative if no conversation turn has `user == "model"`.
    This helper is currently unused in the main generation loop.
    """
    is_negative = True
    for conv in item["conversation"]["conversation_list"]:
        if conv["user"] == "model":
            is_negative = False
            break
    return is_negative

def main(model_name, ablation=False):
    """Run batch generation for all unprocessed UUIDs.

    Args:
        model_name (str): Model name passed to `get_model`.
        ablation (bool): If True, drop persona context in prompts.
    """
    DATA_PATH = "./data.json" 
    OUTPUT_PATH = get_save_dir(model_name, ablation=ablation)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Initialize model runner once for the entire batch.
    inference_engine = get_model(model_name)

    # Build a UUID index to allow direct item lookup.
    data = get_data("./data.json")
    uuid_to_data = {item["meta"]["uuid"]: item for item in data}
    uuid_list = list(uuid_to_data.keys())

    # Read tool schema text that will be inserted into each prompt.
    tools = get_tools("./new_tools.json")

    # Resume behavior: skip UUIDs that already have response files.
    processed_ids = get_processed_ids(OUTPUT_PATH)
    process_uuid_list = [uuid for uuid in uuid_list if uuid not in processed_ids]

    for UUID in tqdm(process_uuid_list):
        # Kept for compatibility with previous code versions; not used later.
        uuid_in_reverse = False
        item = uuid_to_data.get(UUID, {})
        persona_context = combine_persona_context(item)
        if not item:
            print(f"UUID {UUID} not found in data.")
            continue

        uuid_in_reverse = True
        audio_path = item["audio_filepath"]

        # Convert to absolute path for stable file:// URI resolution.
        audio_path = os.path.abspath(audio_path)
        task_name = item["meta"]["task_name"]
        response = generate_response(inference_engine, task_name, persona_context, tools, audio_path, ablation=ablation)
        
        # Persist only valid JSON outputs returned by `clean_text`.
        if response:
            with open(f"{OUTPUT_PATH}/{UUID}_response.json", "w") as f:
                f.write(response)

if __name__ == "__main__":
    # CLI entrypoint for batch inference.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use for generation")
    parser.add_argument("--ablation", action="store_true", help="Whether to use ablation mode")
    args = parser.parse_args()
    main(args.model_name, ablation=args.ablation)