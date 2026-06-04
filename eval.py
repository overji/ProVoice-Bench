import os
import json
from utils import get_save_dir
import argparse
from tqdm import tqdm
from model_runner.Qwen3_80B import Qwen3_80BApiInference
import re

"""Evaluation pipeline for PACA-Bench model outputs.

This module evaluates two aspects:
1. Activity decision quality (whether the model should respond at all).
2. Response/tool-call quality for active cases, using an LLM-as-a-judge.
"""

PROMPT = """
You are a judge. Your task is to evaluate the tool call action predicted by the model.

Specifically, you are given a set of contexts that describe the audio and the user's digital app states, as well as the tool call action that the model might do to help the user. You are also given the response that the model will speak to user.

You need to decide whether the model's predicted tool call action is sensible given the contexts. To facilitate this, you are further given a reference action which suggests what user really want the model to do, you should decide whether the predicted tool call action is sensible in assisting user to achieve their goal.

You also need to decide whether the model's response could help user in the specific situation based on the reference response.

Contexts: {CONTEXTS}

Reference action of the user: {REF_TOOL_CALL}

Reference response: {TARGET_RESPONSE}

Model predicted tool call action: {PRED_TOOL_CALL}

Model predicted response: {RESPONSE}

NOTE:

Please evaluate the prediction with a score that’s either 0, 1 or 2 that corresponds to "not helpful", "borderline helpful", or "very helpful".

A score of 0 → The predicted tool call is not helpful to the person at all, or hallucinates harmful actions. 
A score of 1 → The predicted tool call might be helpful, but misses key information or uses a slightly incorrect tool. Even if the tool differs from the reference action, as long as it somewhat accomplishes the user's goal, score it as 1. 
A score of 2 → The predicted tool call is definitely helpful. It successfully captures the user's core intent (e.g., adding an item to a list). Even if the tool differs from the reference action, or uses different argument phrasing that yields the same result, as long as it logically accomplishes the user's goal, score it as 2.

Please evaluate the response with a score that’s either 0, 1 or 2 that corresponds to "not suitable", "borderline suitable", or "very suitable". 

A score of 0 → The response doesn't match tool call at all, or the response is not helpful to user in the situation. 
A score of 1 → The response might match tool call, and the response might be helpful to user in the situation, but you’re not confident.
A score of 2 → The response definitely matches tool call, and the response is definitely helpful to user.

IMPORTANT SCORING INSTRUCTION:

Please leverage the reference action when scoring, but do not treat it as the single possible answer. Focus on whether the user's core goal is advanced.

Please notice that there is natural difference between the reference action and the predicted action due to the difference of being model and human. Prioritize semantic equivalence over exact string matches.

If the predicted action's arguments (e.g., search queries, item names, descriptions) are phrased differently from the reference but targets the same underlying information or functional outcome, this must be scored as 2.

As long as the predicted action could logically accomplish the user's goal, please score it high even if it differs from the reference action.

Please only output the tool call score and the response score and nothing else. Do not add any explanation to your final answer.

Wrap the tool call score with <tool_call_score> and </tool_call_score> tags and wrap the response score with <response_score> and </response_score>. For instance, an example full output should look like this: <tool_call_score>2</tool_call_score>\n<response_score>2</response_score>
"""

def generate_response(inference_engine, ref_tool_call, pred_tool_call, tool_calls, contexts, pred_response, target_response):
    """Ask the judge model to score one prediction.

    Args:
        inference_engine: Judge model wrapper exposing `infer`.
        ref_tool_call: Ground-truth tool call action.
        pred_tool_call: Predicted tool call from evaluated model.
        tool_calls: Tool specification payload (currently unused in prompt body).
        contexts: Scenario cues and user/app context.
        pred_response: Predicted natural-language response.
        target_response: Ground-truth response.

    Returns:
        str: Raw judge model output string.
    """
    # Generate the conversation prompt with all necessary context for the judge
    conversation = [
        {
            "role": "user",
            "content": PROMPT.format(
                CONTEXTS=json.dumps(contexts),
                REF_TOOL_CALL=json.dumps(ref_tool_call),
                PRED_TOOL_CALL=json.dumps(pred_tool_call),
                RESPONSE=pred_response,
                TARGET_RESPONSE=target_response,
            )
        }
    ]
    # Call the inference engine to get the judge's response
    response = inference_engine.infer(conversation)
    return response[0]

def judge_active_score(uuid_to_target_data,uuid_to_origin_data,origin_path, model_name=""):
    """Compute activity-decision metrics and per-UUID binary correctness.

    A sample is considered positive if target has a tool call or response.
    Predicted positive is based on non-empty tool_calls or agent_response.

    Returns:
        tuple[float, dict]: Overall average active score and UUID->score map.
    """
    final_score = 0
    uuid_to_score = {}
    category_score = {}
    category_nums = {}
    category_tp = {}
    category_fp = {}
    category_fn = {}
    category_tn = {}
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    overall_tn = 0
    # Iterate through all UUIDs to calculate active score
    for uuid,origin_data in tqdm(uuid_to_origin_data.items()):
        # Get target tool_call data for the UUID
        target_data = uuid_to_target_data.get(uuid, {})
        if target_data == {}:
            continue
        if not isinstance(origin_data, dict):
            continue
        # `interaction_score` is read for compatibility with result schema;
        # current active scoring uses tool/response presence checks below.
        active_score = origin_data.get("interaction_score", 0)
        category = target_data.get("category","unknown")
        if category not in category_score:
            category_score[category] = 0
            category_nums[category] = 0
        uuid_score = 0
        # If target tool_call has a value, the model should make a tool_call (i.e., active response)
        # Because in OB2 testing, all cases require a tool_call
        # And in the truncated negative sample testing, no tool_call is required
        if target_data.get("tool_call") or target_data.get("response"):
            if origin_data.get("tool_calls") or origin_data.get("agent_response"):
                uuid_score = 1
                category_tp[category] = category_tp.get(category, 0) + 1
                overall_tp += 1
            else:
                uuid_score = 0
                category_fn[category] = category_fn.get(category, 0) + 1
                overall_fn += 1
        else:
            if not origin_data.get("tool_calls") and not origin_data.get("agent_response"):
                uuid_score = 1
                category_tn[category] = category_tn.get(category, 0) + 1
                overall_tn += 1
            else:
                uuid_score = 0
                category_fp[category] = category_fp.get(category, 0) + 1
                overall_fp += 1
        final_score += uuid_score
        category_score[category] += uuid_score
        category_nums[category] += 1
        uuid_to_score[uuid] = uuid_score
    
    for category in category_score:
        avg_cat_score = category_score[category] / category_nums[category]
        tp = category_tp.get(category, 0)
        fp = category_fp.get(category, 0)
        fn = category_fn.get(category, 0)
        tn = category_tn.get(category, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        print(f"Category: {category}, TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"Category: {category}, Precision: {precision}, Recall: {recall}, F1: {f1}, FPR: {fpr}, Accuracy: {accuracy}")

    overall_total = overall_tp + overall_fp + overall_fn + overall_tn
    overall_accuracy = (overall_tp + overall_tn) / overall_total if overall_total > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_fpr = overall_fp / (overall_fp + overall_tn) if (overall_fp + overall_tn) > 0 else 0

    print(
        f"Overall: TP: {overall_tp}, FP: {overall_fp}, FN: {overall_fn}, TN: {overall_tn}"
    )
    print(
        f"{model_name} - Overall: Accuracy: {overall_accuracy}, Recall: {overall_recall}, FPR: {overall_fpr}"
    )
    return final_score / len(uuid_to_origin_data), uuid_to_score

def get_score_from_response(judge_response):
    """Extract tool-call and response scores from judge XML-like tags."""
    # Extract tool call score and response score using regex from the judge's output
    tool_call_score_match = re.search(r"<tool_call_score>(\d)</tool_call_score>", judge_response)
    response_score_match = re.search(r"<response_score>(\d)</response_score>", judge_response)
    # Parse the scores, defaulting to 0 if not found
    tool_call_score = int(tool_call_score_match.group(1)) if tool_call_score_match else 0
    response_score = int(response_score_match.group(1)) if response_score_match else 0
    return tool_call_score, response_score

def get_processed_ids(path):
    """Return UUIDs that already have saved judge outputs.

    The function creates the directory if missing so callers can use it safely
    without additional existence checks.
    """
    # Initialize a set to keep track of processed UUIDs
    processed_ids = set()
    # Create the directory if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)
        return processed_ids
    # Loop through files in the directory to find already processed results
    for filename in os.listdir(path):
        if filename.endswith("_judge_response.json"):
            # Extract UUID from filename
            uuid = filename.split("_judge_response.json")[0]
            processed_ids.add(uuid)
    return processed_ids

def judge_response_accuracy(uuid_to_target_data, uuid_to_origin_data, origin_path, tool_calls, uuid_to_score):
    """Run LLM-based fine-grained scoring for valid active predictions.

    This stage only evaluates samples that passed the active-stage gate
    (`uuid_to_score[uuid] == 1`) and have not been judged before.
    """
    # Initialize the inference engine for judging
    inference_engine = Qwen3_80BApiInference(url="http://localhost:8947/v1/chat/completions")

    # Get the set of already processed IDs to avoid re-running
    processed_ids = get_processed_ids(os.path.join(origin_path,"judge"))

    # Iterate through all data items with a progress bar
    for uuid, origin_data in tqdm(uuid_to_origin_data.items()):
        # Skip if the item score is 0 (no interaction needed/performed) or already processed
        if uuid_to_score.get(uuid, 0) == 0 or uuid in processed_ids:
            continue
        # Get target and predicted data
        target_tool_call = uuid_to_target_data.get(uuid, {}).get("tool_call", {})
        pred_tool_call = origin_data.get("tool_calls", [])
        contexts = uuid_to_target_data.get(uuid, {}).get("cues", {})
        pred_response = origin_data.get("agent_response", "")
        target_response = uuid_to_target_data.get(uuid, {}).get("response", "")
        
        # Check if the model should respond but didn't
        if target_response or target_tool_call:
            if not pred_tool_call and not pred_response:
                # If the model should respond but did not, skip scoring
                continue
        # Check if the model shouldn't respond
        if not target_response and not target_tool_call:
            # You should respond nothing
            continue
        
        # Generate the judgment using the LLM
        judge_response = generate_response(
            inference_engine,
            ref_tool_call=target_tool_call,
            pred_tool_call=pred_tool_call,
            tool_calls=tool_calls,
            contexts=contexts,
            pred_response=pred_response,
            target_response=target_response,
        )
        # Parse the judgment scores
        tool_call_score, response_score = get_score_from_response(judge_response)
        # Save the judgment result to a JSON file
        os.makedirs(os.path.join(origin_path,"judge"), exist_ok=True)
        with open(os.path.join(origin_path,"judge", f"{uuid}_judge_response.json"), "w") as f:
            judge_response_json = {
                "tool_call_score": tool_call_score,
                "response_score": response_score,
            }
            json.dump(judge_response_json, f, indent=4)

def calculate_res_score(tool_call_score, response_score, target_tool_call, target_response, actual_tool_call, actual_response):
    """Convert judge scores and presence checks into normalized final scores.

    Returns:
        tuple[float, float, str]:
            - normalized tool-call score in [0, 1]
            - normalized response score in [0, 1]
            - confusion label in {TP, TN, FP, FN}
    """
    final_tool_call_score = 0
    final_response_score = 0
    judge_type = ""
    
    # Helper function: check if item is non-empty
    def is_present(item):
        if not item:
            return False
        if isinstance(item, (list, dict, str)) and len(item) == 0:
            return False
        return True

    has_target = is_present(target_tool_call) or is_present(target_response)
    has_pred = is_present(actual_tool_call) or is_present(actual_response)

    if has_target:
        if has_pred:
            judge_type = "TP"
            final_tool_call_score = tool_call_score / 2
            final_response_score = response_score / 2
        else:
            judge_type = "FN"
            final_tool_call_score = 0
            final_response_score = 0
    else:
        if not has_pred:
            judge_type = "TN"
            final_tool_call_score = 1
            final_response_score = 1
        else:
            judge_type = "FP"
            final_tool_call_score = 0
            final_response_score = 0
            
    return final_tool_call_score, final_response_score, judge_type

def process_score(uuid_to_score,output_path,uuid_to_target_data, uuid_to_origin_data):
    """Aggregate per-UUID judged scores into category and overall summaries."""
    uuid_to_tool_call_score = {}
    uuid_to_response_score = {}
    category_to_scores = {}
    uuid_to_final_response_accuracy = {}
    
    # Read scores from files
    if not os.path.exists(os.path.join(output_path,"judge")):
        os.makedirs(os.path.join(output_path,"judge"))

    for filename in os.listdir(os.path.join(output_path,"judge")):
        if filename.endswith("_judge_response.json"):
            uuid = filename.split("_judge_response.json")[0]
            try:
                with open(os.path.join(output_path,"judge", filename), "r") as f:
                    judge_response = json.load(f)
                    uuid_to_tool_call_score[uuid] = judge_response.get("tool_call_score", 0)
                    uuid_to_response_score[uuid] = judge_response.get("response_score", 0)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    for uuid, score in uuid_to_score.items():
        target_data = uuid_to_target_data.get(uuid, {})
        origin_data = uuid_to_origin_data.get(uuid, {}) # Get actual model output
        category = target_data.get("category","unknown")
        
        tool_call_score = uuid_to_tool_call_score.get(uuid, 0)
        response_score = uuid_to_response_score.get(uuid, 0)
        
        # Here pred_tool_call and pred_response should be from model prediction, not from target_data
        pred_tool_call = origin_data.get("tool_calls", [])
        pred_response = origin_data.get("agent_response", "")
        
        final_tool_call_score, final_response_score, judge_type = calculate_res_score(
            tool_call_score,
            response_score,
            target_data.get("tool_call", []), # 确保默认值一致
            target_data.get("response", ""),
            pred_tool_call,
            pred_response,
        )
        if category not in category_to_scores:
            category_to_scores[category] = {
                "response_accuracy": [],
            }
        response_accuracy = ((final_tool_call_score + final_response_score) / 2) * score
        category_to_scores[category]["response_accuracy"].append(response_accuracy)
        uuid_to_final_response_accuracy[uuid] = response_accuracy
        
    # for category in category_to_scores:
        # print(f"Category: {category}, Number of Samples: {len(category_to_scores[category]['response_accuracy'])}")
    # print("Total UUIDs scored for response accuracy:", len(uuid_to_final_response_accuracy))

    for category, scores_dict in category_to_scores.items():
        avg_response_accuracy = sum(scores_dict["response_accuracy"]) / len(scores_dict["response_accuracy"]) if scores_dict["response_accuracy"] else 0
        print(f"Category: {category}, Average Response Accuracy: {avg_response_accuracy}")
    
    print(f"Overall Average Response Accuracy: {sum(uuid_to_final_response_accuracy.values()) / len(uuid_to_final_response_accuracy) if uuid_to_final_response_accuracy else 0}")
       
def main(model_name, ablation=False):
    """Entry point for full evaluation.

    Steps:
    1. Load ground-truth data and summarize per-task label statistics.
    2. Load model predictions from output directory.
    3. Compute active score metrics.
    4. Judge response quality for active positives.
    5. Aggregate and print final accuracy summaries.
    """
    # Initialize data structures
    uuid_to_target_data = {}
    uuid_to_origin_data = {}
    target_path = "data.json"
    origin_path = get_save_dir(model_name, ablation=ablation)
    tool_call_path = "new_tools.json"
    statistics = {}
    # Load tool definitions
    with open(tool_call_path, "r") as f:
        tool_calls = json.load(f)
    # Load ground truth data
    with open(target_path, "r") as f:
        datas = json.load(f)
        for data in datas:
            # load target data
            uuid = data["meta"]["uuid"]
            tool_calls = []
            response = ""
            for conv in data["conversation"]["conversation_list"]:
                if conv["user"] == "model":
                    tool_calls = conv.get("tool_calls", [])
                    response = conv.get("content", "")
            cues = data["scenario_cues"]
            task_name = data["meta"].get("task_name","unknown")
            # Update task statistics
            if task_name not in statistics:
                statistics[task_name] = {
                    "T": 0,
                    "F": 0
                }
            if not tool_calls and not response:
                statistics[task_name]["F"] += 1
            else:
                statistics[task_name]["T"] += 1
            uuid_to_target_data[uuid] = {
                "tool_call": tool_calls,
                "cues": cues,
                "response": response,
                "category": data["meta"]["task_name"],
            }
    print("Data Statistics:")
    # Print statistics for each task
    for task_name, stats in statistics.items():
        total = stats["T"] + stats["F"]
        print(f"Task: {task_name}, Needs Response: {stats['T']} ({(stats['T']/total)*100:.2f}%), No Response: {stats['F']} ({(stats['F']/total)*100:.2f}%)")
    # Load model output data
    for filename in os.listdir(origin_path):
        if filename.endswith("_response.json"):
            uuid = filename.split("_response.json")[0]
            with open(os.path.join(origin_path, filename), "r") as f:
                origin_response = json.load(f)
                uuid_to_origin_data[uuid] = origin_response
    
    # Calculate active score
    active_avg, uuid_to_score = judge_active_score(uuid_to_target_data, uuid_to_origin_data, origin_path, model_name=model_name)
    print(f"Average Active Score: {active_avg}")
    judge_response_accuracy(uuid_to_target_data, uuid_to_origin_data, origin_path, tool_calls, uuid_to_score)
    process_score(uuid_to_score, origin_path, uuid_to_target_data, uuid_to_origin_data)

if __name__ == "__main__":
    # Command-line interface for selecting model output directory to evaluate.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate")
    parser.add_argument("--ablation", action="store_true", help="Whether to use ablation mode")
    args = parser.parse_args()
    print(f"Evaluating model: {args.model_name}, Ablation mode: {args.ablation}")
    main(args.model_name, ablation=args.ablation)
        