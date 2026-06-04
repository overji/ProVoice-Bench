import re
from .BaseModel import BaseModelApiInference
import requests

"""HTTP inference wrapper for a Qwen3-80B style chat-completions endpoint."""

class Qwen3_80BApiInference(BaseModelApiInference):
    """Adapter that sends conversations to a Qwen3-80B API service."""

    def __init__(self, url, enable_thinking=False):
        """Initialize endpoint and optional think/answer splitting behavior.

        Args:
            url (str): Chat completions endpoint URL.
            enable_thinking (bool): Whether to parse `<think>...</think>` blocks
                and return only the final answer text.
        """
        super().__init__(url)
        self.enable_thinking = enable_thinking

    def split_think_and_answer(self, text):
        """Split model output into reasoning section and final answer.

        Expected format:
            <think> ... </think>final answer

        Returns:
            tuple[str, str]: (thinking_text, answer_text). If no closing think
            tag is present, the full text is treated as answer.
        """
        pattern = r'(.*?)</think>(.*)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            answer = match.group(2).strip()
            return thinking, answer
        else:
            # No explicit think block was found; treat all content as final answer.
            return "", text.strip()
    
    def infer(self, conversation):
        """Send a conversation to the remote endpoint and return model output.

        Args:
            conversation (list[dict]): Chat history in API-compatible format.

        Returns:
            tuple: `(text_or_answer, None)` where the first value is either raw
            content or extracted final answer depending on `enable_thinking`.
        """
        # Build OpenAI-compatible payload expected by the served API.
        payload = {
            "messages": conversation,
        }
        headers = {"Content-Type": "application/json"}

        # Perform a single synchronous request.
        response = requests.post(self.url, headers=headers, json=payload)

        # Parse assistant content from standard chat completion response schema.
        text = response.json()["choices"][0]["message"]["content"]
        if self.enable_thinking:
            thinking, answer = self.split_think_and_answer(text)
            return answer, None
        else:
            return text, None