import re
from .BaseModel import BaseModelApiInference
import requests

"""HTTP inference wrapper for Qwen3-Omni style chat completion endpoints."""

class Qwen3OmniApiInference(BaseModelApiInference):
    """Adapter for Qwen Omni-compatible services with optional think parsing."""

    def __init__(self, url, enable_thinking=False, model_name=""):
        """Initialize endpoint metadata and output processing behavior.

        Args:
            url (str): Chat completion API URL.
            enable_thinking (bool): Whether to extract answer after think block.
            model_name (str): Optional label for external logging/selection.
        """
        super().__init__(url)
        self.enable_thinking = enable_thinking
        self.model_name = model_name

    def split_think_and_answer(self, text):
        """Split completion text into thinking section and final answer.

        Returns:
            tuple[str, str]: `(thinking, answer)`. If no closing `</think>` tag
            exists, returns empty thinking and full text as answer.
        """
        pattern = r'(.*?)</think>(.*)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            thinking = match.group(1).strip()
            answer = match.group(2).strip()
            return thinking, answer
        else:
            # Fallback path when server output does not include think tags.
            return "", text.strip()
    
    def infer(self, conversation):
        """Run remote inference for a conversation and normalize output shape.

        Args:
            conversation (list[dict]): Model input messages.

        Returns:
            tuple[list[str], None]: A one-element response list and placeholder
            metadata field to match caller expectations.
        """
        # API payload for chat completion style servers.
        payload = {
            "messages": conversation
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.url, headers=headers, json=payload)
        
        # Decode JSON response first, then read assistant message content.
        res_json = response.json()
            
        text = res_json["choices"][0]["message"]["content"]
        if self.enable_thinking:
            thinking, answer = self.split_think_and_answer(text)
            return [answer], None
        else:
            # Persist raw non-thinking outputs for offline inspection/debugging.
            with open("non_thinking_log.log", "a") as f:
                f.write(f"Output:  {text}\n")
            return [text], None