import requests
from abc import ABC, abstractmethod

"""Base abstractions for model inference backends.

The project supports multiple model providers and transport modes. These base
classes define a minimal common interface so evaluation and generation scripts
can call different backends consistently.
"""

class BaseModelApiInference(ABC):
    """Abstract base class for HTTP/API-based model inference implementations."""

    def __init__(self, url):
        """Store the endpoint URL used by subclasses when making API requests."""
        self.url = url

    @abstractmethod
    def infer(self, conversation):
        """Run model inference for a conversation payload.

        Args:
            conversation: A model-specific conversation object, typically a list
                of role/content dictionaries.

        Returns:
            Subclasses should return model output in the format expected by the
            caller (for this project, usually `(responses, metadata)`).
        """
        pass

class BaseModel:
    """Generic non-API model interface for local/runtime integrated backends."""

    def __init__(self):
        """Initialize base model state."""
        pass
    
    @abstractmethod
    def infer(self, conversation):
        """Run inference for a conversation input.

        Subclasses must implement this to provide a unified calling contract.
        """
        pass