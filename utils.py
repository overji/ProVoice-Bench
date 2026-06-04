from model_runner.Qwen3Omni import Qwen3OmniApiInference

"""Utility helpers for output paths and model backend selection.

The benchmark scripts use these helpers to keep naming conventions and model
router logic centralized in one place.
"""

def get_save_dir(model_name, ablation=False):
    """Return output directory for a model alias.

    Args:
        model_name (str): User-facing model alias used by CLI scripts.
        ablation (bool): Whether to append `_ablation` suffix.

    Returns:
        str: Relative output directory path.
    """
    # Map model aliases to their canonical output folders.
    model_name_to_dir = {
        "qwen3omni": "./output/qwen3omni_base",
    }

    # In ablation mode, outputs are isolated from normal runs.
    if ablation:
        return model_name_to_dir.get(model_name, "./output/default_model_dir") + "_ablation"
    else:
        return model_name_to_dir.get(model_name, "./output/default_model_dir")

def get_model(model_name):
    """Instantiate and return a model runner based on alias.

    Args:
        model_name (str): Model alias used by scripts.

    Returns:
        A model runner instance exposing `infer(conversation)`.

    Raises:
        ValueError: If model alias is not recognized.
    """
    # Route each alias to the corresponding backend endpoint/config.
    if model_name == "qwen3omni":
        return Qwen3OmniApiInference(url="http://localhost:8947/v1/chat/completions")
    else:
        raise ValueError(f"Unsupported model name: {model_name}")