from typing import Dict, Any

def merge_configs(user_config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user configuration with default configuration.
    User config values override defaults, but missing keys are filled with defaults.

    Args:
        user_config: Configuration provided by the user
        default_config: Default configuration with fallback values

    Returns:
        Merged configuration dictionary
    """
    merged = default_config.copy()
    merged.update(user_config)
    return merged

__all__ = ["merge_configs"]