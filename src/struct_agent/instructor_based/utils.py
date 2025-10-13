from __future__ import annotations
from typing import Dict, Any

def print_step_info(step_num: int, thought: str, action: str, observation: str) -> None:
    print(f"\n=== Step {step_num + 1} ===\n\nThought: {'>'*25}\n{thought}\n\nAction: {'>'*25}\n{action}\n\nObservation: {'>'*25}\n{str(observation)[:250]}...\n")

def merge_configs(user_config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user configuration with default configuration.
    User config values override defaults, but missing keys are filled with defaults.
    """
    merged = default_config.copy()
    merged.update(user_config)
    return merged

__all__ = ["merge_configs"]