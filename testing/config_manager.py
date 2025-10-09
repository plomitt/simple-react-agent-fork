"""
Agent Configuration Manager for tracking and comparing different agent configurations.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid


class AgentConfigManager:
    """Manages agent configurations and their performance tracking."""

    def __init__(self, configs_file: str = None):
        if configs_file is None:
            # Get absolute path relative to this script's location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            configs_file = os.path.join(script_dir, "results", "agent_configs.json")
        self.configs_file = configs_file
        self._ensure_configs_file_exists()

    def _ensure_configs_file_exists(self) -> None:
        """Ensure the configs file exists with proper structure."""
        os.makedirs(os.path.dirname(self.configs_file), exist_ok=True)

        if not os.path.exists(self.configs_file):
            # Create initial configs file
            initial_config = {
                "configs": {},
                "global_stats": {
                    "total_configs": 0,
                    "total_runs": 0,
                    "last_updated": datetime.now().isoformat()
                }
            }
            with open(self.configs_file, 'w') as f:
                json.dump(initial_config, f, indent=2)

    def load_configs(self) -> Dict[str, Any]:
        """Load all agent configurations."""
        try:
            with open(self.configs_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"configs": {}, "global_stats": {"total_configs": 0, "total_runs": 0, "last_updated": datetime.now().isoformat()}}

    def save_configs(self, configs: Dict[str, Any]) -> None:
        """Save all agent configurations."""
        # Update global stats
        configs["global_stats"]["last_updated"] = datetime.now().isoformat()
        configs["global_stats"]["total_configs"] = len(configs["configs"])

        # Atomic save
        temp_file = self.configs_file + ".tmp"
        try:
            with open(temp_file, 'w') as f:
                json.dump(configs, f, indent=2)
            os.rename(temp_file, self.configs_file)
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e

    def create_config(
        self,
        name: str,
        model: str,
        max_steps: int,
        description: str = "",
        custom_params: str = ""
    ) -> str:
        """
        Create a new agent configuration.

        Returns:
            The unique ID of the created configuration.
        """
        configs = self.load_configs()

        # Generate unique ID
        config_id = f"config_{uuid.uuid4().hex[:8]}"

        # Create new config
        new_config = {
            "id": config_id,
            "name": name,
            "model": model,
            "max_steps": max_steps,
            "description": description,
            "custom_params": custom_params,
            "created_at": datetime.now().isoformat(),
            "best_score": None,
            "best_run_id": None,
            "total_runs": 0,
            "last_run_at": None
        }

        configs["configs"][config_id] = new_config
        configs["global_stats"]["total_runs"] += 1

        self.save_configs(configs)
        return config_id

    def get_config(self, config_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific agent configuration by ID."""
        configs = self.load_configs()
        return configs["configs"].get(config_id)

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all agent configurations."""
        configs = self.load_configs()
        return list(configs["configs"].values())

    def update_config_performance(
        self,
        config_id: str,
        run_id: str,
        score: float,
        summary: Dict[str, Any]
    ) -> bool:
        """
        Update configuration performance after a run.

        Returns:
            True if this was a new best score, False otherwise.
        """
        configs = self.load_configs()

        if config_id not in configs["configs"]:
            return False

        config = configs["configs"][config_id]

        # Update run tracking
        config["total_runs"] += 1
        config["last_run_at"] = datetime.now().isoformat()

        # Check if this is a new best score
        is_new_best = (
            config["best_score"] is None or
            score > config["best_score"]
        )

        if is_new_best:
            config["best_score"] = score
            config["best_run_id"] = run_id

        self.save_configs(configs)
        return is_new_best

    def calculate_score(self, summary: Dict[str, Any]) -> float:
        """
        Calculate overall score from test run summary.

        Scoring formula:
        - Base score: success_rate * 100
        - Accuracy bonus: accuracy_rate * 50
        - Time efficiency bonus: up to 10 points
        - Step efficiency bonus: up to 10 points

        Returns:
            Overall score (higher is better)
        """
        success_rate = summary.get("success_rate", 0.0)
        accuracy_rate = summary.get("accuracy_rate", 0.0)
        total_questions = summary.get("total_questions", 0)

        if total_questions == 0:
            return 0.0

        # Base score from success rate
        base_score = success_rate / 100

        # Accuracy bonus (only counts if there are successful runs)
        accuracy_bonus = 0.0
        successful_runs = summary.get("successful_runs", 0)
        if successful_runs > 0:
            accuracy_bonus = (accuracy_rate / 100) * 0.5  # Max 0.5 points

        # Time efficiency bonus (faster is better, up to 0.1 points)
        time_bonus = 0.0
        if successful_runs > 0:
            avg_time = summary.get("average_execution_time", 0)
            # Bonus for under 30 seconds average
            if avg_time > 0 and avg_time < 30:
                time_bonus = 0.1 * (1 - avg_time / 30)

        # Step efficiency bonus (fewer steps is better, up to 0.1 points)
        step_bonus = 0.0
        if successful_runs > 0:
            avg_steps = summary.get("average_steps", 0)
            # Bonus for under 10 steps average
            if avg_steps > 0 and avg_steps < 10:
                step_bonus = 0.1 * (1 - avg_steps / 10)

        total_score = base_score + accuracy_bonus + time_bonus + step_bonus
        return round(total_score, 3)

    def get_best_configs(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing configurations."""
        configs = self.load_configs()

        # Filter configs with scores and sort by best_score
        scored_configs = [
            config for config in configs["configs"].values()
            if config["best_score"] is not None
        ]

        scored_configs.sort(key=lambda x: x["best_score"], reverse=True)
        return scored_configs[:limit]

    def delete_config(self, config_id: str) -> bool:
        """Delete an agent configuration."""
        configs = self.load_configs()

        if config_id in configs["configs"]:
            del configs["configs"][config_id]
            self.save_configs(configs)
            return True

        return False

    def get_config_stats(self) -> Dict[str, Any]:
        """Get overall configuration statistics."""
        configs = self.load_configs()

        total_configs = len(configs["configs"])
        configs_with_scores = sum(1 for config in configs["configs"].values() if config["best_score"] is not None)
        total_runs = sum(config.get("total_runs", 0) for config in configs["configs"].values())

        if configs_with_scores > 0:
            scores = [config["best_score"] for config in configs["configs"].values() if config["best_score"] is not None]
            avg_score = sum(scores) / len(scores)
            best_score = max(scores)
        else:
            avg_score = 0.0
            best_score = 0.0

        return {
            "total_configs": total_configs,
            "configs_with_scores": configs_with_scores,
            "total_runs": total_runs,
            "average_score": round(avg_score, 3),
            "best_score": round(best_score, 3),
            "last_updated": configs["global_stats"]["last_updated"]
        }