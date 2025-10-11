from __future__ import annotations

import json
import traceback
import os
import csv
import uuid
import shutil
import io
import sys
import time
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import docker

from struct_agent.instructor_based.new_agent import VERBOSITY_STANDARD, run_react_loop
from struct_agent.instructor_based.client_manager import build_client
try:
    from testing.config_manager import AgentConfigManager
except ImportError:
    from config_manager import AgentConfigManager


# Docker Container Restart Feature
#
# This testing system includes functionality to automatically restart related Docker containers
# before each test question to reset search engine request limits and prevent empty search results.
#
# Usage:
#   1. Command line: poetry run python testing/cli.py --restart-searxng --agent-config-id <config_id>
#   2. Environment variables: export RESTART_SEARCH_CONTAINERS=true
#   3. Custom container list: --containers-to-restart redis,searxng,caddy,nginx
#   4. Environment variable for containers: export CONTAINERS_TO_RESTART=redis,searxng,caddy
#
# The feature will:
#   - Restart multiple containers that support search functionality (redis, searxng, caddy by default)
#   - Use Docker Python SDK for reliable container operations
#   - Log restart timing and status for each container
#   - Continue testing even if some restarts fail (with warning)
#   - Show clear status messages about restart operations
#   - Handle Docker daemon connection issues gracefully


def restart_search_containers(containers_to_restart: List[str] = None, timeout: int = 30) -> Dict[str, bool]:
    """
    Restart multiple Docker containers to reset search engine request limits.

    Args:
        containers_to_restart: List of container names to restart (default: ["redis", "searxng", "caddy"])
        timeout: Maximum time to wait for container restart (default: 30 seconds)

    Returns:
        Dict[str, bool]: Dictionary mapping container names to their restart success status
    """
    if containers_to_restart is None:
        containers_to_restart = ["redis", "searxng", "caddy"]

    print(f"🔄 Restarting {len(containers_to_restart)} containers: {', '.join(containers_to_restart)}")

    results = {}

    try:
        # Connect to Docker daemon (OrbStack uses the same socket)
        client = docker.from_env()

        # Test connection
        client.ping()
        print(f"  ✓ Connected to Docker daemon")

    except docker.errors.DockerException as e:
        print(f"  ❌ Error: Failed to connect to Docker daemon: {e}")
        print(f"     Is Docker/OrbStack running and accessible?")
        # Return failure for all containers
        return {container: False for container in containers_to_restart}

    restart_start_time = time.time()

    for container_name in containers_to_restart:
        try:
            print(f"  Restarting {container_name}...")
            container = client.containers.get(container_name)
            container.restart(timeout=timeout)
            print(f"  ✓ {container_name} restarted successfully")
            results[container_name] = True

        except docker.errors.NotFound:
            print(f"  ⚠ Warning: Container '{container_name}' not found - skipping")
            results[container_name] = False

        except docker.errors.APIError as e:
            print(f"  ❌ Error: Failed to restart '{container_name}': {e}")
            results[container_name] = False

        except Exception as e:
            print(f"  ❌ Error: Unexpected error restarting '{container_name}': {e}")
            results[container_name] = False

    restart_time = time.time() - restart_start_time

    # Summary
    successful_restarts = sum(results.values())
    total_containers = len(containers_to_restart)

    if successful_restarts == total_containers:
        print(f"  ✓ All containers restarted successfully in {restart_time:.1f}s")
    elif successful_restarts > 0:
        print(f"  ⚠ {successful_restarts}/{total_containers} containers restarted in {restart_time:.1f}s")
        print(f"    Some containers failed - search results may be affected")
    else:
        print(f"  ❌ All container restarts failed in {restart_time:.1f}s")
        print(f"    Search results will likely be affected by rate limits")

    return results


class TeeStream:
    """A stream that writes to both the original stdout and a capture buffer."""

    def __init__(self, original_stdout, capture_buffer):
        self.original_stdout = original_stdout
        self.capture_buffer = capture_buffer

    def write(self, text):
        """Write to both original stdout and capture buffer."""
        # Write to original stdout for immediate console display
        self.original_stdout.write(text)
        # Write to capture buffer for file saving
        self.capture_buffer.write(text)
        return len(text)

    def flush(self):
        """Flush both streams."""
        self.original_stdout.flush()
        self.capture_buffer.flush()

    def __getattr__(self, name):
        """Delegate any other attribute access to original stdout."""
        return getattr(self.original_stdout, name)


class ConsoleCapture:
    """Captures console output for checkpointing and logging while displaying to console."""

    def __init__(self):
        self.buffer = io.StringIO()
        self.original_stdout = sys.stdout
        self.tee_stream = None
        self.is_capturing = False

    def start_capture(self):
        """Start capturing console output while displaying to console."""
        if not self.is_capturing:
            self.tee_stream = TeeStream(self.original_stdout, self.buffer)
            sys.stdout = self.tee_stream
            self.is_capturing = True

    def stop_capture(self):
        """Stop capturing and restore original stdout."""
        if self.is_capturing:
            sys.stdout = self.original_stdout
            self.is_capturing = False

    def get_output(self) -> str:
        """Get the captured output."""
        return self.buffer.getvalue()

    def clear(self):
        """Clear the captured output."""
        self.buffer = io.StringIO()


class RunManager:
    """Manages test runs with checkpoint and resume functionality."""

    def __init__(self, base_output_dir: str = "testing/results"):
        self.base_output_dir = base_output_dir
        os.makedirs(base_output_dir, exist_ok=True)

    def generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"run_{timestamp}_{unique_id}"

    def create_run_directory(self, run_id: str) -> str:
        """Create a new run directory with subdirectories and return its path."""
        run_dir = os.path.join(self.base_output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Create subdirectories for organization
        temp_dir = os.path.join(run_dir, "temp")
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

        return run_dir

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all runs with their status."""
        runs = []

        if not os.path.exists(self.base_output_dir):
            return runs

        for item in os.listdir(self.base_output_dir):
            run_path = os.path.join(self.base_output_dir, item)
            if os.path.isdir(run_path) and item.startswith("run_"):
                checkpoint_file = os.path.join(run_path, "checkpoint.json")

                run_info = {
                    "run_id": item,
                    "path": run_path,
                    "is_complete": False,
                    "current_question": 0,
                    "total_questions": 0,
                    "start_time": None
                }

                if os.path.exists(checkpoint_file):
                    try:
                        with open(checkpoint_file, 'r') as f:
                            checkpoint = json.load(f)
                        run_info.update({
                            "is_complete": checkpoint.get("is_complete", False),
                            "current_question": checkpoint.get("current_question", 0),
                            "total_questions": checkpoint.get("total_questions", 0),
                            "start_time": checkpoint.get("start_time")
                        })
                    except (json.JSONDecodeError, KeyError):
                        pass

                runs.append(run_info)

        # Sort by start time (newest first), None values last
        runs.sort(key=lambda x: (x.get("start_time") is None, x.get("start_time", "")), reverse=True)
        return runs

    def save_checkpoint(self, run_id: str, checkpoint_data: Dict[str, Any], agent_config_id: Optional[str] = None) -> None:
        """Save checkpoint data atomically."""
        run_dir = os.path.join(self.base_output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Include agent config ID in checkpoint data
        if agent_config_id:
            checkpoint_data["agent_config_id"] = agent_config_id

        checkpoint_file = os.path.join(run_dir, "checkpoint.json")
        temp_file = checkpoint_file + ".tmp"

        try:
            with open(temp_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)

            # Atomic rename
            os.rename(temp_file, checkpoint_file)
        except Exception as e:
            # Clean up temp file if something went wrong
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise e

    def load_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data for a run."""
        run_dir = os.path.join(self.base_output_dir, run_id)
        checkpoint_file = os.path.join(run_dir, "checkpoint.json")

        if not os.path.exists(checkpoint_file):
            return None

        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def save_checkpoint_results(self, run_id: str, results: List[Dict[str, Any]], summary: Dict[str, Any], console_output: str = "", agent_config_id: Optional[str] = None) -> None:
        """Save intermediate results atomically to checkpoints folder."""
        run_dir = os.path.join(self.base_output_dir, run_id)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        temp_dir = os.path.join(run_dir, "temp")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save CSV checkpoint
        csv_file = os.path.join(checkpoints_dir, f"CHECKPOINT_results_{timestamp}.csv")
        temp_csv = os.path.join(temp_dir, f"csv_temp_{timestamp}.tmp")

        # Save comprehensive summary checkpoint
        summary_file = os.path.join(checkpoints_dir, f"CHECKPOINT_summary_{timestamp}.txt")
        temp_summary = os.path.join(temp_dir, f"summary_temp_{timestamp}.tmp")

        # Save console log checkpoint
        log_file = os.path.join(checkpoints_dir, f"CHECKPOINT_log_{timestamp}.txt")
        temp_log = os.path.join(temp_dir, f"log_temp_{timestamp}.tmp")

        try:
            # Save CSV checkpoint with enhanced columns
            with open(temp_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['question_level', 'question_text', 'agent_answer', 'expected_answer', 'is_correct', 'failure_type'])

                for result in results:
                    question_level = result.get('question_level', 'unknown')
                    question_text = result.get('question', '')
                    # Truncate question text for CSV readability
                    if len(question_text) > 100:
                        question_text = question_text[:97] + "..."
                    
                    agent_answer = result['agent_answer'] if result['success'] else result['error']
                    expected_answer = result['expected_answer']
                    is_correct = result['is_correct'] if result['success'] else False
                    failure_type = result.get('failure_type', 'unknown')

                    writer.writerow([question_level, question_text, agent_answer, expected_answer, is_correct, failure_type])

            # Atomic rename
            os.rename(temp_csv, csv_file)

            # Save comprehensive summary checkpoint
            with open(temp_summary, 'w', encoding='utf-8') as f:
                f.write(f"CHECKPOINT - Run {run_id}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Current question: {summary.get('current_question', 0)}/{summary.get('total_questions', 0)}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if agent_config_id:
                    f.write(f"Agent Config ID: {agent_config_id}\n")
                f.write("")
                
                # Add comprehensive summary content
                f.write("BASIC STATISTICS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Total questions:           {summary['total_questions']}\n")
                f.write(f"Successfully answered:     {summary['successful_runs']}\n")
                f.write(f"Failed to answer:          {summary['failed_runs']}\n")
                f.write(f"Success rate:              {summary['success_rate']:.1f}%\n")
                f.write(f"Correct answers:           {summary['correct_answers']}\n")
                f.write(f"Incorrect answers:         {summary['incorrect_answers']}\n")
                if summary['successful_runs'] > 0:
                    f.write(f"Accuracy rate:             {summary['accuracy_rate']:.1f}%\n")
                f.write(f"Total execution time:      {summary['total_execution_time']:.2f}s\n")
                f.write(f"Average time per question: {summary['average_execution_time']:.2f}s\n")
                if summary['successful_runs'] > 0:
                    f.write(f"Total steps taken:         {summary['total_steps']}\n")
                    f.write(f"Average steps per question:{summary['average_steps']:.1f}\n")
                
                # Add detailed question table
                f.write("")
                f.write(format_question_table(results))
                
                # Add enhanced sections
                f.write(format_failure_breakdown(summary))
                f.write(format_level_statistics(summary))
                
                # Add insights
                f.write("")
                f.write("KEY INSIGHTS\n")
                f.write("=" * 80 + "\n")
                
                # Performance insights
                if summary.get('level_stats'):
                    best_level = None
                    worst_level = None
                    best_success_rate = -1
                    worst_success_rate = 101
                    
                    for level, stats in summary['level_stats'].items():
                        success_rate = (stats['successful_runs'] / stats['total'] * 100) if stats['total'] > 0 else 0
                        if success_rate > best_success_rate:
                            best_success_rate = success_rate
                            best_level = level
                        if success_rate < worst_success_rate:
                            worst_success_rate = success_rate
                            worst_level = level
                    
                    if best_level and worst_level and best_level != worst_level:
                        f.write(f"- Agent performs best on Level {best_level} questions ({best_success_rate:.1f}% success rate)\n")
                        f.write(f"- Agent struggles most with Level {worst_level} questions ({worst_success_rate:.1f}% success rate)\n")
                
                # Failure analysis
                failure_breakdown = summary.get('failure_breakdown', {})
                if failure_breakdown:
                    total_failures = sum(failure_breakdown.values())
                    if total_failures > 0:
                        primary_failure = max(failure_breakdown.items(), key=lambda x: x[1])
                        failure_percent = (primary_failure[1] / summary['total_questions'] * 100)
                        f.write(f"- Primary failure mode: {primary_failure[0].replace('_', ' ').title()} ({failure_percent:.1f}% of questions)\n")
                
                # Timing insights
                if summary['successful_runs'] > 0:
                    avg_time = summary['average_execution_time']
                    avg_steps = summary['average_steps']
                    f.write(f"- Average time per successful question: {avg_time:.1f}s\n")
                    f.write(f"- Average steps per successful question: {avg_steps:.1f}\n")
                
                # Overall assessment - use score-based if agent config available
                if agent_config_id:
                    try:
                        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                        from config_manager import AgentConfigManager
                        config_manager = AgentConfigManager()
                        score = config_manager.calculate_score(summary)
                        assessment = get_score_assessment(score)
                        f.write(f"- Score-based assessment: {assessment} ({score:.3f})\n")
                    except Exception:
                        # Fallback to success rate based assessment
                        success_rate = summary['success_rate']
                        if success_rate >= 80:
                            assessment = "Excellent performance"
                        elif success_rate >= 60:
                            assessment = "Good performance"
                        elif success_rate >= 40:
                            assessment = "Moderate performance"
                        else:
                            assessment = "Needs improvement"
                        f.write(f"- Overall assessment: {assessment} ({success_rate:.1f}% success rate)\n")
                else:
                    # Fallback to success rate based assessment
                    success_rate = summary['success_rate']
                    if success_rate >= 80:
                        assessment = "Excellent performance"
                    elif success_rate >= 60:
                        assessment = "Good performance"
                    elif success_rate >= 40:
                        assessment = "Moderate performance"
                    else:
                        assessment = "Needs improvement"
                    f.write(f"- Overall assessment: {assessment} ({success_rate:.1f}% success rate)\n")

            os.rename(temp_summary, summary_file)

            # Save console log checkpoint
            with open(temp_log, 'w', encoding='utf-8') as f:
                f.write(f"CONSOLE LOG - Run {run_id} - Question {summary.get('current_question', 0)}/{summary.get('total_questions', 0)}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                if agent_config_id:
                    f.write(f"Agent Config ID: {agent_config_id}\n")
                f.write("=" * 80 + "\n\n")
                f.write(console_output)

            os.rename(temp_log, log_file)

        except Exception as e:
            # Clean up temp files if something went wrong
            for temp_file in [temp_csv, temp_summary, temp_log]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            raise e

    def cleanup_temp_files(self, run_id: str) -> None:
        """Clean up temporary files after successful run completion."""
        run_dir = os.path.join(self.base_output_dir, run_id)
        temp_dir = os.path.join(run_dir, "temp")

        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"✓ Cleaned up temporary files for run {run_id}")
            except Exception as e:
                print(f"⚠ Warning: Failed to clean up temp files: {e}")

    def cleanup_checkpoints(self, run_id: str) -> None:
        """Optionally clean up checkpoint files after successful run completion."""
        run_dir = os.path.join(self.base_output_dir, run_id)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")

        if os.path.exists(checkpoints_dir):
            try:
                shutil.rmtree(checkpoints_dir)
                print(f"✓ Cleaned up checkpoint files for run {run_id}")
            except Exception as e:
                print(f"⚠ Warning: Failed to clean up checkpoint files: {e}")

    def save_final_results(self, run_id: str, summary: Dict[str, Any], results: List[Dict[str, Any]], console_output: str = "", agent_config_id: Optional[str] = None) -> None:
        """Save final results with clear, standardized names."""
        run_dir = os.path.join(self.base_output_dir, run_id)
        temp_dir = os.path.join(run_dir, "temp")

        # Include agent config info in summary
        if agent_config_id:
            summary["agent_config_id"] = agent_config_id

        # Final files with clear names
        final_csv_file = os.path.join(run_dir, "FINAL_RESULTS.csv")
        final_summary_file = os.path.join(run_dir, "FINAL_SUMMARY.txt")
        final_output_file = os.path.join(run_dir, "FULL_OUTPUT.txt")

        # Save final CSV with enhanced columns (including agent config)
        temp_csv = os.path.join(temp_dir, "final_results.tmp")
        try:
            with open(temp_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['question_level', 'question_text', 'agent_answer', 'expected_answer', 'is_correct', 'failure_type'])

                for result in results:
                    question_level = result.get('question_level', 'unknown')
                    question_text = result.get('question', '')
                    # Truncate question text for CSV readability
                    if len(question_text) > 100:
                        question_text = question_text[:97] + "..."
                    
                    agent_answer = result['agent_answer'] if result['success'] else result['error']
                    expected_answer = result['expected_answer']
                    is_correct = result['is_correct'] if result['success'] else False
                    failure_type = result.get('failure_type', 'unknown')

                    writer.writerow([question_level, question_text, agent_answer, expected_answer, is_correct, failure_type])

            os.rename(temp_csv, final_csv_file)
        except Exception as e:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            raise e

        # Save comprehensive final summary
        temp_summary = os.path.join(temp_dir, "final_summary.tmp")
        try:
            with open(temp_summary, 'w', encoding='utf-8') as f:
                f.write(generate_comprehensive_summary(summary, results))

            os.rename(temp_summary, final_summary_file)
        except Exception as e:
            if os.path.exists(temp_summary):
                os.remove(temp_summary)
            raise e

        # Save full output using captured console output
        temp_output = os.path.join(temp_dir, "full_output.tmp")
        try:
            with open(temp_output, 'w', encoding='utf-8') as f:
                f.write(console_output)

            os.rename(temp_output, final_output_file)
        except Exception as e:
            if os.path.exists(temp_output):
                os.remove(temp_output)
            raise e

        print(f"\n✓ Final results saved:")
        print(f"  CSV Data: {final_csv_file}")
        print(f"  Summary: {final_summary_file}")
        print(f"  Full Output: {final_output_file}")
        if agent_config_id:
            print(f"  Agent Config: {agent_config_id}")


class AnswerValidation(BaseModel):
    """Response model for answer validation using LLM judge."""
    is_correct: bool
    reasoning: str = Field(description="Detailed explanation of why the answer is correct or incorrect")
    confidence: float = Field(description="Confidence level from 0.0 to 1.0", ge=0.0, le=1.0)


def load_questions_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load questions from JSONL file."""
    questions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def validate_answer_with_llm(
    agent_answer: str,
    expected_answer: str,
    question: str,
    validation_client
) -> Tuple[bool, str]:
    """
    Use LLM to validate if the agent's answer is correct.

    Args:
        agent_answer: The answer provided by the agent
        expected_answer: The expected answer from the dataset
        question: The original question
        validation_client: Client for LLM validation

    Returns:
        Tuple of (is_correct, reasoning)
    """
    validation_prompt = f"""
You are an expert answer evaluator. Compare the agent's answer with the expected answer for the given question.

Question: {question}

Agent's Answer: "{agent_answer}"

Expected Answer: "{expected_answer}"

Evaluate if the agent's answer is correct. Consider:
1. Exact match: Does the agent's answer exactly match the expected answer?
2. Semantic equivalence: Does the agent's answer mean the same thing as the expected answer?
3. Completeness: Does the agent's answer contain all the necessary information?
4. Accuracy: Is the agent's answer factually correct?

Be very strict in your evaluation. The answer must be essentially equivalent to the expected answer to be marked correct.
Minor differences in formatting, capitalization, or punctuation are acceptable, but the core meaning must be the same.

Examples:
- Agent: "Paris" vs Expected: "Paris" → CORRECT (exact match)
- Agent: "paris" vs Expected: "Paris" → CORRECT (case difference)  
- Agent: "The capital is Paris" vs Expected: "Paris" → CORRECT (same core answer)
- Agent: "France" vs Expected: "Paris" → INCORRECT (different answer)
- Agent: "I don't know" vs Expected: "Paris" → INCORRECT (no answer provided)

Return your evaluation with confidence level.
"""

    try:
        validation = validation_client.chat.completions.create(
            messages=[{"role": "user", "content": validation_prompt}],
            response_model=AnswerValidation,
            extra_body={"provider": {"require_parameters": True}}
        )
        
        # Add additional check for very low confidence
        if validation.confidence < 0.7:
            return False, f"Low confidence validation: {validation.reasoning}"
            
        return validation.is_correct, validation.reasoning
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def run_single_test(
    question_data: Dict[str, Any],
    agent_client,
    validation_client,
    max_steps: int = 10
) -> Dict[str, Any]:
    """
    Run a single test case.

    Args:
        question_data: Dictionary containing question and expected answer
        agent_client: Client for running the agent
        validation_client: Client for validating answers
        max_steps: Maximum steps for the agent

    Returns:
        Dictionary with test results
    """
    query = question_data["Question"]
    expected_answer = question_data["Final answer"]
    task_id = question_data.get("task_id", "unknown")
    question_level = question_data.get("Level", "unknown")

    result = {
        "task_id": task_id,
        "question": query,
        "question_level": question_level,
        "expected_answer": expected_answer,
        "agent_answer": None,
        "is_correct": False,
        "success": False,
        "error": None,
        "execution_time": None,
        "steps_taken": None,
        "validation_reasoning": None,
        "failure_type": None,
        "timestamp": datetime.now().isoformat()
    }

    try:
        # Run the agent
        start_time = datetime.now()
        agent_answer, steps_taken = run_react_loop(
            query,
            agent_client,
            user_config={"max_steps": max_steps, 'verbosity': VERBOSITY_STANDARD}
        )
        end_time = datetime.now()

        # Check if max steps was reached
        if "Max steps" in agent_answer:
            result["error"] = f"Max steps ({max_steps}) reached without answer"
            result["steps_taken"] = max_steps
            result["failure_type"] = "max_steps_reached"
        else:
            result["agent_answer"] = agent_answer
            result["steps_taken"] = steps_taken
            result["success"] = True
            result["failure_type"] = "success"

        result["execution_time"] = (end_time - start_time).total_seconds()

    except Exception as e:
        end_time = datetime.now()
        result["error"] = str(e)
        result["execution_time"] = (end_time - start_time).total_seconds()
        result["traceback"] = traceback.format_exc()
        result["failure_type"] = "crashed"

    # Validate answer if agent succeeded
    if result["success"] and result["agent_answer"]:
        is_correct, validation_reasoning = validate_answer_with_llm(
            result["agent_answer"],
            expected_answer,
            query,
            validation_client
        )
        result["is_correct"] = is_correct
        result["validation_reasoning"] = validation_reasoning

    return result




def print_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted summary of test results."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    print(f"Total questions:           {summary['total_questions']}")
    print(f"Successfully answered:     {summary['successful_runs']}")
    print(f"Failed to answer:          {summary['failed_runs']}")
    print(f"Success rate:              {summary['success_rate']:.1f}%")
    print(f"Correct answers:           {summary['correct_answers']}")
    print(f"Incorrect answers:         {summary['incorrect_answers']}")
    if summary['successful_runs'] > 0:
        print(f"Accuracy rate:             {summary['accuracy_rate']:.1f}%")
    print(f"Total execution time:      {summary['total_execution_time']:.2f}s")
    print(f"Average time per question: {summary['average_execution_time']:.2f}s")
    if summary['successful_runs'] > 0:
        print(f"Total steps taken:         {summary['total_steps']}")
        print(f"Average steps per question:{summary['average_steps']:.1f}")

    # Print failure breakdown
    if summary.get('failure_breakdown'):
        print("\n" + "=" * 80)
        print("FAILURE BREAKDOWN")
        print("=" * 80)
        crashed = summary['failure_breakdown'].get('crashed', 0)
        max_steps = summary['failure_breakdown'].get('max_steps_reached', 0)
        total_failed = crashed + max_steps
        
        if total_failed > 0:
            print(f"Crashed/Errors:           {crashed} ({(crashed/summary['total_questions']*100):.1f}%)")
            print(f"Max steps reached:       {max_steps} ({(max_steps/summary['total_questions']*100):.1f}%)")

    # Print level-specific statistics
    if summary.get('level_stats'):
        print("\n" + "=" * 80)
        print("PERFORMANCE BY QUESTION LEVEL")
        print("=" * 80)
        
        for level, stats in sorted(summary['level_stats'].items()):
            level_total = stats['total']
            level_success = stats['successful_runs']
            level_correct = stats['correct_answers']
            success_rate = (level_success / level_total * 100) if level_total > 0 else 0
            accuracy_rate = (level_correct / level_total * 100) if level_total > 0 else 0
            
            print(f"Level {level} ({level_total} questions): {success_rate:.1f}% success rate, {accuracy_rate:.1f}% accuracy rate")

    # Add score-based assessment if we have an agent config
    if summary.get('agent_config_id'):
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from config_manager import AgentConfigManager
            config_manager = AgentConfigManager()
            score = config_manager.calculate_score(summary)

            # Provide assessment based on score
            if score >= 1.5:
                assessment = "Outstanding performance"
            elif score >= 1.0:
                assessment = "Very good performance"
            elif score >= 0.5:
                assessment = "Good performance"
            elif score >= 0.2:
                assessment = "Moderate performance"
            else:
                assessment = "Needs significant improvement"

            print("\n" + "=" * 80)
            print("SCORE-BASED ASSESSMENT")
            print("=" * 80)
            print(f"Overall score:             {score:.3f}")
            print(f"Performance assessment:   {assessment}")

        except Exception:
            pass

def print_question_table(results: List[Dict[str, Any]]) -> None:
    """Print a detailed table of all questions with their results."""
    print("\n" + "=" * 80)
    print("DETAILED QUESTION RESULTS TABLE")
    print("=" * 80)
    print(f"{'Level':<6} | {'Question #':<11} | {'Status':<16} | {'Time(s)':<8} | {'Steps':<6} | {'Question Preview'}")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        level = str(result.get('question_level', 'unknown'))
        question_num = str(i)
        
        # Determine status
        if result['success']:
            if result['is_correct']:
                status = "Correct"
            else:
                status = "Incorrect"
        else:
            failure_type = result.get('failure_type', 'unknown')
            if failure_type == 'max_steps_reached':
                status = "Max Steps"
            elif failure_type == 'crashed':
                status = "Crashed"
            else:
                status = "Failed"
        
        time_str = f"{result['execution_time']:.1f}" if result['execution_time'] else "N/A"
        steps_str = str(result['steps_taken']) if result['steps_taken'] else "N/A"
        
        # Truncate question for preview
        question_preview = result.get('question', '')[:50]
        if len(result.get('question', '')) > 50:
            question_preview += "..."
        
        print(f"{level:<6} | {question_num:<11} | {status:<16} | {time_str:<8} | {steps_str:<6} | {question_preview}")
    
    print("-" * 80)
    print(f"Total: {len(results)} questions")

def format_failure_breakdown(summary: Dict[str, Any]) -> str:
    """Format failure breakdown section for file output."""
    if not summary.get('failure_breakdown'):
        return ""
    
    crashed = summary['failure_breakdown'].get('crashed', 0)
    max_steps = summary['failure_breakdown'].get('max_steps_reached', 0)
    total_failed = crashed + max_steps
    
    if total_failed == 0:
        return ""
    
    output = []
    output.append("\n" + "=" * 80)
    output.append("FAILURE BREAKDOWN")
    output.append("=" * 80)
    output.append(f"Crashed/Errors:           {crashed} ({(crashed/summary['total_questions']*100):.1f}%)")
    output.append(f"Max steps reached:       {max_steps} ({(max_steps/summary['total_questions']*100):.1f}%)")
    
    return "\n".join(output)


def format_level_statistics(summary: Dict[str, Any]) -> str:
    """Format level-specific statistics section for file output."""
    if not summary.get('level_stats'):
        return ""
    
    output = []
    output.append("\n" + "=" * 80)
    output.append("PERFORMANCE BY QUESTION LEVEL")
    output.append("=" * 80)
    
    for level, stats in sorted(summary['level_stats'].items()):
        level_total = stats['total']
        level_success = stats['successful_runs']
        level_correct = stats['correct_answers']
        success_rate = (level_success / level_total * 100) if level_total > 0 else 0
        accuracy_rate = (level_correct / level_total * 100) if level_total > 0 else 0
        
        output.append(f"Level {level} ({level_total} questions): {success_rate:.1f}% success rate, {accuracy_rate:.1f}% accuracy rate")
    
    return "\n".join(output)


def format_question_table(results: List[Dict[str, Any]]) -> str:
    """Format detailed question table for file output."""
    output = []
    output.append("\n" + "=" * 80)
    output.append("DETAILED QUESTION RESULTS TABLE")
    output.append("=" * 80)
    output.append(f"{'Level':<6} | {'Question #':<11} | {'Status':<16} | {'Time(s)':<8} | {'Steps':<6} | {'Question Preview'}")
    output.append("-" * 80)
    
    for i, result in enumerate(results, 1):
        level = str(result.get('question_level', 'unknown'))
        question_num = str(i)
        
        # Determine status
        if result['success']:
            if result['is_correct']:
                status = "Correct"
            else:
                status = "Incorrect"
        else:
            failure_type = result.get('failure_type', 'unknown')
            if failure_type == 'max_steps_reached':
                status = "Max Steps"
            elif failure_type == 'crashed':
                status = "Crashed"
            else:
                status = "Failed"
        
        time_str = f"{result['execution_time']:.1f}" if result['execution_time'] else "N/A"
        steps_str = str(result['steps_taken']) if result['steps_taken'] else "N/A"
        
        # Truncate question for preview
        question_preview = result.get('question', '')[:50]
        if len(result.get('question', '')) > 50:
            question_preview += "..."
        
        output.append(f"{level:<6} | {question_num:<11} | {status:<16} | {time_str:<8} | {steps_str:<6} | {question_preview}")
    
    output.append("-" * 80)
    output.append(f"Total: {len(results)} questions")
    
    return "\n".join(output)


def generate_comprehensive_summary(summary: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    """Generate comprehensive summary combining basic statistics and detailed analysis."""
    output = []
    output.append("FINAL TEST RESULTS")
    output.append("=" * 80)
    output.append(f"Run ID: {summary.get('run_id', 'unknown')}")
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add agent configuration information if available
    if summary.get('agent_config_id'):
        output.append(f"Agent Config ID: {summary['agent_config_id']}")
    
    output.append("")
    
    # Basic statistics section
    output.append("BASIC STATISTICS")
    output.append("=" * 80)
    output.append(f"Total questions:           {summary['total_questions']}")
    output.append(f"Successfully answered:     {summary['successful_runs']}")
    output.append(f"Failed to answer:          {summary['failed_runs']}")
    output.append(f"Success rate:              {summary['success_rate']:.1f}%")
    output.append(f"Correct answers:           {summary['correct_answers']}")
    output.append(f"Incorrect answers:         {summary['incorrect_answers']}")
    if summary['successful_runs'] > 0:
        output.append(f"Accuracy rate:             {summary['accuracy_rate']:.1f}%")
    output.append(f"Total execution time:      {summary['total_execution_time']:.2f}s")
    output.append(f"Average time per question: {summary['average_execution_time']:.2f}s")
    if summary['successful_runs'] > 0:
        output.append(f"Total steps taken:         {summary['total_steps']}")
        output.append(f"Average steps per question:{summary['average_steps']:.1f}")
    
    # Calculate and display overall score if we have an agent config
    if summary.get('agent_config_id'):
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from config_manager import AgentConfigManager
            config_manager = AgentConfigManager()
            score = config_manager.calculate_score(summary)
            output.append(f"Overall score:             {score:.3f}")
        except Exception:
            output.append(f"Overall score:             Unable to calculate")
    
    # Add detailed question table
    output.append("")
    output.append(format_question_table(results))
    
    # Add failure breakdown
    output.append(format_failure_breakdown(summary))
    
    # Add level statistics
    output.append(format_level_statistics(summary))
    
    # Add key insights
    output.append("")
    output.append("KEY INSIGHTS")
    output.append("=" * 80)
    
    # Performance insights
    if summary.get('level_stats'):
        best_level = None
        worst_level = None
        best_success_rate = -1
        worst_success_rate = 101
        
        for level, stats in summary['level_stats'].items():
            success_rate = (stats['successful_runs'] / stats['total'] * 100) if stats['total'] > 0 else 0
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_level = level
            if success_rate < worst_success_rate:
                worst_success_rate = success_rate
                worst_level = level
        
        if best_level and worst_level and best_level != worst_level:
            output.append(f"- Agent performs best on Level {best_level} questions ({best_success_rate:.1f}% success rate)")
            output.append(f"- Agent struggles most with Level {worst_level} questions ({worst_success_rate:.1f}% success rate)")
    
    # Failure analysis
    failure_breakdown = summary.get('failure_breakdown', {})
    if failure_breakdown:
        total_failures = sum(failure_breakdown.values())
        if total_failures > 0:
            primary_failure = max(failure_breakdown.items(), key=lambda x: x[1])
            failure_percent = (primary_failure[1] / summary['total_questions'] * 100)
            output.append(f"- Primary failure mode: {primary_failure[0].replace('_', ' ').title()} ({failure_percent:.1f}% of questions)")
    
    # Timing insights
    if summary['successful_runs'] > 0:
        avg_time = summary['average_execution_time']
        avg_steps = summary['average_steps']
        output.append(f"- Average time per successful question: {avg_time:.1f}s")
        output.append(f"- Average steps per successful question: {avg_steps:.1f}")
    
    # Overall assessment
    success_rate = summary['success_rate']
    if success_rate >= 80:
        assessment = "Excellent performance"
    elif success_rate >= 60:
        assessment = "Good performance"
    elif success_rate >= 40:
        assessment = "Moderate performance"
    else:
        assessment = "Needs improvement"
    
    output.append(f"- Overall assessment: {assessment} ({success_rate:.1f}% success rate)")
    
    # Add score-based insights if we have an agent config
    if summary.get('agent_config_id'):
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from config_manager import AgentConfigManager
            config_manager = AgentConfigManager()
            score = config_manager.calculate_score(summary)
            
            # Provide additional insights based on score
            if score >= 1.5:
                output.append(f"- Score-based assessment: Outstanding performance ({score:.3f})")
            elif score >= 1.0:
                output.append(f"- Score-based assessment: Very good performance ({score:.3f})")
            elif score >= 0.5:
                output.append(f"- Score-based assessment: Good performance ({score:.3f})")
            elif score >= 0.2:
                output.append(f"- Score-based assessment: Moderate performance ({score:.3f})")
            else:
                output.append(f"- Score-based assessment: Needs significant improvement ({score:.3f})")
        except Exception:
            output.append(f"- Score-based assessment: Unable to calculate")
    
    return "\n".join(output)


def print_detailed_results(results: List[Dict[str, Any]], show_validation_details: bool = False) -> None:
    """Print detailed results for all test cases."""
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for result in results:
        print(f"\n--- Task {result['task_id']} ---")
        print(f"Status: {' SUCCESS' if result['success'] else ' FAILED'}")
        if result["success"]:
            print(f"Correct: {' YES' if result['is_correct'] else ' NO'}")
        print(f"Time: {result['execution_time']:.2f}s")
        print(f"Steps: {result['steps_taken']}")
        print(f"Question: {result['question']}")
        print(f"Expected: {result['expected_answer']}")

        if result['success']:
            print(f"Agent Answer: {result['agent_answer']}")
            if show_validation_details and result['validation_reasoning']:
                print(f"Validation: {result['validation_reasoning']}")
        else:
            print(f"Error: {result['error']}")
            if 'traceback' in result:
                print("Traceback:")
                tb_lines = result['traceback'].split('\n')
                for line in tb_lines[-5:]:
                    if line.strip():
                        print(f"  {line}")

        print("-" * 60)


def list_available_runs(output_dir: str = "testing/results") -> None:
    """List all available runs with their status."""
    run_manager = RunManager(output_dir)
    runs = run_manager.list_runs()

    if not runs:
        print("No runs found.")
        return

    print("\n" + "=" * 80)
    print("AVAILABLE RUNS")
    print("=" * 80)

    for run in runs:
        status = "✓ COMPLETE" if run["is_complete"] else f"⏸ INCOMPLETE ({run['current_question']}/{run['total_questions']})"
        print(f"Run ID: {run['run_id']}")
        print(f"Status:  {status}")
        print(f"Path:    {run['path']}")
        if run["start_time"]:
            print(f"Started: {run['start_time']}")
        print("-" * 60)


def get_score_assessment(score: Optional[float]) -> str:
    """Get performance assessment based on score."""
    if score is None:
        return "Not tested"
    elif score >= 1.5:
        return "Outstanding performance"
    elif score >= 1.0:
        return "Very good performance"
    elif score >= 0.5:
        return "Good performance"
    elif score >= 0.2:
        return "Moderate performance"
    else:
        return "Needs significant improvement"


def list_configurations_sorted_by_score(config_manager: AgentConfigManager) -> None:
    """List all agent configurations sorted by best score (highest to lowest)."""
    configs = config_manager.list_configs()

    if not configs:
        print("No agent configurations found.")
        return

    # Sort by best_score (highest first), None values last
    sorted_configs = sorted(
        configs,
        key=lambda x: (x.get('best_score') is None, x.get('best_score') or 0),
        reverse=True
    )

    print("\n" + "=" * 100)
    print("AGENT CONFIGURATIONS (Sorted by Best Score)")
    print("=" * 100)

    for i, config in enumerate(sorted_configs, 1):
        best_score = config.get('best_score')
        best_score_str = f'{best_score:.3f}' if best_score is not None else 'Not tested'
        assessment = get_score_assessment(best_score)
        best_run_id = config.get('best_run_id', 'N/A')
        total_runs = config.get('total_runs', 0)
        last_run = config.get('last_run_at', 'Never')

        print(f"{i:2d}. {config['name']}")
        print(f"    ID:           {config['id']}")
        print(f"    Model:        {config['model']}")
        print(f"    Max Steps:    {config['max_steps']}")
        print(f"    Best Score:   {best_score_str} ({assessment})")
        print(f"    Best Run ID:  {best_run_id}")
        print(f"    Total Runs:   {total_runs}")
        print(f"    Last Run:     {last_run}")
        print(f"    Description:  {config['description']}")
        print("-" * 100)

    print(f"\nTotal configurations: {len(sorted_configs)}")
    tested_configs = sum(1 for c in sorted_configs if c.get('best_score') is not None)
    print(f"Tested configurations: {tested_configs}")
    if tested_configs > 0:
        avg_score = sum(c.get('best_score', 0) for c in sorted_configs if c.get('best_score') is not None) / tested_configs
        print(f"Average best score: {avg_score:.3f}")


def list_runs_sorted_by_score(output_dir: str = "testing/results") -> None:
    """List all runs sorted by score (highest to lowest)."""
    import re

    def extract_score_from_summary(summary_file):
        """Extract score from FINAL_SUMMARY.txt."""
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
            # Look for 'Overall score:' line
            match = re.search(r'Overall score:\s*([\d.]+)', content)
            if match:
                return float(match.group(1))
        except:
            pass
        return None

    def extract_config_id_from_summary(summary_file):
        """Extract config ID from FINAL_SUMMARY.txt."""
        try:
            with open(summary_file, 'r') as f:
                content = f.read()
            # Look for 'Agent Config ID:' line
            match = re.search(r'Agent Config ID:\s*(\w+)', content)
            if match:
                return match.group(1)
        except:
            pass
        return None

    def get_run_info(run_dir):
        """Extract run information from directory."""
        run_id = os.path.basename(run_dir)
        summary_file = os.path.join(run_dir, 'FINAL_SUMMARY.txt')
        checkpoint_file = os.path.join(run_dir, 'checkpoint.json')

        info = {
            'run_id': run_id,
            'score': None,
            'config_id': None,
            'complete': False,
            'path': run_dir
        }

        # Extract info from summary file
        if os.path.exists(summary_file):
            info['score'] = extract_score_from_summary(summary_file)
            info['config_id'] = extract_config_id_from_summary(summary_file)
            info['complete'] = True

        # Extract config ID from checkpoint if not found in summary
        elif os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                info['config_id'] = checkpoint.get('agent_config_id')
                info['complete'] = checkpoint.get('is_complete', False)
            except:
                pass

        return info

    if not os.path.exists(output_dir):
        print(f"Results directory not found: {output_dir}")
        return

    run_dirs = [d for d in os.listdir(output_dir) if d.startswith('run_') and os.path.isdir(os.path.join(output_dir, d))]
    runs_info = []

    for run_dir_name in run_dirs:
        run_path = os.path.join(output_dir, run_dir_name)
        run_info = get_run_info(run_path)
        runs_info.append(run_info)

    if not runs_info:
        print("No runs found.")
        return

    # Sort by score (highest first), None values last
    sorted_runs = sorted(
        runs_info,
        key=lambda x: (x['score'] is None, -(x['score'] or 0))
    )

    print("\n" + "=" * 120)
    print("TEST RUNS (Sorted by Score)")
    print("=" * 120)

    for i, run in enumerate(sorted_runs, 1):
        score_str = f'{run["score"]:.3f}' if run['score'] is not None else 'No score'
        status = '✓ COMPLETE' if run['complete'] else '⏸ INCOMPLETE'
        config_id = run['config_id'] or 'Unknown'

        print(f"{i:2d}. Run ID: {run['run_id']}")
        print(f"    Status:       {status}")
        print(f"    Score:        {score_str}")
        print(f"    Config ID:    {config_id}")
        print(f"    Path:         {run['path']}")
        print("-" * 120)

    completed_runs = sum(1 for r in sorted_runs if r['complete'])
    scored_runs = sum(1 for r in sorted_runs if r['score'] is not None)

    print(f"\nTotal runs found: {len(sorted_runs)}")
    print(f"Completed runs: {completed_runs}")
    print(f"Runs with scores: {scored_runs}")


def print_system_summary(config_manager: AgentConfigManager, output_dir: str = "testing/results") -> None:
    """Print a combined summary of configurations and their best runs."""
    import re

    configs = config_manager.list_configs()

    if not configs:
        print("No agent configurations found.")
        return

    # Sort configurations by best score
    sorted_configs = sorted(
        configs,
        key=lambda x: (x.get('best_score') is None, x.get('best_score') or 0),
        reverse=True
    )

    print("\n" + "=" * 120)
    print("SYSTEM PERFORMANCE SUMMARY")
    print("=" * 120)

    for i, config in enumerate(sorted_configs, 1):
        best_score = config.get('best_score')
        best_score_str = f'{best_score:.3f}' if best_score is not None else 'Not tested'
        assessment = get_score_assessment(best_score)
        best_run_id = config.get('best_run_id')

        print(f"{i}. CONFIG: {config['name']}")
        print(f"   Config ID:     {config['id']}")
        print(f"   Model:         {config['model']}")
        print(f"   Max Steps:     {config['max_steps']}")
        print(f"   Best Score:    {best_score_str} ({assessment})")

        if best_run_id:
            print(f"   Best Run ID:   {best_run_id}")

            # Try to get more info about the best run
            run_path = os.path.join(output_dir, best_run_id)
            summary_file = os.path.join(run_path, 'FINAL_SUMMARY.txt')

            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        content = f.read()

                    # Extract key stats
                    success_match = re.search(r'Success rate:\s*([\d.]+)%', content)
                    accuracy_match = re.search(r'Accuracy rate:\s*([\d.]+)%', content)
                    questions_match = re.search(r'Total questions:\s*(\d+)', content)

                    if success_match:
                        print(f"   Success Rate:  {success_match.group(1)}%")
                    if accuracy_match:
                        print(f"   Accuracy Rate: {accuracy_match.group(1)}%")
                    if questions_match:
                        print(f"   Questions:     {questions_match.group(1)}")

                except:
                    print(f"   Run details:   Could not read summary file")
            else:
                print(f"   Run details:   Summary file not found")
        else:
            print(f"   Best Run ID:   No runs yet")

        print(f"   Total Runs:   {config.get('total_runs', 0)}")
        print(f"   Last Run:     {config.get('last_run_at', 'Never')}")
        print("-" * 120)

    # Overall statistics
    total_configs = len(sorted_configs)
    tested_configs = sum(1 for c in sorted_configs if c.get('best_score') is not None)

    print(f"\nOVERALL STATISTICS")
    print(f"Total configurations: {total_configs}")
    print(f"Tested configurations: {tested_configs}")

    if tested_configs > 0:
        scores = [c.get('best_score', 0) for c in sorted_configs if c.get('best_score') is not None]
        avg_score = sum(scores) / len(scores)
        best_overall = max(scores)
        print(f"Average best score: {avg_score:.3f}")
        print(f"Best overall score: {best_overall:.3f}")

        # Best performing config
        best_config = sorted_configs[0]
        print(f"Best config: {best_config['name']} ({best_config['id']})")

    # Run statistics
    all_runs = []
    if os.path.exists(output_dir):
        run_dirs = [d for d in os.listdir(output_dir) if d.startswith('run_') and os.path.isdir(os.path.join(output_dir, d))]
        all_runs = run_dirs

    completed_runs = 0
    scored_runs = 0

    for run_dir in all_runs:
        run_path = os.path.join(output_dir, run_dir)
        summary_file = os.path.join(run_path, 'FINAL_SUMMARY.txt')
        if os.path.exists(summary_file):
            completed_runs += 1
            scored_runs += 1

    print(f"\nTotal runs: {len(all_runs)}")
    print(f"Completed runs: {completed_runs}")
    print(f"Runs with scores: {scored_runs}")




def select_agent_config_interactive(config_manager: AgentConfigManager) -> Optional[str]:
    """Interactive agent configuration selection."""
    configs = config_manager.list_configs()

    if not configs:
        print("No agent configurations found. Please create one first.")
        return None

    print("\n" + "=" * 80)
    print("AVAILABLE AGENT CONFIGURATIONS")
    print("=" * 80)

    for i, config in enumerate(configs, 1):
        status = f"Best: {config['best_score']:.3f}" if config['best_score'] else "Not tested"
        total_runs = config.get('total_runs', 0)
        print(f"{i}. {config['name']}")
        print(f"   Model: {config['model']}")
        print(f"   Max Steps: {config['max_steps']}")
        print(f"   Description: {config['description']}")
        print(f"   Status: {status}")
        print(f"   Runs: {total_runs}")
        print("-" * 40)

    while True:
        try:
            choice = input(f"\nSelect configuration (1-{len(configs)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(configs):
                selected_config = configs[choice_idx]
                print(f"\nSelected: {selected_config['name']}")
                confirm = input("Confirm selection? (y/n): ").strip().lower()
                if confirm == 'y':
                    return selected_config['id']
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")


def select_run_to_resume_interactive(run_manager: RunManager) -> Optional[str]:
    """Interactive run selection for resuming."""
    runs = run_manager.list_runs()
    incomplete_runs = [run for run in runs if not run["is_complete"]]

    if not incomplete_runs:
        print("No incomplete runs found.")
        return None

    print("\n" + "=" * 80)
    print("INCOMPLETE RUNS TO RESUME")
    print("=" * 80)

    for i, run in enumerate(incomplete_runs, 1):
        print(f"{i}. Run ID: {run['run_id']}")
        print(f"   Progress: {run['current_question']}/{run['total_questions']} questions")
        print(f"   Started: {run.get('start_time', 'Unknown')}")
        print("-" * 40)

    while True:
        try:
            choice = input(f"\nSelect run to resume (1-{len(incomplete_runs)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(incomplete_runs):
                selected_run = incomplete_runs[choice_idx]
                print(f"\nSelected run: {selected_run['run_id']}")
                confirm = input("Confirm selection? (y/n): ").strip().lower()
                if confirm == 'y':
                    return selected_run['run_id']
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")


def display_manual_validation(
    question_data: Dict[str, Any],
    result: Dict[str, Any]
) -> None:
    """
    Display validation results for manual review.

    Args:
        question_data: Original question data
        result: Test result containing agent answer and validation
    """
    print("\n" + "="*80)
    print("MANUAL VALIDATION REVIEW")
    print("="*80)

    # Display question info
    print(f"\n📋 QUESTION {str(question_data.get('Level', 'Unknown')).upper()}")
    print(f"Question: {question_data['Question']}")
    print(f"Task ID: {question_data.get('task_id', 'Unknown')}")

    # Display agent answer
    print(f"\n🤖 AGENT'S ANSWER:")
    if result['success']:
        print(f"{result['agent_answer']}")
        print(f"Steps taken: {result['steps_taken']}")
        print(f"Execution time: {result['execution_time']:.2f}s")
    else:
        print(f"❌ AGENT FAILED: {result.get('error', 'Unknown error')}")
        print(f"Failure type: {result.get('failure_type', 'Unknown')}")
        if result.get('steps_taken'):
            print(f"Steps taken: {result['steps_taken']}")
        if result.get('execution_time'):
            print(f"Execution time: {result['execution_time']:.2f}s")

    # Display expected answer
    print(f"\n✅ EXPECTED ANSWER:")
    print(f"{question_data['Final answer']}")

    # Display validator decision
    print(f"\n🔍 VALIDATOR DECISION:")
    if result['success'] and result.get('validation_reasoning'):
        is_correct = result['is_correct']
        status = "CORRECT" if is_correct else "INCORRECT"
        status_emoji = "✅" if is_correct else "❌"
        print(f"{status_emoji} Status: {status}")
        print(f"Reasoning: {result['validation_reasoning']}")
    else:
        print("⚠️ No validation performed (agent failed)")

    print("\n" + "="*80)


def get_manual_decision() -> str:
    """
    Get user's decision for manual validation mode.

    Returns:
        User's choice: 'overwrite', 'redo', 'continue', or 'quit'
    """
    print("\n🎯 MANUAL VALIDATION OPTIONS:")
    print("1. 📝 Overwrite validator's decision")
    print("2. 🔄 Redo current question")
    print("3. ⏭️  Continue to next question")
    print("4. 🚪 Quit testing")

    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()

            if choice == '1':
                return 'overwrite'
            elif choice == '2':
                return 'redo'
            elif choice == '3':
                return 'continue'
            elif choice == '4':
                return 'quit'
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
        except (EOFError, KeyboardInterrupt):
            print("\n\n⚠️ Input interrupted. Continuing to next question...")
            return 'continue'


def handle_overwrite_validation(result: Dict[str, Any]) -> bool:
    """
    Handle user overwriting the validator's decision.

    Args:
        result: Test result to modify

    Returns:
        New is_correct value chosen by user
    """
    current_status = "CORRECT" if result['is_correct'] else "INCORRECT"
    print(f"\n📝 Current validator decision: {current_status}")

    while True:
        try:
            choice = input("Mark as (c)orrect, (i)ncorrect, or (esc)ape: ").strip().lower()

            if choice in ['c', 'correct']:
                print("✅ Marked as CORRECT")
                result['is_correct'] = True
                result['validation_reasoning'] = "Manually marked as correct by user"
                return True
            elif choice in ['i', 'incorrect']:
                print("❌ Marked as INCORRECT")
                result['is_correct'] = False
                result['validation_reasoning'] = "Manually marked as incorrect by user"
                return False
            elif choice in ['esc', 'escape', '']:
                print("↩️ Keeping validator's original decision")
                return result['is_correct']
            else:
                print("Invalid choice. Please enter 'c', 'i', or 'esc'.")
        except (EOFError, KeyboardInterrupt):
            print("\n↩️ Keeping validator's original decision")
            return result['is_correct']


def handle_redo_question(
    question_data: Dict[str, Any],
    agent_client,
    validation_client,
    max_steps: int
) -> Dict[str, Any]:
    """
    Re-run a question with the same parameters.

    Args:
        question_data: Question data to retry
        agent_client: Agent client for running the test
        validation_client: Validation client
        max_steps: Maximum steps for agent

    Returns:
        New test result
    """
    print("\n🔄 Re-running question...")

    # Run the test again
    new_result = run_single_test(
        question_data=question_data,
        agent_client=agent_client,
        validation_client=validation_client,
        max_steps=max_steps
    )

    print(f"✅ Question re-run completed")
    return new_result


def toggle_searxng_restart(current_setting: bool) -> bool:
    """
    Toggle searxng restart setting during manual mode.

    Args:
        current_setting: Current restart_searxng setting

    Returns:
        New restart setting
    """
    status = "ENABLED" if current_setting else "DISABLED"
    print(f"\n🔧 Searxng restart is currently {status}")

    while True:
        try:
            choice = input("Toggle restart? (y/n/current): ").strip().lower()

            if choice in ['y', 'yes']:
                new_setting = not current_setting
                new_status = "ENABLED" if new_setting else "DISABLED"
                print(f"🔧 Searxng restart {new_status}")
                return new_setting
            elif choice in ['n', 'no', 'current']:
                print(f"🔧 Keeping current setting ({status})")
                return current_setting
            elif choice == '':
                print(f"🔧 Keeping current setting ({status})")
                return current_setting
            else:
                print("Invalid choice. Please enter 'y', 'n', or just press Enter.")
        except (EOFError, KeyboardInterrupt):
            print(f"\n🔧 Keeping current setting ({status})")
            return current_setting


def run_complete_test(
    questions_file: str,
    use_lmstudio: bool = True,
    max_steps: int = 10,
    limit: Optional[int] = None,
    show_validation_details: bool = False,
    output_dir: str = "testing/results",
    resume_mode: bool = False,
    run_id: Optional[str] = None,
    list_runs: bool = False,
    cleanup_checkpoints: bool = False,
    agent_config_id: Optional[str] = None,
    create_config: bool = False,
    config_name: str = "",
    config_model: str = "",
    config_description: str = "",
    custom_params: str = "",
    list_configs: bool = False,
    list_configs_sorted: bool = False,
    list_runs_sorted: bool = False,
    list_summary: bool = False,
    restart_searxng: bool = False,
    containers_to_restart: List[str] = None,
    manual_mode: bool = False
) -> None:
    """
    Main function to run complete testing with agent configuration management.

    Args:
        questions_file: Path to questions JSONL file
        use_lmstudio: Whether to use LM Studio
        max_steps: Maximum steps for agent (overridden by config if agent_config_id provided)
        limit: Limit number of questions to test
        show_validation_details: Show detailed validation information
        output_dir: Output directory for results
        resume_mode: Whether to resume an incomplete run
        run_id: Specific run ID to resume (if resume_mode=True)
        list_runs: Whether to list available runs
        cleanup_checkpoints: Whether to clean up checkpoints after completion
        agent_config_id: Agent configuration ID to use
        create_config: Whether to create a new configuration
        config_name: Name for new configuration
        config_model: Model for new configuration
        config_description: Description for new configuration
        custom_params: Custom parameters for new configuration
        list_configs: Whether to list available configurations
        list_configs_sorted: Whether to list configurations sorted by score
        list_runs_sorted: Whether to list runs sorted by score
        list_summary: Whether to show system performance summary
        restart_searxng: Whether to restart search containers before each question
        containers_to_restart: List of container names to restart (default: ["redis", "searxng", "caddy"])
        manual_mode: Whether to enable manual validation mode where user can review and override validation decisions
    """

    # Initialize managers
    config_manager = AgentConfigManager()
    run_manager = RunManager(output_dir)

    # Handle configuration management
    if create_config:
        if not all([config_name, config_model]):
            print("Error: config_name and config_model are required when creating a config.")
            return

        print(f"Creating new agent configuration...")
        new_config_id = config_manager.create_config(
            name=config_name,
            model=config_model,
            max_steps=max_steps,
            description=config_description,
            custom_params=custom_params
        )
        print(f"✓ Created configuration: {new_config_id}")
        print(f"  Name: {config_name}")
        print(f"  Model: {config_model}")
        print(f"  Max Steps: {max_steps}")
        if config_description:
            print(f"  Description: {config_description}")
        return

    # Handle listing configurations
    if list_configs:
        configs = config_manager.list_configs()
        if not configs:
            print("No agent configurations found.")
        else:
            print("\n" + "=" * 80)
            print("AGENT CONFIGURATIONS")
            print("=" * 80)
            for config in configs:
                status = f"Best: {config['best_score']:.3f}" if config['best_score'] else "Not tested"
                total_runs = config.get('total_runs', 0)
                print(f"ID: {config['id']}")
                print(f"Name: {config['name']}")
                print(f"Model: {config['model']}")
                print(f"Max Steps: {config['max_steps']}")
                print(f"Description: {config['description']}")
                print(f"Performance: {status}")
                print(f"Total Runs: {total_runs}")
                print("-" * 40)
        return

    # Handle listing runs
    if list_runs:
        list_available_runs(output_dir)
        return

    # Handle listing configurations sorted by score
    if list_configs_sorted:
        list_configurations_sorted_by_score(config_manager)
        return

    # Handle listing runs sorted by score
    if list_runs_sorted:
        list_runs_sorted_by_score(output_dir)
        return

    # Handle system summary
    if list_summary:
        print_system_summary(config_manager, output_dir)
        return

    # Handle run resumption
    if resume_mode:
        if run_id:
            # Use specific run ID
            checkpoint = run_manager.load_checkpoint(run_id)
            if not checkpoint:
                print(f"Error: Run {run_id} not found or has no checkpoint.")
                return
            agent_config_id = checkpoint.get("agent_config_id")
            if agent_config_id:
                print(f"Resuming run {run_id} with saved configuration...")
        else:
            # Interactive selection
            run_id = select_run_to_resume_interactive(run_manager)
            if not run_id:
                print("No run selected for resuming.")
                return

            checkpoint = run_manager.load_checkpoint(run_id)
            if checkpoint:
                agent_config_id = checkpoint.get("agent_config_id")
                print(f"Resuming run {run_id} with saved configuration...")

    # Handle new run with configuration selection (only if not resuming)
    elif not resume_mode and not agent_config_id:
        agent_config_id = select_agent_config_interactive(config_manager)
        if not agent_config_id:
            print("No configuration selected. Exiting.")
            return

    # Load the agent configuration
    if agent_config_id:
        config = config_manager.get_config(agent_config_id)
        if not config:
            print(f"Error: Agent configuration {agent_config_id} not found.")
            return

        # Use config parameters
        model_name = config["model"]
        max_steps = config["max_steps"]

        print(f"\nUsing configuration: {config['name']}")
        print(f"Model: {model_name}")
        print(f"Max Steps: {max_steps}")
        if config.get("description"):
            print(f"Description: {config['description']}")
    else:
        # Use default parameters
        model_name = "default-model"  # You might want to make this configurable
        print(f"\nUsing default parameters:")
        print(f"Max Steps: {max_steps}")

    # Load questions
    if not os.path.exists(questions_file):
        print(f"Error: Questions file {questions_file} not found.")
        return

    questions = load_questions_from_jsonl(questions_file)
    if limit:
        questions = questions[:limit]

    print(f"\nLoaded {len(questions)} questions from {questions_file}")

    # Show container restart status
    if restart_searxng:
        if containers_to_restart is None:
            containers_to_restart = ["redis", "searxng", "caddy"]
        print(f"🔄 Container restart enabled: {', '.join(containers_to_restart)}")
        print(f"   Containers will be restarted before each question to reset search limits")
    else:
        print(f"🔍 Container restart disabled")

    # Show manual mode status
    if manual_mode:
        print(f"🖱️  Manual validation mode ENABLED")
        print(f"   You can review and override validation decisions after each question")
    else:
        print(f"🤖 Automatic validation mode")

    # Build clients
    try:
        print("Building agent client...")
        agent_client = build_client(use_lmstudio=use_lmstudio, config={"model": model_name})

        print("Building validation client...")
        validation_client = build_client(use_lmstudio=use_lmstudio, config={"model": model_name})

    except Exception as e:
        print(f"Error building clients: {e}")
        return

    # Initialize or resume run
    if resume_mode and run_id:
        # Resume existing run
        checkpoint = run_manager.load_checkpoint(run_id)
        if not checkpoint:
            print(f"Error: No checkpoint found for run {run_id}")
            return

        start_question = checkpoint.get("current_question", 0)
        results = checkpoint.get("results", [])

        print(f"Resuming run {run_id} from question {start_question + 1}")
    else:
        # Start new run
        run_id = run_manager.generate_run_id()
        run_dir = run_manager.create_run_directory(run_id)
        start_question = 0
        results = []

        print(f"Starting new run: {run_id}")
        print(f"Output directory: {run_dir}")

    # Setup console capture
    console_capture = ConsoleCapture()
    console_capture.start_capture()

    # Run tests
    try:
        for i in range(start_question, len(questions)):
            question = questions[i]
            current_q = i + 1

            print(f"\n{'='*60}")
            print(f"QUESTION {current_q}/{len(questions)}")
            print(f"{'='*60}")
            print(f"Level: {question.get('Level', 'unknown')}")
            print(f"Question: {question['Question'][:200]}{'...' if len(question['Question']) > 200 else ''}")

            # Restart search containers if enabled
            if restart_searxng:
                if containers_to_restart is None:
                    containers_to_restart = ["redis", "searxng", "caddy"]

                restart_results = restart_search_containers(containers_to_restart)
                successful_restarts = sum(restart_results.values())
                total_containers = len(containers_to_restart)

                if successful_restarts == total_containers:
                    print(f"✓ All {total_containers} containers restarted successfully")
                elif successful_restarts > 0:
                    print(f"⚠ Warning: Only {successful_restarts}/{total_containers} containers restarted successfully")
                    print(f"  Search results may be affected by rate limits")
                else:
                    print(f"⚠ Warning: All container restarts failed, continuing with test...")
                    print(f"  Search results will likely be affected by rate limits")

            # Run single test
            result = run_single_test(
                question_data=question,
                agent_client=agent_client,
                validation_client=validation_client,
                max_steps=max_steps
            )

            results.append(result)

            # Handle manual mode if enabled
            if manual_mode:
                while True:
                    # Display validation results for manual review
                    display_manual_validation(question, result)

                    # Get user's decision
                    decision = get_manual_decision()

                    if decision == 'overwrite':
                        # Handle overwriting the validator's decision
                        handle_overwrite_validation(result)
                        break
                    elif decision == 'redo':
                        # Re-run the current question
                        result = handle_redo_question(
                            question, agent_client, validation_client, max_steps
                        )
                        # Replace the last result in results list
                        results[-1] = result
                        continue  # Show the new result for review
                    elif decision == 'continue':
                        # Continue to next question
                        break
                    elif decision == 'quit':
                        # Exit testing early
                        print("\n🚪 Exiting testing as requested by user...")
                        raise KeyboardInterrupt()

                # Offer option to toggle searxng restart
                if restart_searxng:
                    restart_searxng = toggle_searxng_restart(restart_searxng)

            # Print result summary
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            if result['success'] and result['is_correct']:
                status += " (CORRECT)"
            elif result['success']:
                status += " (INCORRECT)"

            print(f"Result: {status}")
            print(f"Time: {result['execution_time']:.2f}s")
            print(f"Steps: {result['steps_taken']}")

            # Save checkpoint every 5 questions or at the end
            if (current_q % 5 == 0) or (current_q == len(questions)):
                # Calculate summary
                summary = calculate_summary(results, len(questions))

                # Save checkpoint
                checkpoint_data = {
                    "is_complete": current_q == len(questions),
                    "current_question": current_q,
                    "total_questions": len(questions),
                    "start_time": datetime.now().isoformat(),
                    "results": results
                }

                run_manager.save_checkpoint(run_id, checkpoint_data, agent_config_id)
                run_manager.save_checkpoint_results(run_id, results, summary,
                                                  console_capture.get_output(), agent_config_id)

                print(f"✓ Checkpoint saved at question {current_q}")

        # Final summary and results
        console_capture.stop_capture()
        final_output = console_capture.get_output()

        summary = calculate_summary(results, len(questions))

        # Update configuration performance
        if agent_config_id:
            score = config_manager.calculate_score(summary)
            is_new_best = config_manager.update_config_performance(
                agent_config_id, run_id, score, summary
            )
            if is_new_best:
                print(f"\n🎉 NEW BEST SCORE for configuration {config['name']}: {score:.3f}")

        # Save final results
        run_manager.save_final_results(run_id, summary, results, final_output, agent_config_id)

        # Cleanup if requested
        if cleanup_checkpoints:
            run_manager.cleanup_checkpoints(run_id)

        run_manager.cleanup_temp_files(run_id)

        # Print final summary
        print_summary(summary)

    except KeyboardInterrupt:
        print(f"\n\n⚠ Testing interrupted by user at question {current_q}")
        console_capture.stop_capture()

        # Save checkpoint before exit
        summary = calculate_summary(results, len(questions))
        checkpoint_data = {
            "is_complete": False,
            "current_question": current_q - 1,  # We didn't complete the current question
            "total_questions": len(questions),
            "start_time": datetime.now().isoformat(),
            "results": results
        }

        run_manager.save_checkpoint(run_id, checkpoint_data, agent_config_id)
        run_manager.save_checkpoint_results(run_id, results, summary,
                                          console_capture.get_output(), agent_config_id)

        print(f"✓ Progress saved. Resume with run_id: {run_id}")

    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        console_capture.stop_capture()

        # Save checkpoint before exit
        summary = calculate_summary(results, len(questions))
        checkpoint_data = {
            "is_complete": False,
            "current_question": current_q,
            "total_questions": len(questions),
            "start_time": datetime.now().isoformat(),
            "results": results
        }

        run_manager.save_checkpoint(run_id, checkpoint_data, agent_config_id)
        run_manager.save_checkpoint_results(run_id, results, summary,
                                          console_capture.get_output(), agent_config_id)

        print(f"✓ Progress saved. Resume with run_id: {run_id}")
        raise


def calculate_summary(results: List[Dict[str, Any]], total_questions: int) -> Dict[str, Any]:
    """Calculate summary statistics from test results."""
    successful_runs = sum(1 for r in results if r['success'])
    failed_runs = total_questions - successful_runs
    correct_answers = sum(1 for r in results if r['is_correct'])
    incorrect_answers = successful_runs - correct_answers

    total_execution_time = sum(r['execution_time'] for r in results if r['execution_time'])
    total_steps = sum(r['steps_taken'] for r in results if r['steps_taken'])

    # Calculate rates
    success_rate = (successful_runs / total_questions * 100) if total_questions > 0 else 0
    accuracy_rate = (correct_answers / successful_runs * 100) if successful_runs > 0 else 0
    average_execution_time = total_execution_time / successful_runs if successful_runs > 0 else 0
    average_steps = total_steps / successful_runs if successful_runs > 0 else 0

    # Failure breakdown
    failure_breakdown = {}
    for result in results:
        if not result['success']:
            failure_type = result.get('failure_type', 'unknown')
            failure_breakdown[failure_type] = failure_breakdown.get(failure_type, 0) + 1

    # Level statistics
    level_stats = {}
    for result in results:
        level = result.get('question_level', 'unknown')
        if level not in level_stats:
            level_stats[level] = {
                'total': 0,
                'successful_runs': 0,
                'correct_answers': 0
            }
        level_stats[level]['total'] += 1
        if result['success']:
            level_stats[level]['successful_runs'] += 1
        if result['is_correct']:
            level_stats[level]['correct_answers'] += 1

    return {
        'total_questions': total_questions,
        'successful_runs': successful_runs,
        'failed_runs': failed_runs,
        'correct_answers': correct_answers,
        'incorrect_answers': incorrect_answers,
        'success_rate': success_rate,
        'accuracy_rate': accuracy_rate,
        'total_execution_time': total_execution_time,
        'average_execution_time': average_execution_time,
        'total_steps': total_steps,
        'average_steps': average_steps,
        'failure_breakdown': failure_breakdown,
        'level_stats': level_stats
    }


def main(**kwargs) -> None:
    """Wrapper function for backward compatibility."""
    run_complete_test(**kwargs)


if __name__ == "__main__":
    # You can modify these parameters as needed

    # Example usage modes:

    # 1. Start a new run with agent configuration
    # main(
    #     questions_file="testing/questions/questions.jsonl",
    #     use_lmstudio=True,
    #     max_steps=10,
    #     limit=None,  # Set to a number to test fewer questions
    #     show_validation_details=True,
    #     output_dir="testing/results",
    #     resume_mode=False,
    #     cleanup_checkpoints=False,  # Set to True to clean up checkpoints after completion
    #     agent_config_id="config_001"  # Use a specific agent configuration
    # )

    # 2. Create a new agent configuration
    # main(
    #     create_config=True,
    #     config_name="GPT-4o Test Config",
    #     config_model="gpt-4o-mini",
    #     config_description="Testing configuration with GPT-4o-mini model",
    #     max_steps=15
    # )

    # 3. List available agent configurations
    # main(list_configs=True)

    # 4. List available runs
    # main(list_runs=True, output_dir="testing/results")

    # 5. Resume a specific run with agent configuration
    # main(
    #     questions_file="testing/questions/questions.jsonl",
    #     use_lmstudio=True,
    #     max_steps=10,
    #     resume_mode=True,
    #     run_id="run_20250108_143022_abc12345",  # Specific run ID
    #     output_dir="testing/results",
    #     agent_config_id="config_001"  # Agent configuration will be restored from checkpoint
    # )

    # 6. Resume with interactive selection
    # main(
    #     questions_file="testing/questions/questions.jsonl",
    #     use_lmstudio=True,
    #     max_steps=10,
    #     resume_mode=True,  # Will prompt to select from incomplete runs
    #     output_dir="testing/results"
    # )

    # 7. Run with container restart (prevents rate limiting issues)
    # main(
    #     questions_file="testing/questions/questions.jsonl",
    #     use_lmstudio=True,
    #     max_steps=10,
    #     restart_searxng=True,  # Restart containers before each question
    #     containers_to_restart=["redis", "searxng", "caddy"],  # Containers to restart
    #     agent_config_id="config_001",  # Use specific agent configuration
    #     limit=None  # Test all questions
    # )

    # Default: Start new run with agent configuration tracking
    main(
        questions_file="testing/questions/questions.jsonl",
        use_lmstudio=True,
        max_steps=10,
        limit=None,  # Set to a number to test fewer questions
        show_validation_details=True,
        output_dir="testing/results",
        resume_mode=False,
        run_id=None,
        list_runs=False,
        cleanup_checkpoints=False,  # Set to True if you want to clean up checkpoint files after completion
        agent_config_id=None,  # Set to a config ID to track performance
        list_configs=False,  # List configurations (basic)
        list_configs_sorted=False,  # List configurations sorted by score
        list_runs_sorted=False,  # List runs sorted by score
        list_summary=False,  # Show system performance summary
        restart_searxng=True,  # Set to True to restart containers before each question
        containers_to_restart=None  # Use default containers: ["redis", "searxng", "caddy"]
    )