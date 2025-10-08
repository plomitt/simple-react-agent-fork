from __future__ import annotations

import json
import traceback
import os
import csv
import uuid
import shutil
import io
import sys
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from struct_agent.instructor_based.new_agent import run_react_loop
from struct_agent.instructor_based.client_manager import build_client


class ConsoleCapture:
    """Captures console output for checkpointing and logging."""

    def __init__(self):
        self.buffer = io.StringIO()
        self.original_stdout = sys.stdout
        self.is_capturing = False

    def start_capture(self):
        """Start capturing console output."""
        if not self.is_capturing:
            sys.stdout = self.buffer
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

        # Sort by start time (newest first)
        runs.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        return runs

    def save_checkpoint(self, run_id: str, checkpoint_data: Dict[str, Any]) -> None:
        """Save checkpoint data atomically."""
        run_dir = os.path.join(self.base_output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

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

    def save_checkpoint_results(self, run_id: str, results: List[Dict[str, Any]], summary: Dict[str, Any], console_output: str = "") -> None:
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
                
                f.write(f"- Overall assessment: {assessment} ({success_rate:.1f}% success rate)\n")

            os.rename(temp_summary, summary_file)

            # Save console log checkpoint
            with open(temp_log, 'w', encoding='utf-8') as f:
                f.write(f"CONSOLE LOG - Run {run_id} - Question {summary.get('current_question', 0)}/{summary.get('total_questions', 0)}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
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

    def save_final_results(self, run_id: str, summary: Dict[str, Any], results: List[Dict[str, Any]], console_output: str = "") -> None:
        """Save final results with clear, standardized names."""
        run_dir = os.path.join(self.base_output_dir, run_id)
        temp_dir = os.path.join(run_dir, "temp")

        # Final files with clear names
        final_csv_file = os.path.join(run_dir, "FINAL_RESULTS.csv")
        final_summary_file = os.path.join(run_dir, "FINAL_SUMMARY.txt")
        final_output_file = os.path.join(run_dir, "FULL_OUTPUT.txt")

        # Save final CSV with enhanced columns
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
            user_config={"max_steps": max_steps}
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


def run_test_suite(
    questions_file: str,
    use_lmstudio: bool = True,
    max_steps: int = 10,
    limit: Optional[int] = None,
    run_manager: Optional[RunManager] = None,
    run_id: Optional[str] = None,
    resume_mode: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """
    Run the complete test suite with checkpoint and resume functionality.

    Args:
        questions_file: Path to the JSONL questions file
        use_lmstudio: Whether to use LM Studio or OpenRouter
        max_steps: Maximum steps per question
        limit: Optional limit on number of questions to test
        run_manager: RunManager instance for checkpointing
        run_id: Specific run ID to use or resume
        resume_mode: Whether to resume from checkpoint

    Returns:
        Tuple of (summary_stats, detailed_results, run_id)
    """
    if run_manager is None:
        run_manager = RunManager()

    # Initialize console capture
    console_capture = ConsoleCapture()
    console_capture.start_capture()

    print("Starting comprehensive agent test suite...")

    # Load questions
    try:
        questions = load_questions_from_jsonl(questions_file)
        if limit:
            questions = questions[:limit]
        print(f"✓ Loaded {len(questions)} questions from {questions_file}")
    except Exception as e:
        print(f"✗ Failed to load questions: {e}")
        return {"error": "Failed to load questions"}, [], ""

    # Initialize clients
    try:
        agent_client = build_client(use_lmstudio=use_lmstudio)
        validation_client = build_client(use_lmstudio=use_lmstudio)
        print("✓ Clients initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize clients: {e}")
        return {"error": "Client initialization failed"}, [], ""

    # Handle run ID and resume logic
    start_question = 0
    results = []

    if resume_mode and run_id:
        # Resume existing run
        checkpoint = run_manager.load_checkpoint(run_id)
        if checkpoint:
            start_question = checkpoint.get("current_question", 0)
            results = checkpoint.get("results", [])
            print(f"✓ Resuming run {run_id} from question {start_question + 1}/{len(questions)}")
        else:
            print(f"✗ No checkpoint found for run {run_id}, starting fresh")
            resume_mode = False
    elif not run_id:
        # Create new run
        run_id = run_manager.generate_run_id()
        print(f"✓ Starting new run: {run_id}")

    # Create run directory
    run_dir = run_manager.create_run_directory(run_id)

    # Initialize checkpoint data
    start_time = datetime.now().isoformat()
    checkpoint_data = {
        "run_id": run_id,
        "questions_file": questions_file,
        "use_lmstudio": use_lmstudio,
        "max_steps": max_steps,
        "limit": limit,
        "start_time": start_time,
        "total_questions": len(questions),
        "current_question": start_question,
        "results": results,
        "is_complete": False
    }

    # Save initial checkpoint
    run_manager.save_checkpoint(run_id, checkpoint_data)

    def calculate_enhanced_summary(results: List[Dict[str, Any]], total_questions: int, current_question: int) -> Dict[str, Any]:
        """Calculate summary with level-specific stats and failure breakdown."""
        # Basic stats
        successful_runs = sum(1 for r in results if r["success"])
        failed_runs = sum(1 for r in results if not r["success"])
        correct_answers = sum(1 for r in results if r["success"] and r["is_correct"])
        incorrect_answers = sum(1 for r in results if r["success"] and not r["is_correct"])
        total_time = sum(r["execution_time"] or 0 for r in results)
        total_steps = sum(r["steps_taken"] or 0 for r in results if r["success"])

        # Failure breakdown
        failure_breakdown = {}
        for result in results:
            if not result["success"]:
                failure_type = result.get("failure_type", "unknown")
                failure_breakdown[failure_type] = failure_breakdown.get(failure_type, 0) + 1

        # Level-specific statistics
        level_stats = {}
        for result in results:
            level = str(result.get("question_level", "unknown"))
            if level not in level_stats:
                level_stats[level] = {
                    "total": 0,
                    "successful_runs": 0,
                    "correct_answers": 0,
                    "failed_runs": 0,
                    "failure_breakdown": {}
                }
            
            level_stats[level]["total"] += 1
            
            if result["success"]:
                level_stats[level]["successful_runs"] += 1
                if result["is_correct"]:
                    level_stats[level]["correct_answers"] += 1
            else:
                level_stats[level]["failed_runs"] += 1
                failure_type = result.get("failure_type", "unknown")
                level_stats[level]["failure_breakdown"][failure_type] = level_stats[level]["failure_breakdown"].get(failure_type, 0) + 1

        return {
            "total_questions": total_questions,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": (successful_runs / current_question) * 100 if current_question > 0 else 0,
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
            "accuracy_rate": (correct_answers / successful_runs) * 100 if successful_runs > 0 else 0,
            "total_execution_time": total_time,
            "average_execution_time": total_time / current_question if current_question > 0 else 0,
            "total_steps": total_steps,
            "average_steps": total_steps / successful_runs if successful_runs > 0 else 0,
            "questions_answered": successful_runs,
            "questions_not_answered": failed_runs,
            "current_question": current_question,
            "failure_breakdown": failure_breakdown,
            "level_stats": level_stats
        }

    # Track results
    current_summary = calculate_enhanced_summary(results, len(questions), start_question)

    # Run questions (starting from resume point if applicable)
    for i in range(start_question, len(questions)):
        question_data = questions[i]
        question_num = i + 1

        print(f"\n--- Test {question_num}/{len(questions)} ---")
        print(f"Task ID: {question_data.get('task_id', 'unknown')}")
        print(f"Question: {question_data['Question'][:100]}{'...' if len(question_data['Question']) > 100 else ''}")
        print(f"Expected: {question_data['Final answer']}")

        result = run_single_test(
            question_data,
            agent_client,
            validation_client,
            max_steps
        )

        # Update counters
        if result["success"]:
            if result["is_correct"]:
                print(f"✓ Correct ({result['execution_time']:.2f}s, {result['steps_taken']} steps)")
            else:
                print(f"✗ Incorrect ({result['execution_time']:.2f}s, {result['steps_taken']} steps)")
                print(f"  Agent answered: {result['agent_answer'][:100]}{'...' if len(result['agent_answer']) > 100 else ''}")
        else:
            print(f"✗ Failed ({result['execution_time']:.2f}s): {result['error']}")

        results.append(result)

        # Calculate current summary
        current_summary = calculate_enhanced_summary(results, len(questions), question_num)

        # Update and save checkpoint
        checkpoint_data.update({
            "current_question": question_num,
            "results": results
        })
        run_manager.save_checkpoint(run_id, checkpoint_data)

        # Save intermediate results
        console_output = console_capture.get_output()
        run_manager.save_checkpoint_results(run_id, results, current_summary, console_output)

        print(f"✓ Checkpoint saved after question {question_num}/{len(questions)}")

    # Mark run as complete
    checkpoint_data["is_complete"] = True
    checkpoint_data["end_time"] = datetime.now().isoformat()
    run_manager.save_checkpoint(run_id, checkpoint_data)

    # Calculate final summary statistics
    final_summary = calculate_enhanced_summary(results, len(questions), len(questions))
    final_summary.update({
        "run_id": run_id,
        "run_dir": run_dir
    })

    # Save final results with clear file names
    console_output = console_capture.get_output()
    run_manager.save_final_results(run_id, final_summary, results, console_output)

    # Stop console capture
    console_capture.stop_capture()

    # Clean up temporary files
    run_manager.cleanup_temp_files(run_id)

    print(f"\n✓ Run {run_id} completed successfully!")
    return final_summary, results, run_id


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


def main(
    questions_file: str = "testing/questions/questions.jsonl",
    use_lmstudio: bool = True,
    max_steps: int = 10,
    limit: Optional[int] = None,
    show_validation_details: bool = False,
    output_dir: str = "testing/results",
    resume_mode: bool = False,
    run_id: Optional[str] = None,
    list_runs: bool = False,
    cleanup_checkpoints: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Main test execution function with checkpoint and resume support.

    Args:
        questions_file: Path to questions JSONL file
        use_lmstudio: Whether to use LM Studio client
        max_steps: Maximum steps per question
        limit: Optional limit on number of questions
        show_validation_details: Whether to show validation reasoning
        output_dir: Directory to save output files
        resume_mode: Whether to resume from an existing run
        run_id: Specific run ID to resume (required if resume_mode=True)
        list_runs: Whether to list available runs and exit
        cleanup_checkpoints: Whether to clean up checkpoint files after completion

    Returns:
        Tuple of (summary, results)
    """
    # Handle list runs mode
    if list_runs:
        list_available_runs(output_dir)
        return {}, []

    # Initialize run manager
    run_manager = RunManager(output_dir)

    # Handle resume mode validation
    if resume_mode and not run_id:
        # List available runs and let user choose
        runs = run_manager.list_runs()
        incomplete_runs = [r for r in runs if not r["is_complete"]]

        if not incomplete_runs:
            print("No incomplete runs found to resume.")
            return {}, []

        print("\n" + "=" * 80)
        print("AVAILABLE RUNS TO RESUME")
        print("=" * 80)

        for i, run in enumerate(incomplete_runs, 1):
            print(f"{i}. {run['run_id']}")
            print(f"   Status: Incomplete ({run['current_question']}/{run['total_questions']} questions)")
            print(f"   Started: {run['start_time']}")
            print(f"   Path: {run['path']}")
            print()

        try:
            choice = int(input(f"Select run to resume (1-{len(incomplete_runs)}): ")) - 1
            if 0 <= choice < len(incomplete_runs):
                run_id = incomplete_runs[choice]["run_id"]
            else:
                print("Invalid selection.")
                return {}, []
        except (ValueError, KeyboardInterrupt):
            print("Invalid selection or cancelled.")
            return {}, []

    print("Comprehensive Agent Testing Suite")
    print(f"Questions file: {questions_file}")
    print(f"Max steps per question: {max_steps}")
    if limit:
        print(f"Testing limited to {limit} questions")
    if resume_mode:
        print(f"Resume mode: ON (Run ID: {run_id})")
    print("=" * 80)

    # Run tests with checkpoint support
    summary, results, actual_run_id = run_test_suite(
        questions_file=questions_file,
        use_lmstudio=use_lmstudio,
        max_steps=max_steps,
        limit=limit,
        run_manager=run_manager,
        run_id=run_id,
        resume_mode=resume_mode
    )

    # Handle error case
    if "error" in summary:
        print(f"Error: {summary['error']}")
        return summary, results

    # Print summary
    print_summary(summary)

    # Print question table
    print_question_table(results)

    # Print detailed results
    print_detailed_results(results, show_validation_details=show_validation_details)

    # Clean up checkpoint files if requested
    if cleanup_checkpoints and actual_run_id:
        run_manager.cleanup_checkpoints(actual_run_id)

    return summary, results


if __name__ == "__main__":
    # You can modify these parameters as needed

    # Example usage modes:

    # 1. Start a new run (default)
    # main(
    #     questions_file="testing/questions_2.jsonl",
    #     use_lmstudio=True,
    #     max_steps=10,
    #     limit=None,  # Set to a number to test fewer questions
    #     show_validation_details=True,
    #     output_dir="testing/results",
    #     resume_mode=False,
    #     cleanup_checkpoints=False  # Set to True to clean up checkpoints after completion
    # )

    # 2. List available runs
    # main(list_runs=True, output_dir="testing/results")

    # 3. Resume a specific run
    # main(
    #     questions_file="testing/questions_2.jsonl",
    #     use_lmstudio=True,
    #     max_steps=10,
    #     resume_mode=True,
    #     run_id="run_20250108_143022_abc12345",  # Specific run ID
    #     output_dir="testing/results"
    # )

    # 4. Resume with interactive selection
    # main(
    #     questions_file="testing/questions_2.jsonl",
    #     use_lmstudio=True,
    #     max_steps=10,
    #     resume_mode=True,  # Will prompt to select from incomplete runs
    #     output_dir="testing/results"
    # )

    # Default: Start new run
    main(
        questions_file="testing/questions/questions.jsonl",
        use_lmstudio=True,
        max_steps=10,
        limit=1,  # Set to a number to test fewer questions
        show_validation_details=True,
        output_dir="testing/results",
        resume_mode=False,
        run_id=None,
        list_runs=False,
        cleanup_checkpoints=False  # Set to True if you want to clean up checkpoint files after completion
    )