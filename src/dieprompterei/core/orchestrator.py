"""
Der Orchestrator - Coordinates the optimization loop.

Manages the interaction between Poet, Judge, and Guardian while enforcing
the sacred separation between creation and validation.
"""

import json
import time
from pathlib import Path

from ..adapters.metrics import compute_final_score
from ..adapters.schema_validator import safe_extract_output
from .guardian import Guardian
from .judge import Judge
from .llm_client import LLMClient, create_llm_client
from .models import (
    OptimizationReceipt,
    RoundReceipt,
    TaskConfig,
)
from .poet import Poet


class ConvergenceTracker:
    """Tracks convergence based on improvement threshold and plateau detection."""

    def __init__(self, config):
        """
        Initialize convergence tracker.

        Args:
            config: ConvergenceConfig from task configuration
        """
        self.config = config
        self.rounds_without_improvement = 0
        self.best_score = -float("inf")
        self.history = []

    def update(self, current_round: int, current_score: float) -> tuple[bool, str]:
        """
        Update tracker and check convergence.

        Args:
            current_round: Current round number (0-indexed)
            current_score: Current final score

        Returns:
            Tuple of (should_stop, reason)
        """
        self.history.append(current_score)

        # Always run minimum rounds
        if current_round < self.config.min_rounds:
            return (False, "")

        improvement = current_score - self.best_score

        # Track improvement
        if improvement >= self.config.improvement_threshold:
            self.rounds_without_improvement = 0
            self.best_score = current_score
        else:
            self.rounds_without_improvement += 1

        # Check plateau
        if self.rounds_without_improvement >= self.config.plateau_rounds:
            return (True, f"no_improvement_for_{self.config.plateau_rounds}_rounds")

        # Check single-round threshold (after min_rounds)
        if 0 <= improvement < self.config.improvement_threshold:
            return (True, "improvement_below_threshold")

        return (False, "")


class Orchestrator:
    """
    Coordinates the prompt optimization loop.

    Enforces separation: Poet never sees validation data, only filtered feedback.
    """

    def __init__(self, task_config: TaskConfig):
        """
        Initialize orchestrator with task configuration.

        Args:
            task_config: Complete task configuration
        """
        self.task_config = task_config

        # Create LLM clients
        base_client = create_llm_client(task_config.task.llm)

        # Initialize agents with appropriate temperatures
        self.poet = Poet(
            base_client, temperature=task_config.task.optimization.poet_temperature
        )
        self.judge = Judge(base_client, temperature=task_config.task.llm.judge_temperature)
        self.guardian = Guardian(
            base_client, temperature=task_config.task.llm.guardian_temperature
        )

        # Load validation data
        self.validation_data = self._load_validation_data()

        # Tracking
        self.total_api_calls = 0
        self.total_tokens_input = 0
        self.total_tokens_output = 0

    def _load_validation_data(self) -> list[dict]:
        """
        Load validation dataset from JSONL file.

        Returns:
            List of validation examples with 'input' and 'expected' keys
        """
        validation_path = self.task_config.task.validation.path
        data = []

        with validation_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        return data

    def _run_prompt_on_input(
        self, prompt_template: str, input_text: str, client: LLMClient
    ) -> dict:
        """
        Apply prompt to a single input and extract structured output.

        Args:
            prompt_template: Prompt with {input} placeholder
            input_text: Input text to process
            client: LLM client to use

        Returns:
            Extracted structured output dict
        """
        # Fill in the {input} placeholder
        filled_prompt = prompt_template.replace("{input}", input_text)

        # Call LLM (deterministic for evaluation)
        response = client.generate(
            system_prompt="",
            user_prompt=filled_prompt,
            temperature=0.0,
        )

        # Track API usage
        self.total_api_calls += 1
        self.total_tokens_input += response.tokens_input
        self.total_tokens_output += response.tokens_output

        # Extract structured output
        schema = (
            self.task_config.task.output_schema.model_dump()
            if self.task_config.task.output_schema
            else None
        )
        return safe_extract_output(response.text, schema)

    def _evaluate_prompt(self, prompt: str, client: LLMClient) -> dict:
        """
        Evaluate a prompt on the full validation set.

        Args:
            prompt: Prompt template to evaluate
            client: LLM client to use

        Returns:
            Dict with predictions, judge_output, guardian_output, final_score
        """
        # Run prompt on all validation inputs (pre-compute predictions)
        predictions = []
        inputs = []
        expected = []

        for item in self.validation_data:
            input_text = item["input"]
            inputs.append(input_text)
            expected.append(item["expected"])

            prediction = self._run_prompt_on_input(prompt, input_text, client)
            predictions.append(prediction)

        # Judge evaluates predictions
        judge_output = self.judge.evaluate(
            predictions=predictions,
            expected=expected,
            inputs=inputs,
            task_config=self.task_config,
        )

        # Guardian filters feedback
        guardian_output = self.guardian.filter_feedback(judge_output)

        # Verify no leakage (assertion)
        no_leakage = self.guardian.verify_no_leakage(guardian_output, inputs)
        if not no_leakage:
            raise RuntimeError(
                "CRITICAL: Guardian allowed validation data to leak! "
                "This violates the sacred separation."
            )

        # Compute final score
        final_score = compute_final_score(
            primary_metric=judge_output.scores.primary,
            variance=judge_output.scores.variance,
            variance_penalty_weight=self.task_config.task.scoring.variance_penalty_weight,
        )

        return {
            "predictions": predictions,
            "judge_output": judge_output,
            "guardian_output": guardian_output,
            "final_score": final_score,
        }

    def optimize(self) -> OptimizationReceipt:
        """
        Run the full optimization loop.

        Returns:
            Complete optimization receipt with history and best prompt
        """
        start_time = time.time()
        rounds = []
        convergence_tracker = ConvergenceTracker(
            self.task_config.task.optimization.convergence
        )

        # Create shared LLM client for prompt execution
        execution_client = create_llm_client(self.task_config.task.llm)

        # Round 0: Evaluate baseline if provided
        best_prompt = self.task_config.task.baseline_prompt
        best_score = -float("inf")

        if best_prompt:
            round_start = time.time()
            result = self._evaluate_prompt(best_prompt, execution_client)

            rounds.append(
                RoundReceipt(
                    round=0,
                    prompt=best_prompt,
                    scores=result["judge_output"].scores,
                    feedback_summary="Baseline evaluation",
                    duration_seconds=time.time() - round_start,
                    api_calls=self.total_api_calls,
                    tokens_used={
                        "input": self.total_tokens_input,
                        "output": self.total_tokens_output,
                    },
                    improvement=None,
                )
            )

            best_score = result["final_score"]
            convergence_tracker.update(0, best_score)

        # Optimization loop
        current_feedback = None
        current_round = 1 if best_prompt else 0

        for round_num in range(
            current_round, self.task_config.task.optimization.max_rounds
        ):
            round_start = time.time()

            # Poet generates candidates based on filtered feedback
            candidates = self.poet.generate_candidates(
                task_config=self.task_config,
                feedback=current_feedback,
                num_candidates=self.task_config.task.optimization.candidates_per_round,
            )

            # Evaluate all candidates
            best_candidate_prompt = None
            best_candidate_score = -float("inf")
            best_candidate_guardian_output = None
            best_candidate_judge_output = None

            for candidate in candidates:
                result = self._evaluate_prompt(candidate.prompt, execution_client)

                if result["final_score"] > best_candidate_score:
                    best_candidate_score = result["final_score"]
                    best_candidate_prompt = candidate.prompt
                    best_candidate_guardian_output = result["guardian_output"]
                    best_candidate_judge_output = result["judge_output"]

            # Update best overall prompt
            improvement = best_candidate_score - best_score
            if best_candidate_score > best_score:
                best_score = best_candidate_score
                best_prompt = best_candidate_prompt

            # Record round
            rounds.append(
                RoundReceipt(
                    round=round_num,
                    prompt=best_candidate_prompt,
                    scores=best_candidate_judge_output.scores,
                    feedback_summary=f"Evaluated {len(candidates)} candidates",
                    duration_seconds=time.time() - round_start,
                    api_calls=self.total_api_calls,
                    tokens_used={
                        "input": self.total_tokens_input,
                        "output": self.total_tokens_output,
                    },
                    improvement=improvement,
                )
            )

            # Update feedback for next round
            current_feedback = best_candidate_guardian_output

            # Check convergence
            should_stop, reason = convergence_tracker.update(round_num, best_score)
            if should_stop:
                convergence_reason = reason
                break
        else:
            convergence_reason = "max_rounds_reached"

        # Create receipt
        return OptimizationReceipt(
            task_name=self.task_config.task.name,
            timestamp=OptimizationReceipt.create_timestamp(),
            rounds=rounds,
            final_prompt=best_prompt,
            final_score=best_score,
            convergence_reason=convergence_reason,
            total_cost={
                "api_calls": self.total_api_calls,
                "tokens_input": self.total_tokens_input,
                "tokens_output": self.total_tokens_output,
                "duration_seconds": time.time() - start_time,
            },
        )

    def save_receipt(self, receipt: OptimizationReceipt, output_dir: Path):
        """
        Save optimization receipt to file.

        Args:
            receipt: Optimization receipt
            output_dir: Directory to save receipt
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full receipt as JSON
        receipt_path = output_dir / f"{receipt.task_name}_{receipt.timestamp}.json"
        with receipt_path.open("w", encoding="utf-8") as f:
            json.dump(receipt.model_dump(), f, indent=2)

        # Save best prompt as text file
        prompt_path = output_dir / f"{receipt.task_name}_best.txt"
        with prompt_path.open("w", encoding="utf-8") as f:
            f.write(receipt.final_prompt)
