"""
Der Orchestrator - Coordinates the optimization loop.

Manages the interaction between Poet, Judge, and Guardian while enforcing
the sacred separation between creation and validation.
"""

import asyncio
import json
import time
from pathlib import Path

from ..adapters.metrics import compute_final_score
from ..adapters.schema_validator import safe_extract_output
from .guardian import Guardian
from .judge import Judge
from .llm_client import AsyncLLMClient, LLMClient, create_async_llm_client, create_llm_client
from .logger import logger
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

        # Check target score threshold (always, even before min_rounds)
        if self.config.target_score is not None and current_score >= self.config.target_score:
            return (True, f"target_score_reached_{self.config.target_score:.3f}")

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

    async def _run_prompt_on_input(
        self, prompt_template: str, input_text: str, client: AsyncLLMClient
    ) -> dict:
        """
        Apply prompt to a single input and extract structured output asynchronously.

        Args:
            prompt_template: Prompt with {input} placeholder
            input_text: Input text to process
            client: Async LLM client to use

        Returns:
            Extracted structured output dict
        """
        # Fill in the {input} placeholder
        filled_prompt = prompt_template.replace("{input}", input_text)

        # Call LLM asynchronously (deterministic for evaluation)
        response = await client.generate(
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

    async def _evaluate_prompt(self, prompt: str, client: AsyncLLMClient) -> dict:
        """
        Evaluate a prompt on the full validation set with parallel execution.

        Args:
            prompt: Prompt template to evaluate
            client: Async LLM client to use

        Returns:
            Dict with predictions, judge_output, guardian_output, final_score
        """
        # Prepare inputs and expected outputs
        inputs = []
        expected = []

        for item in self.validation_data:
            inputs.append(item["input"])
            expected.append(item["expected"])

        # Execute all validation examples in parallel
        num_examples = len(inputs)
        logger.info(f"Executing {num_examples} validation examples in parallel...")

        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(self._run_prompt_on_input(prompt, input_text, client))
                for input_text in inputs
            ]

        predictions = [task.result() for task in tasks]

        logger.info(
            f"Completed {num_examples} executions "
            f"({self.total_tokens_input} input tokens, {self.total_tokens_output} output tokens)"
        )

        # Judge evaluates predictions (sync call, single LLM request)
        judge_output = self.judge.evaluate(
            predictions=predictions,
            expected=expected,
            inputs=inputs,
            task_config=self.task_config,
        )

        # Guardian filters feedback (sync call, single LLM request)
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

    async def optimize(self) -> OptimizationReceipt:
        """
        Run the full optimization loop with parallel execution.

        Returns:
            Complete optimization receipt with history and best prompt
        """
        start_time = time.time()
        rounds = []
        convergence_tracker = ConvergenceTracker(
            self.task_config.task.optimization.convergence
        )

        # Create shared async LLM client for prompt execution
        execution_client = create_async_llm_client(self.task_config.task.llm)

        # Round 0: Evaluate baseline if provided
        best_prompt = self.task_config.task.baseline_prompt
        best_score = -float("inf")

        if best_prompt:
            round_start = time.time()
            result = await self._evaluate_prompt(best_prompt, execution_client)

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
            should_stop, reason = convergence_tracker.update(0, best_score)

            # Early exit if baseline is perfect or reaches target
            if should_stop:
                logger.info(
                    f"Baseline achieves target score ({best_score:.3f}). "
                    f"Skipping optimization (reason: {reason})."
                )
                return OptimizationReceipt(
                    task_name=self.task_config.task.name,
                    timestamp=OptimizationReceipt.create_timestamp(),
                    rounds=rounds,
                    final_prompt=best_prompt,
                    final_score=best_score,
                    convergence_reason=reason,
                    total_cost={
                        "api_calls": self.total_api_calls,
                        "tokens_input": self.total_tokens_input,
                        "tokens_output": self.total_tokens_output,
                        "duration_seconds": time.time() - start_time,
                    },
                )

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

            # Evaluate all candidates in parallel
            num_candidates = len(candidates)
            logger.info(f"Evaluating {num_candidates} candidates in parallel...")

            async with asyncio.TaskGroup() as tg:
                candidate_tasks = [
                    tg.create_task(self._evaluate_prompt(candidate.prompt, execution_client))
                    for candidate in candidates
                ]

            # Find best candidate
            best_candidate_prompt = None
            best_candidate_score = -float("inf")
            best_candidate_guardian_output = None
            best_candidate_judge_output = None

            for candidate, task in zip(candidates, candidate_tasks):
                result = task.result()

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
