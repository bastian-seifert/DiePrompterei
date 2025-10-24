"""
Der Richter - The Judge agent.

Evaluates prompts against validation data with full access.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..adapters.metrics import compute_variance, get_metric_function
from .llm_client import LLMClient
from .logger import logger
from .models import EvaluationScores, IndividualResult, JudgeOutput, TaskConfig


class Judge:
    """
    Der Richter - Evaluates prompts against validation data.

    The Judge has exclusive access to validation data and performs
    detailed error analysis.
    """

    def __init__(self, client: LLMClient, temperature: float = 0.2):
        """
        Initialize Judge agent.

        Args:
            client: LLM client for analysis
            temperature: Low temperature for consistent evaluation
        """
        self.client = client
        self.temperature = temperature
        self.template = self._load_template()

    def _load_template(self):
        """Load Jinja2 template for judge prompts."""
        template_dir = Path(__file__).parent.parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        return env.get_template("judge.jinja2")

    def evaluate(
        self,
        predictions: list[dict],
        expected: list[dict],
        inputs: list[str],
        task_config: TaskConfig,
    ) -> JudgeOutput:
        """
        Evaluate predictions against expected outputs.

        Args:
            predictions: Predicted outputs from candidate prompt
            expected: Expected outputs from validation data
            inputs: Input texts from validation data
            task_config: Task configuration

        Returns:
            Complete evaluation with scores, error analysis, and suggestions
        """
        logger.info(
            f"Judge: Evaluating {len(predictions)} predictions using {task_config.task.metrics.primary} metric"
        )

        # Compute primary metric
        metric_func = get_metric_function(
            task_config.task.type, task_config.task.metrics.primary
        )
        primary_score = metric_func(predictions, expected)

        # Compute individual correctness
        individual_results = []
        individual_scores = []

        for inp, pred, exp in zip(inputs, predictions, expected):
            correct = pred == exp
            individual_scores.append(1.0 if correct else 0.0)

            individual_results.append(
                IndividualResult(
                    input=inp,
                    predicted=pred,
                    expected=exp,
                    correct=correct,
                    error_type=None if correct else "mismatch",
                )
            )

        # Compute variance
        variance = compute_variance(individual_scores)

        # Compute secondary metrics if requested
        secondary_scores = {}
        for metric_name in task_config.task.metrics.secondary:
            try:
                secondary_func = get_metric_function(task_config.task.type, metric_name)
                secondary_scores[metric_name] = secondary_func(predictions, expected)
            except ValueError:
                # Skip unsupported metrics
                pass

        # Get LLM-based error analysis and suggestions
        logger.info("Judge: Requesting LLM-based error analysis")
        analysis = self._get_llm_analysis(
            individual_results=individual_results,
            task_config=task_config,
            primary_score=primary_score,
        )

        logger.info(
            f"Judge: Evaluation complete. Primary score: {primary_score:.3f}, Variance: {variance:.3f}"
        )

        return JudgeOutput(
            scores=EvaluationScores(
                primary=primary_score,
                variance=variance,
                secondary=secondary_scores,
            ),
            individual_results=individual_results,
            error_analysis=analysis.get("error_analysis", []),
            suggestions=analysis.get("suggestions", []),
        )

    def _get_llm_analysis(
        self,
        individual_results: list[IndividualResult],
        task_config: TaskConfig,
        primary_score: float,
    ) -> dict:
        """
        Use LLM to analyze error patterns and generate suggestions.

        Args:
            individual_results: Individual evaluation results
            task_config: Task configuration
            primary_score: Primary metric score

        Returns:
            Dict with error_analysis and suggestions lists
        """
        # Prepare results for template
        results_data = [
            {
                "input": r.input,
                "predicted": r.predicted,
                "expected": r.expected,
                "correct": r.correct,
            }
            for r in individual_results
        ]

        # Render template
        rendered = self.template.render(
            task_goal=task_config.task.goal,
            output_schema=(
                task_config.task.output_schema.model_dump()
                if task_config.task.output_schema
                else None
            ),
            candidate_prompt="[Evaluated via predictions]",
            results=results_data,
            num_examples=len(results_data),
            primary_metric=task_config.task.metrics.primary,
        )

        # Call LLM for analysis
        try:
            response = self.client.generate_json(
                system_prompt="You are an impartial evaluator. Analyze prompt performance and provide specific, actionable feedback in valid JSON format.",
                user_prompt=rendered,
                temperature=self.temperature,
            )
            return response
        except (ValueError, KeyError) as e:
            # Fallback if LLM fails to provide valid JSON
            return {
                "error_analysis": [
                    f"Automatic analysis unavailable. Primary score: {primary_score:.2f}"
                ],
                "suggestions": ["Review individual predictions for patterns"],
            }
