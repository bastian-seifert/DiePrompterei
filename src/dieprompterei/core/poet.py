"""
Der Dichter - The Poet agent.

Generates and refines prompt candidates based on filtered feedback.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .llm_client import LLMClient
from .logger import logger
from .models import GuardianOutput, PromptCandidate, TaskConfig


class Poet:
    """
    Der Dichter - Creates and refines prompts.

    The Poet never sees validation data, only aggregated feedback
    filtered by the Guardian.
    """

    def __init__(self, client: LLMClient, temperature: float = 0.5):
        """
        Initialize Poet agent.

        Args:
            client: LLM client for generation
            temperature: Sampling temperature for creativity
        """
        self.client = client
        self.temperature = temperature
        self.template = self._load_template()

    def _load_template(self):
        """Load Jinja2 template for poet prompts."""
        template_dir = Path(__file__).parent.parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        return env.get_template("poet.jinja2")

    def generate_candidates(
        self,
        task_config: TaskConfig,
        feedback: GuardianOutput | None = None,
        num_candidates: int = 4,
    ) -> list[PromptCandidate]:
        """
        Generate multiple prompt candidates.

        Args:
            task_config: Task configuration
            feedback: Filtered feedback from Guardian (None for first round)
            num_candidates: Number of candidates to generate

        Returns:
            List of prompt candidates with reasoning
        """
        if feedback:
            logger.info(
                f"Poet: Generating {num_candidates} refined candidates (primary score: {feedback.scores.primary:.3f})"
            )
        else:
            logger.info(f"Poet: Generating {num_candidates} baseline candidates")

        # Render prompt from template
        rendered = self.template.render(
            task_goal=task_config.task.goal,
            output_schema=(
                task_config.task.output_schema.model_dump()
                if task_config.task.output_schema
                else None
            ),
            feedback=(
                {
                    "scores": {
                        "primary": feedback.scores.primary,
                        "variance": feedback.scores.variance,
                    },
                    "error_patterns": feedback.error_patterns,
                    "suggestions": feedback.suggestions,
                }
                if feedback
                else None
            ),
            num_candidates=num_candidates,
        )

        # Call LLM
        response = self.client.generate_json(
            system_prompt="You are a master prompt engineer. Generate high-quality prompt candidates in valid JSON format.",
            user_prompt=rendered,
            temperature=self.temperature,
        )

        # Parse candidates
        candidates_data = response.get("candidates", [])
        candidates = [
            PromptCandidate(
                id=c["id"],
                prompt=c["prompt"],
                reasoning=c.get("reasoning", ""),
            )
            for c in candidates_data
        ]

        logger.info(f"Poet: Generated {len(candidates)} candidates")
        return candidates

    def generate_baseline_candidates(
        self, task_config: TaskConfig, num_candidates: int = 4
    ) -> list[PromptCandidate]:
        """
        Generate initial baseline candidates without feedback.

        Args:
            task_config: Task configuration
            num_candidates: Number of candidates to generate

        Returns:
            List of baseline prompt candidates
        """
        return self.generate_candidates(
            task_config=task_config,
            feedback=None,
            num_candidates=num_candidates,
        )
