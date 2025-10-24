"""
Der Wächter - The Guardian agent.

Sanitizes Judge feedback to prevent validation data leakage.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .llm_client import LLMClient
from .models import GuardianOutput, JudgeOutput


class Guardian:
    """
    Der Wächter - Filters feedback to prevent data leakage.

    The Guardian ensures the Poet never sees validation data by
    sanitizing the Judge's detailed analysis into aggregated patterns.
    """

    def __init__(self, client: LLMClient, temperature: float = 0.3):
        """
        Initialize Guardian agent.

        Args:
            client: LLM client for sanitization
            temperature: Moderate temperature for filtering
        """
        self.client = client
        self.temperature = temperature
        self.template = self._load_template()

    def _load_template(self):
        """Load Jinja2 template for guardian prompts."""
        template_dir = Path(__file__).parent.parent / "templates"
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        return env.get_template("guardian.jinja2")

    def filter_feedback(self, judge_output: JudgeOutput) -> GuardianOutput:
        """
        Sanitize Judge feedback to remove validation data leakage.

        Args:
            judge_output: Raw Judge evaluation output

        Returns:
            Sanitized feedback safe for Poet consumption
        """
        # Render prompt from template
        rendered = self.template.render(
            judge_output=judge_output.model_dump()
        )

        # Call LLM for sanitization
        try:
            response = self.client.generate_json(
                system_prompt="You are a strict data guardian. Filter feedback to prevent any validation data leakage. Output valid JSON only.",
                user_prompt=rendered,
                temperature=self.temperature,
            )

            return GuardianOutput(
                scores=judge_output.scores,  # Scores are safe to pass through
                error_patterns=response.get("error_patterns", []),
                suggestions=response.get("suggestions", []),
            )

        except (ValueError, KeyError) as e:
            # Fallback: provide minimal safe feedback
            return GuardianOutput(
                scores=judge_output.scores,
                error_patterns=[
                    f"Some predictions failed (primary score: {judge_output.scores.primary:.2f})"
                ],
                suggestions=[
                    "Consider reviewing prompt clarity and format constraints"
                ],
            )

    def verify_no_leakage(
        self, guardian_output: GuardianOutput, validation_inputs: list[str]
    ) -> bool:
        """
        Verify that filtered feedback contains no validation data.

        Args:
            guardian_output: Filtered feedback
            validation_inputs: Original validation inputs to check against

        Returns:
            True if no leakage detected, False otherwise
        """
        # Collect all text from guardian output
        output_text = " ".join(
            guardian_output.error_patterns + guardian_output.suggestions
        ).lower()

        # Check for any validation input fragments (> 3 words)
        for validation_input in validation_inputs:
            # Split into words and check for sequences of 4+ words
            words = validation_input.lower().split()
            for i in range(len(words) - 3):
                fragment = " ".join(words[i : i + 4])
                if fragment in output_text:
                    return False

        return True
