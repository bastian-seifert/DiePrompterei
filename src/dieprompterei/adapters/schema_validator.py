"""
JSON schema validation for structured task outputs.
"""

import json
import re
from typing import Any

import jsonschema
from jsonschema import ValidationError


def validate_output(output: dict, schema: dict) -> tuple[bool, str]:
    """
    Validate output against JSON schema.

    Args:
        output: Output dict to validate
        schema: JSON schema dict

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        jsonschema.validate(instance=output, schema=schema)
        return (True, "")
    except ValidationError as e:
        return (False, str(e))


def extract_structured_output(text: str) -> dict[str, Any]:
    """
    Extract JSON from LLM response text.

    Handles various formats:
    - Direct JSON
    - Markdown code blocks (```json ... ```)
    - Inline code blocks

    Args:
        text: Raw LLM response text

    Returns:
        Extracted JSON dict

    Raises:
        ValueError: If no valid JSON found
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json block
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try extracting from ``` block (no json marker)
    match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } in text
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from text: {text[:200]}...")


def safe_extract_output(text: str, schema: dict | None = None) -> dict[str, Any]:
    """
    Safely extract and optionally validate structured output.

    Args:
        text: Raw LLM response text
        schema: Optional JSON schema for validation

    Returns:
        Extracted and validated JSON dict, or error dict if extraction/validation fails
    """
    try:
        output = extract_structured_output(text)

        if schema:
            is_valid, error_msg = validate_output(output, schema)
            if not is_valid:
                return {"error": "validation_failed", "message": error_msg}

        return output

    except ValueError as e:
        return {"error": "extraction_failed", "message": str(e)}
