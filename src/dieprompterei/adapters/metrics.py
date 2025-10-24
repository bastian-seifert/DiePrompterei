"""
Task-specific evaluation metrics.

Implements metrics for classification, extraction, QA, and generation tasks.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


def exact_match(predictions: Sequence[dict], expected: Sequence[dict]) -> float:
    """
    Compute exact match accuracy.

    Args:
        predictions: List of predicted outputs
        expected: List of expected outputs

    Returns:
        Accuracy score (0.0-1.0)
    """
    if len(predictions) != len(expected):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(expected)} expected"
        )

    matches = sum(pred == exp for pred, exp in zip(predictions, expected))
    return matches / len(predictions) if predictions else 0.0


def extract_labels(data: Sequence[dict], key: str = "label") -> list[str]:
    """
    Extract label values from structured outputs.

    Args:
        data: List of dicts containing labels
        key: Key to extract (default: "label")

    Returns:
        List of label strings
    """
    labels = []
    for item in data:
        if isinstance(item, dict):
            # Try common label keys
            label = item.get(key) or item.get("sentiment") or item.get("class")
            labels.append(str(label) if label is not None else "")
        else:
            labels.append(str(item))
    return labels


def classification_metrics(
    predictions: Sequence[dict], expected: Sequence[dict]
) -> dict[str, float]:
    """
    Compute classification metrics (F1, precision, recall).

    Args:
        predictions: List of predicted outputs
        expected: List of expected outputs

    Returns:
        Dict with f1_macro, f1_micro, precision, recall
    """
    pred_labels = extract_labels(predictions)
    exp_labels = extract_labels(expected)

    # Get unique labels for averaging
    labels = list(set(exp_labels))

    return {
        "f1_macro": f1_score(exp_labels, pred_labels, labels=labels, average="macro", zero_division=0),
        "f1_micro": f1_score(exp_labels, pred_labels, labels=labels, average="micro", zero_division=0),
        "precision": precision_score(
            exp_labels, pred_labels, labels=labels, average="macro", zero_division=0
        ),
        "recall": recall_score(exp_labels, pred_labels, labels=labels, average="macro", zero_division=0),
    }


def token_f1(predictions: Sequence[dict], expected: Sequence[dict]) -> dict[str, float]:
    """
    Compute token-level F1 for QA/generation tasks.

    Args:
        predictions: List of predicted text outputs
        expected: List of expected text outputs

    Returns:
        Dict with token_f1, token_precision, token_recall
    """

    def tokenize(text: Any) -> set[str]:
        """Simple whitespace tokenization."""
        if isinstance(text, dict):
            # Extract answer field if present
            text = text.get("answer", "")
        return set(str(text).lower().split())

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    for pred, exp in zip(predictions, expected):
        pred_tokens = tokenize(pred)
        exp_tokens = tokenize(exp)

        if not pred_tokens and not exp_tokens:
            # Both empty - perfect match
            total_f1 += 1.0
            total_precision += 1.0
            total_recall += 1.0
            continue

        if not pred_tokens or not exp_tokens:
            # One empty - no match
            continue

        overlap = len(pred_tokens & exp_tokens)
        precision = overlap / len(pred_tokens)
        recall = overlap / len(exp_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    n = len(predictions)
    return {
        "token_f1": total_f1 / n if n > 0 else 0.0,
        "token_precision": total_precision / n if n > 0 else 0.0,
        "token_recall": total_recall / n if n > 0 else 0.0,
    }


def span_f1(predictions: Sequence[dict], expected: Sequence[dict]) -> dict[str, float]:
    """
    Compute span-level F1 for entity extraction tasks.

    Expects entities in format: [{"text": "...", "type": "..."}]

    Args:
        predictions: List of predicted entity lists
        expected: List of expected entity lists

    Returns:
        Dict with span_f1, span_precision, span_recall
    """

    def extract_spans(data: dict) -> set[tuple[str, str]]:
        """Extract (text, type) tuples from entity dict."""
        if isinstance(data, dict) and "entities" in data:
            entities = data["entities"]
            if isinstance(entities, list):
                return {(e.get("text", ""), e.get("type", "")) for e in entities if isinstance(e, dict)}
        return set()

    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    for pred, exp in zip(predictions, expected):
        pred_spans = extract_spans(pred)
        exp_spans = extract_spans(exp)

        if not pred_spans and not exp_spans:
            total_f1 += 1.0
            total_precision += 1.0
            total_recall += 1.0
            continue

        if not pred_spans or not exp_spans:
            continue

        overlap = len(pred_spans & exp_spans)
        precision = overlap / len(pred_spans)
        recall = overlap / len(exp_spans)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    n = len(predictions)
    return {
        "span_f1": total_f1 / n if n > 0 else 0.0,
        "span_precision": total_precision / n if n > 0 else 0.0,
        "span_recall": total_recall / n if n > 0 else 0.0,
    }


def compute_variance(scores: Sequence[float]) -> float:
    """
    Compute normalized variance (standard deviation).

    Args:
        scores: List of scores (0.0-1.0)

    Returns:
        Standard deviation
    """
    if not scores:
        return 0.0
    return float(np.std(scores))


def compute_final_score(
    primary_metric: float, variance: float, variance_penalty_weight: float
) -> float:
    """
    Compute final optimization score.

    Formula: score = primary_metric - (variance_penalty_weight * variance)

    Args:
        primary_metric: Primary metric value (0.0-1.0)
        variance: Variance/std deviation
        variance_penalty_weight: Weight for variance penalty

    Returns:
        Final score (higher is better)
    """
    return primary_metric - (variance_penalty_weight * variance)


# Metric registry by task type
METRICS_BY_TASK = {
    "classification": {
        "exact_match": exact_match,
        "f1_macro": lambda p, e: classification_metrics(p, e)["f1_macro"],
        "f1_micro": lambda p, e: classification_metrics(p, e)["f1_micro"],
        "precision": lambda p, e: classification_metrics(p, e)["precision"],
        "recall": lambda p, e: classification_metrics(p, e)["recall"],
    },
    "extraction": {
        "exact_match": exact_match,
        "span_f1": lambda p, e: span_f1(p, e)["span_f1"],
        "span_precision": lambda p, e: span_f1(p, e)["span_precision"],
        "span_recall": lambda p, e: span_f1(p, e)["span_recall"],
    },
    "qa": {
        "exact_match": exact_match,
        "token_f1": lambda p, e: token_f1(p, e)["token_f1"],
        "token_precision": lambda p, e: token_f1(p, e)["token_precision"],
        "token_recall": lambda p, e: token_f1(p, e)["token_recall"],
    },
    "generation": {
        "exact_match": exact_match,  # For numeric/structured generation tasks
        "token_f1": lambda p, e: token_f1(p, e)["token_f1"],
        "token_precision": lambda p, e: token_f1(p, e)["token_precision"],
        "token_recall": lambda p, e: token_f1(p, e)["token_recall"],
    },
}


def get_metric_function(task_type: str, metric_name: str):
    """
    Get metric function for task type and metric name.

    Args:
        task_type: Task type (classification, extraction, qa, generation)
        metric_name: Metric name

    Returns:
        Metric function

    Raises:
        ValueError: If task type or metric not supported
    """
    if task_type not in METRICS_BY_TASK:
        raise ValueError(
            f"Unsupported task type: {task_type}. "
            f"Supported: {', '.join(METRICS_BY_TASK.keys())}"
        )

    metrics = METRICS_BY_TASK[task_type]
    if metric_name not in metrics:
        raise ValueError(
            f"Metric '{metric_name}' not available for task '{task_type}'. "
            f"Available: {', '.join(metrics.keys())}"
        )

    return metrics[metric_name]
