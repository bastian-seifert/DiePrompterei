"""
Structured logging for DiePrompterei.

Provides consistent status updates across agents and LLM calls.
"""

import logging
import sys

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger("dieprompterei")
