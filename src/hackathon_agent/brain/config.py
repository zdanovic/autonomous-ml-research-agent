"""
DSPy Language Model Configuration.
"""
import logging
import os

import dspy

logger = logging.getLogger(__name__)

def init_dspy():
    """
    Initialize the DSPy LM backend.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found. DSPy modules will fail if called.")
        return

    # Use GPT-4o-mini as a cost-effective default for high-volume agent reasoning
    lm = dspy.LM("openai/gpt-4o-mini", api_key=api_key, max_tokens=1000)
    dspy.configure(lm=lm)
    logger.info("DSPy LM initialized: openai/gpt-4o-mini")
