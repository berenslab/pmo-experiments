"""Project-wide logging utility functions."""

import logging
from functools import partialmethod

from tqdm import tqdm


def setup_logging(logger_name: str | None, level: str = "INFO") -> logging.Logger:
    """
    Set up logging for the project.

    Args:
        logger_name (str | None): Name of the logger. If None, uses the root logger.
        level (str, optional): Logging level. Defaults to "INFO".

    Returns:
        logging.Logger: Initialized logger instance.

    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if numeric_level >= logging.WARNING:
        # Progress bars are noise once the user has asked for quiet output.
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # pyright: ignore[reportAttributeAccessIssue]

    logger = logging.getLogger(__name__ if logger_name is None else logger_name)
    return logger
