"""
Project-wide logging configuration for news_nli.

Usage:
    from news_nli.logging import logger
    logger.info("message")

Writes to both the console and logs/running_logs.log.
"""

import os
import sys
import logging

LOGGING_FORMAT = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

LOG_DIR = "logs"
LOG_FILEPATH = os.path.join(LOG_DIR, "running_logs.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILEPATH),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("news_nli_logger")
