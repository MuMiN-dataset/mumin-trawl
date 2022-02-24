# Ignore logging from other modules
import logging
logging.getLogger('urllib3').setLevel(logging.WARNING)

from .fact_checker import FactChecker  # noqa
