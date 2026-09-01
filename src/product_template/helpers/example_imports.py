"""
Example script importing the ai-models and eegprep packages.
"""
# pylint: disable=unused-import
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL, Logger
from typing import Optional

# Example imports from ai-models and eegprep packages
from galen_functionality import GalenClient, GalenConfig

def verify_imports(logger: Optional[Logger]) -> None:
    """
    Helper function to verify imports from ai-models and eegprep packages.
    Logs the successful imports.
    
    Args:
        logger (Optional[Logger]): Logger instance for logging messages.
        
    Returns:
        None
        
    Raises:
        ImportError: If any of the imports fail.
    """
    
    if logger is None:
        import logging
        logging.basicConfig(level=logging.INFO)
        logger: Logger = logging.getLogger(__name__)
    
    logger.info("Imports from ai-models and eegprep packages were successful.")
    logger.info(f"Galen Client class: {GalenClient}")
    logger.info(f"Galen Config class: {GalenConfig}")
    
    return None

if __name__ == "__main__":
    verify_imports(None)