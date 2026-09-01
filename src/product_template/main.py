from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL, Logger
from typing import Optional
import logging

# Try to import the project's logging helper. If it's not available (for example
# in minimal test environments), provide a lightweight fallback so tests can
# import this module without failing.
try:
    from galen_functionality.helpers.util_logging import setup_logging  # galen_functionality GAR import
except Exception:  # pragma: no cover - fallback used in test/dev environments
    def setup_logging(level: int = INFO, log_path: Optional[str] = None) -> Logger:
        """Simple fallback logger used when the project's helper is unavailable.

        Returns a console logger configured at `level`.
        """
        logger = logging.getLogger("product_template")
        logger.setLevel(level)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ch.setFormatter(fmt)
            logger.addHandler(ch)
        return logger

from product_template.helpers.example_imports import verify_imports


def main(do_verify_imports: bool = False, logger: Optional[Logger] = None) -> None:
    """
    Main function for the product project template.
    
    Args:
        do_verify_imports (bool): If True, verifies that all example imports work correctly.
        logger (Optional[Logger]): Logger instance for logging messages. If None, a default logger will be used.
        
    Returns:
        None
    """
    if do_verify_imports:
        verify_imports(logger)
        
    if logger is not None:
        logger.info("Hello from product-template!")
        
    return None
    
if __name__ == "__main__":
    # Sets up logging to console and file (if log_path is provided)
    logger: Logger = setup_logging(level=INFO, log_path=None)
    main(do_verify_imports=True, logger=logger)