import logging

def test_main_logs_message(caplog):
    """Verify main logs the hello message when a logger is provided."""
    from product_template.main import main

    caplog.set_level(logging.INFO)
    logger = logging.getLogger("test_logger")

    # Ensure no prior handlers interfere
    logger.handlers = []

    main(do_verify_imports=False, logger=logger)

    assert any("Hello from product-template!" in rec.getMessage() for rec in caplog.records)
