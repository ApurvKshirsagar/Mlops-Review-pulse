import logging
import logging.handlers
import os


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with both a stream handler (stdout) and a
    rotating file handler writing to LOG_DIR/app.log.

    Calling get_logger multiple times with the same name is safe —
    Python's logging module returns the same Logger object and this
    function only adds handlers once.
    """
    logger = logging.getLogger(name)

    # Only add handlers the very first time this logger is configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)  # capture everything; handlers filter by level

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # ── stdout handler (INFO+) ────────────────────────────────────────────────
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    # ── rotating file handler (DEBUG+) ────────────────────────────────────────
    # LOG_DIR must be a host-mounted volume in docker-compose.yml so that
    # log files are NOT stored inside the container filesystem.
    log_dir = os.environ.get("LOG_DIR", "/opt/logs")
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        maxBytes=10 * 1024 * 1024,   # rotate after 10 MB
        backupCount=5,                # keep last 5 rotated files
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger