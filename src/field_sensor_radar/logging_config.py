"""Application logging setup."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a concise console logger once for CLI and dashboard use."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
