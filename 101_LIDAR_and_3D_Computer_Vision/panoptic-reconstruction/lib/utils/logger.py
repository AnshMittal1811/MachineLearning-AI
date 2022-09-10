import logging
import sys
from pathlib import Path

logger = logging.getLogger("trainer")


def setup_logger(save_path: Path, filename: str = "log.txt") -> None:
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt='%d.%m %H:%M:%S')

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_path:
        fh = logging.FileHandler(save_path / filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
