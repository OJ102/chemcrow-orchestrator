"""The OPERATIONAL LOG: diagnostic messages in size-limited, rotating files (decision Q6).

WHAT THIS FILE IS
-----------------
Two different kinds of record exist (D2 G4):
  - the AUDIT LOG (storage.py's audit_events table): the official record of what
    happened to each run. Required: if it can't be saved, the change fails.
  - the OPERATIONAL LOG (this file): messages for whoever is running the programs,
    e.g. "run X claimed", "generator: temporary error, retrying in 1.1 s".
    Nice to have; never stores run data.

setup_logging() sends every message from our code (Python's `logging` module)
both to the screen and to runs/logs/<program>.log. The file is ROTATED: when it
reaches 5 MB it's renamed to <program>.log.1 and a fresh file starts; at most 3
old files are kept, so the logs can never fill the disk (D2: "bounded rotating
files"; important on the cluster, whose main disk is 97% full).
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
BACKUP_COUNT = 3  # how many old files to keep
# Each line looks like: 2026-11-10 14:03:12 INFO orchestrator.worker: run 3f2a... claimed
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(runs_dir: Path, program: str) -> Path:
    """Send our log messages to the screen and to runs_dir/logs/<program>.log. Returns the file's path.

    program: "api", "worker" or "cli", which gives one log file per program.
    Messages of level INFO and above are kept (DEBUG is ignored).
    """
    log_dir = Path(runs_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{program}.log"
    formatter = logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT,
                                       encoding="utf-8")
    screen_handler = logging.StreamHandler()  # the terminal
    for handler in (file_handler, screen_handler):
        handler.setFormatter(formatter)

    # "orchestrator" is the parent of every logger in our code (orchestrator.worker,
    # orchestrator.retry, ...), so setting it up once covers them all.
    root = logging.getLogger("orchestrator")
    root.setLevel(logging.INFO)
    root.handlers = [file_handler, screen_handler]  # replace, so calling twice doesn't double lines
    root.propagate = False  # don't ALSO pass messages up to Python's default handler
    return path
