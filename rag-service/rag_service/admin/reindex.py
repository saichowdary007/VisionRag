from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config import get_config
from ..cache.redis_cache import get_client as get_redis_client

# Resolve project root inside the container. This file lives at
# /app/rag_service/admin/reindex.py, so parents[2] == /app.
ROOT = Path(__file__).resolve().parents[2]
# Use an in-container builder module so we don't depend on files outside the image
BUILD_MODULE = "rag_service.admin.build_inplace"
# Keep state/logs alongside other app data under /app/data
STATE_DIR = ROOT / "data"
STATE_DIR.mkdir(parents=True, exist_ok=True)
PID_FILE = STATE_DIR / "reindex.pid"
LOG_FILE = STATE_DIR / "reindex.log"


@dataclass
class ReindexStatus:
    running: bool
    pid: int | None
    log_path: str


def start_reindex() -> ReindexStatus:
    # Optional: invalidate cache before starting reindex
    try:
        cfg = get_config()
        if cfg.invalidate_on_reindex and cfg.redis_url:
            r = get_redis_client(cfg.redis_url)
            if r is not None:
                # Delete common prefixes
                for pattern in ("search:v1:*", "chat:v1:*"):
                    try:
                        cursor = 0
                        while True:
                            cursor, keys = r.scan(cursor=cursor, match=pattern, count=500)
                            if keys:
                                r.delete(*keys)
                            if cursor == 0:
                                break
                    except Exception:
                        pass
    except Exception:
        pass

    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            # If still running, do not start a new one
            if _pid_alive(pid):
                return ReindexStatus(True, pid, str(LOG_FILE))
        except Exception:
            pass

    with open(LOG_FILE, "ab", buffering=0) as log:
        proc = subprocess.Popen([sys.executable, "-m", BUILD_MODULE], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    PID_FILE.write_text(str(proc.pid))
    return ReindexStatus(True, proc.pid, str(LOG_FILE))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def get_status() -> ReindexStatus:
    if not PID_FILE.exists():
        return ReindexStatus(False, None, str(LOG_FILE))
    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception:
        return ReindexStatus(False, None, str(LOG_FILE))
    return ReindexStatus(_pid_alive(pid), pid, str(LOG_FILE))
