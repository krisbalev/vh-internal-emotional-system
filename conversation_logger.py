import atexit
import os
import random
from threading import RLock

_logs: list[str] = []
_logfile: str | None = None
_lock = RLock()


def init(filename: str) -> None:
    """
    Initialize the logger with the given filename pattern.
    Generates a unique filename by appending a random number so logs don't overwrite.
    Registers an atexit handler to save automatically.
    """
    global _logfile
    base, ext = os.path.splitext(filename)
    unique = f"{base}_{random.randint(10000, 99999)}{ext}"
    _logfile = unique
    print(f"[conversation_logger] Logging to '{_logfile}'")
    atexit.register(_save)


def log(role: str, mood: str, message: str) -> None:
    """
    Append one conversation turn.
    Example: log("BOSS", "Calm", "Good morning, Alex.")
    """
    entry = f"{role} (mood: {mood}) ▶ {message}"
    with _lock:
        _logs.append(entry)


def save(filename: str | None = None) -> None:
    """
    Force-write all collected logs to disk.
    If no filename is passed, uses the one generated in init().
    """
    target = filename or _logfile
    if not target:
        raise RuntimeError("No logfile specified. Call init(...) first.")
    with _lock, open(target, "w", encoding="utf-8") as f:
        for line in _logs:
            f.write(line + "\n")
    print(f"[conversation_logger] Saved {len(_logs)} entries to '{target}'")


def _save() -> None:
    # internal; used by atexit to flush logs
    try:
        save()
    except Exception:
        pass
