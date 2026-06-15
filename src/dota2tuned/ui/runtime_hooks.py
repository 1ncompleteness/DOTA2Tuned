from __future__ import annotations

import asyncio
import sys
import traceback


def _is_benign_loop_teardown(unraisable) -> bool:
    """True for the harmless asyncio loop-teardown noise Gradio can emit.

    Gradio 6 on Python 3.12 can leave an idle main-thread event loop behind in
    HF Spaces. During garbage collection, Python reports the harmless loop
    close as `Exception ignored ... ValueError: Invalid file descriptor: -1`.
    """
    exc = getattr(unraisable, "exc_value", None)
    if not isinstance(exc, ValueError) or "Invalid file descriptor" not in str(exc):
        return False
    obj = getattr(unraisable, "object", None)
    qualname = getattr(obj, "__qualname__", "") or ""
    if "BaseEventLoop.__del__" in qualname:
        return True
    tb = getattr(unraisable, "exc_traceback", None)
    frames = traceback.extract_tb(tb) if tb else []
    return any(
        frame.filename.endswith(
            (
                "asyncio/base_events.py",
                "asyncio/unix_events.py",
                "asyncio/selector_events.py",
                "selectors.py",
            )
        )
        for frame in frames
    )


def install_quiet_unraisablehook() -> None:
    """Silence only the benign asyncio loop-teardown noise."""
    install_safe_asyncio_loop_del()
    if getattr(sys.unraisablehook, "_dota2tuned_quiet_unraisablehook", False):
        return

    previous_hook = sys.unraisablehook

    def hook(unraisable):
        if _is_benign_loop_teardown(unraisable):
            return
        previous_hook(unraisable)

    hook._dota2tuned_quiet_unraisablehook = True
    sys.unraisablehook = hook


def install_safe_asyncio_loop_del() -> None:
    """Prevent Python 3.12 from logging Gradio's invalid-fd loop destructor."""
    current_del = asyncio.BaseEventLoop.__del__
    if getattr(current_del, "_dota2tuned_safe_loop_del", False):
        return

    original_del = current_del

    def safe_loop_del(self):
        try:
            original_del(self)
        except ValueError as exc:
            if "Invalid file descriptor" in str(exc):
                return
            raise

    safe_loop_del._dota2tuned_safe_loop_del = True
    safe_loop_del._dota2tuned_original_loop_del = original_del
    asyncio.BaseEventLoop.__del__ = safe_loop_del


def close_idle_main_event_loop() -> None:
    """Close Gradio's orphaned main-thread loop before Uvicorn runs.

    This is intentionally conservative: it only touches the current event loop
    if one exists, is not running, and is not already closed.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        return
    if loop.is_running() or loop.is_closed():
        return
    loop.close()
    asyncio.set_event_loop(None)
