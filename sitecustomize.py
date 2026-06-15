"""Process startup tweaks for the Hugging Face Space runtime.

Python imports this module automatically during interpreter startup when the
repository root is on sys.path. Keep it stdlib-only: it must run before Gradio
or the project package is imported.
"""

from __future__ import annotations

import asyncio
import sys
import traceback
import warnings

_STARLETTE_422_DEPRECATION_PATTERN = (
    r"'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated\. "
    r"Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead\."
)


def _is_benign_loop_teardown(unraisable) -> bool:
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


def _install_quiet_unraisablehook() -> None:
    _install_quiet_starlette_422_warning()
    _install_safe_asyncio_loop_del()
    if getattr(sys.unraisablehook, "_dota2tuned_quiet_unraisablehook", False):
        return

    previous_hook = sys.unraisablehook

    def hook(unraisable):
        if _is_benign_loop_teardown(unraisable):
            return
        previous_hook(unraisable)

    hook._dota2tuned_quiet_unraisablehook = True
    sys.unraisablehook = hook


def _install_safe_asyncio_loop_del() -> None:
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


def _install_quiet_starlette_422_warning() -> None:
    warnings.filterwarnings(
        "ignore",
        message=_STARLETTE_422_DEPRECATION_PATTERN,
        category=Warning,
    )


_install_quiet_unraisablehook()
