import asyncio
import warnings

import pytest

from dota2tuned.ui.runtime_hooks import (
    install_quiet_starlette_422_warning,
    install_safe_asyncio_loop_del,
)


def test_install_safe_asyncio_loop_del_suppresses_only_invalid_fd(monkeypatch):
    def invalid_fd_del(self):
        raise ValueError("Invalid file descriptor: -1")

    monkeypatch.setattr(asyncio.BaseEventLoop, "__del__", invalid_fd_del)
    install_safe_asyncio_loop_del()

    asyncio.BaseEventLoop.__del__(object())
    first_patch = asyncio.BaseEventLoop.__del__

    install_safe_asyncio_loop_del()
    assert asyncio.BaseEventLoop.__del__ is first_patch


def test_install_safe_asyncio_loop_del_preserves_other_value_errors(monkeypatch):
    def other_value_error_del(self):
        raise ValueError("not the Gradio fd noise")

    monkeypatch.setattr(asyncio.BaseEventLoop, "__del__", other_value_error_del)
    install_safe_asyncio_loop_del()

    with pytest.raises(ValueError, match="not the Gradio fd noise"):
        asyncio.BaseEventLoop.__del__(object())


def test_install_quiet_starlette_422_warning_suppresses_only_target_message():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        install_quiet_starlette_422_warning()
        import starlette.status

        assert starlette.status.HTTP_422_UNPROCESSABLE_ENTITY == 422
        warnings.warn(
            "'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated. "
            "Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead.",
            UserWarning,
            stacklevel=2,
        )
        warnings.warn("other warning", UserWarning, stacklevel=2)

    assert [str(item.message) for item in caught] == ["other warning"]
