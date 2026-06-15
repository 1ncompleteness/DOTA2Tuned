import asyncio

import pytest

from dota2tuned.ui.runtime_hooks import install_safe_asyncio_loop_del


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
