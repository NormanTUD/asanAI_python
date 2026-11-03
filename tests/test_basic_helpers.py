"""Tests for small, mostly pure helpers at the top of ``asanai.py``."""

from __future__ import annotations

import subprocess
import urllib.error
from types import SimpleNamespace

import numpy as np
import pytest

import asanai


def test_signal_handler_closes_windows(monkeypatch):
    closed = []
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: closed.append(True))
    with pytest.raises(SystemExit) as exc:
        asanai.signal_handler(2, None)
    assert exc.value.code == 0
    assert closed


def test_dier_prints_and_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        asanai.dier({"error": "boom"})
    assert exc.value.code == 1
    assert "boom" in capsys.readouterr().out


def test_print_predictions_line_normal(monkeypatch):
    written = []
    monkeypatch.setattr(asanai.sys.stdout, "write", lambda value: written.append(value))
    monkeypatch.setattr(asanai.sys.stdout, "flush", lambda: None)
    asanai.print_predictions_line(np.array([[0.1, 0.9]]), ["a", "b"])
    assert written[0] == "\r"
    assert "a: 0.1000000000" in "".join(written)


class _TypeErrorArray(np.ndarray):
    def __getitem__(self, item):
        raise TypeError("bad array")


def test_print_predictions_line_handles_bad_input():
    asanai.print_predictions_line(np.array([]), ["a"])
    asanai.print_predictions_line(np.array([1]).view(_TypeErrorArray), ["a"])
    asanai.print_predictions_line(np.array([[]]), ["a"])


@pytest.mark.parametrize(
    "replacement",
    [
        lambda *a, **k: (_ for _ in ()).throw(ValueError("empty")),
        lambda *a, **k: None,
        lambda *a, **k: (_ for _ in ()).throw(AttributeError("bad")),
        lambda *a, **k: float("inf"),
    ],
)
def test_print_predictions_line_argmax_errors(monkeypatch, replacement):
    monkeypatch.setattr(asanai.np, "argmax", replacement)
    asanai.print_predictions_line(np.array([[0.5, 0.5]]), ["a", "b"])


def test_rule(quiet_console):
    asanai.rule("hello")
    assert quiet_console.messages


def test_in_virtual_env_true_and_false(monkeypatch):
    monkeypatch.setattr(asanai.sys, "prefix", "/venv")
    monkeypatch.setattr(asanai.sys, "base_prefix", "/usr")
    assert asanai._in_virtual_env() is True

    monkeypatch.setattr(asanai.sys, "prefix", "/usr")
    monkeypatch.setattr(asanai.sys, "base_prefix", "/usr")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    if hasattr(asanai.sys, "real_prefix"):
        monkeypatch.delattr(asanai.sys, "real_prefix")
    assert asanai._in_virtual_env() is False

    monkeypatch.setenv("CONDA_PREFIX", "/conda")
    assert asanai._in_virtual_env() is True


def test_pip_available(monkeypatch):
    monkeypatch.setattr(asanai.shutil, "which", lambda name: "/usr/bin/pip")
    assert asanai._pip_available() is True

    monkeypatch.setattr(asanai.shutil, "which", lambda name: None)
    monkeypatch.setattr(asanai.util, "find_spec", lambda name: object())
    assert asanai._pip_available() is True

    monkeypatch.setattr(asanai.util, "find_spec", lambda name: None)
    assert asanai._pip_available() is False


def test_proxy_hint(quiet_console, monkeypatch):
    asanai._proxy_hint()
    monkeypatch.setenv("HTTP_PROXY", "http://proxy")
    asanai._proxy_hint()
    assert quiet_console.messages


def test_gpu_hint_variants(quiet_console, monkeypatch):
    monkeypatch.setattr(asanai.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    asanai._gpu_hint()

    monkeypatch.setattr(asanai.shutil, "which", lambda name: None)
    monkeypatch.setattr(asanai.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(asanai.platform, "machine", lambda: "arm64")
    asanai._gpu_hint()

    monkeypatch.setattr(asanai.platform, "system", lambda: "Linux")
    monkeypatch.setattr(asanai.platform, "machine", lambda: "x86_64")
    asanai._gpu_hint()
    assert quiet_console.messages


@pytest.mark.parametrize(
    ("system", "machine"),
    [
        ("Darwin", "arm64"),
        ("Darwin", "x86_64"),
        ("Linux", "aarch64"),
        ("Linux", "x86_64"),
        ("Windows", "arm64"),
        ("Windows", "amd64"),
    ],
)
def test_platform_wheel_warning(quiet_console, monkeypatch, system, machine):
    monkeypatch.setattr(asanai.platform, "system", lambda: system)
    monkeypatch.setattr(asanai.platform, "machine", lambda: machine)
    asanai._platform_wheel_warning()
    assert isinstance(quiet_console.messages, list)


class _Response:
    def __init__(self, status=200, data=b"data"):
        self.status = status
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._data


def test_download_file_success(tmp_path, monkeypatch):
    destination = tmp_path / "file.bin"
    monkeypatch.setattr(asanai.urllib.request, "urlopen", lambda url: _Response())
    assert asanai.download_file("https://example.invalid/file", str(destination)) is True
    assert destination.read_bytes() == b"data"


def test_download_file_bad_status(tmp_path, monkeypatch):
    destination = tmp_path / "file.bin"
    monkeypatch.setattr(asanai.urllib.request, "urlopen", lambda url: _Response(status=404))
    assert asanai.download_file("https://example.invalid/file", str(destination)) is False


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("url", 500, "server", None, None),
        urllib.error.URLError("offline"),
        ValueError("bad url"),
    ],
)
def test_download_file_errors(tmp_path, monkeypatch, error):
    destination = tmp_path / "file.bin"

    def raise_error(url):
        raise error

    monkeypatch.setattr(asanai.urllib.request, "urlopen", raise_error)
    assert asanai.download_file("https://example.invalid/file", str(destination)) is False


def test_run_ms_visual_cpp_installer(monkeypatch):
    monkeypatch.setattr(
        asanai.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0)
    )
    assert asanai.run_ms_visual_cpp_installer("installer.exe") is True

    monkeypatch.setattr(
        asanai.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 5)
    )
    assert asanai.run_ms_visual_cpp_installer("installer.exe") is False

    for error in (
        FileNotFoundError("missing"),
        PermissionError("denied"),
        subprocess.SubprocessError("boom"),
    ):
        def raise_error(*a, **k):
            raise error

        monkeypatch.setattr(asanai.subprocess, "run", raise_error)
        assert asanai.run_ms_visual_cpp_installer("installer.exe") is False


def test_normalize_input():
    assert asanai.normalize_input("  YeS ") == "yes"
    assert asanai.normalize_input("ＮＯ") == "no"


def test_ask_yes_no_ci(monkeypatch):
    monkeypatch.setenv("CI", "1")
    assert asanai.ask_yes_no("question") is True


@pytest.mark.parametrize(
    ("answers", "expected"),
    [
        (["yes"], True),
        (["j"], True),
        (["no"], False),
        (["nein"], False),
        (["maybe", "y"], True),
    ],
)
def test_ask_yes_no_loop(monkeypatch, quiet_console, answers, expected):
    monkeypatch.setattr(
        asanai, "Prompt", SimpleNamespace(ask=lambda *a, **k: answers.pop(0))
    )
    assert asanai.ask_yes_no("question") is expected


def test_download_and_install_ms_visual_cpp_cancelled(monkeypatch, tmp_path):
    monkeypatch.setattr(asanai, "ask_yes_no", lambda prompt: False)
    asanai.download_and_install_ms_visual_cpp()


def test_download_and_install_ms_visual_cpp_download_failure(monkeypatch):
    monkeypatch.setattr(asanai, "ask_yes_no", lambda prompt: True)
    monkeypatch.setattr(asanai, "download_file", lambda url, path: False)
    with pytest.raises(SystemExit) as exc:
        asanai.download_and_install_ms_visual_cpp()
    assert exc.value.code == 1


def test_download_and_install_ms_visual_cpp_install_failure(monkeypatch):
    monkeypatch.setattr(asanai, "ask_yes_no", lambda prompt: True)
    monkeypatch.setattr(asanai, "download_file", lambda url, path: True)
    monkeypatch.setattr(asanai, "run_ms_visual_cpp_installer", lambda path: False)
    with pytest.raises(SystemExit) as exc:
        asanai.download_and_install_ms_visual_cpp()
    assert exc.value.code == 1


def test_download_and_install_ms_visual_cpp_success(monkeypatch):
    monkeypatch.setattr(asanai, "ask_yes_no", lambda prompt: True)
    monkeypatch.setattr(asanai, "download_file", lambda url, path: True)
    monkeypatch.setattr(asanai, "run_ms_visual_cpp_installer", lambda path: True)
    asanai.download_and_install_ms_visual_cpp()


def test_pip_install_unavailable(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "_pip_available", lambda: False)
    assert asanai._pip_install("pkg") is False


def test_pip_install_success_and_failure(monkeypatch, fake_progress):
    monkeypatch.setattr(asanai, "_pip_available", lambda: True)
    monkeypatch.setattr(
        asanai.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0)
    )
    assert asanai._pip_install("pkg", quiet=True) is True

    monkeypatch.setattr(
        asanai.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "err")
    )
    assert asanai._pip_install("pkg") is False


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("python"),
        subprocess.CalledProcessError(1, "pip", stderr="failed"),
        subprocess.SubprocessError("boom"),
        KeyboardInterrupt(),
    ],
)
def test_pip_install_errors(monkeypatch, fake_progress, error):
    monkeypatch.setattr(asanai, "_pip_available", lambda: True)

    def raise_error(*args, **kwargs):
        raise error

    monkeypatch.setattr(asanai.subprocess, "run", raise_error)
    assert asanai._pip_install("pkg") is False
