"""Coverage for the tensorflowjs converter helper and remaining edge cases."""

from __future__ import annotations

import builtins
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import asanai


def test_is_command_available(monkeypatch):
    monkeypatch.setattr(asanai.shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    assert asanai._is_command_available("tool") is True
    monkeypatch.setattr(asanai.shutil, "which", lambda cmd: None)
    assert asanai._is_command_available("tool") is False


def test_converter_local_success(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "_is_command_available", lambda cmd: True)
    calls = []
    monkeypatch.setattr(
        asanai.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr=""),
    )
    assert asanai._pip_install_tensorflowjs_converter_and_run_it(["--input", "x"]) is True
    assert calls == [["tensorflowjs_converter", "--input", "x"]]


def test_converter_local_failure_falls_back(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "_is_command_available", lambda cmd: True)

    def fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="failed")

    monkeypatch.setattr(asanai.subprocess, "run", fail)
    assert asanai._pip_install_tensorflowjs_converter_and_run_it(["x"]) is False


def test_converter_local_timeout(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "_is_command_available", lambda cmd: True)

    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(asanai.subprocess, "run", timeout)
    assert asanai._pip_install_tensorflowjs_converter_and_run_it(["x"]) is False


def test_converter_keyboard_interrupt(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "_is_command_available", lambda cmd: True)

    def interrupt(cmd, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(asanai.subprocess, "run", interrupt)
    with pytest.raises(SystemExit) as exc:
        asanai._pip_install_tensorflowjs_converter_and_run_it(["x"])
    assert exc.value.code == 0


def test_converter_install_then_not_found(monkeypatch, quiet_console):
    results = iter([False, True, False])
    monkeypatch.setattr(asanai, "_is_command_available", lambda cmd: next(results))
    installed = []
    monkeypatch.setattr(
        asanai, "_pip_install", lambda package, quiet=False: installed.append((package, quiet))
    )
    assert asanai._pip_install_tensorflowjs_converter_and_run_it(["x"]) is False
    assert installed == [("tensorflowjs", True)]


@pytest.mark.parametrize("system", ["Windows", "Linux"])
def test_converter_install_and_missing(monkeypatch, quiet_console, system):
    monkeypatch.setattr(asanai, "_is_command_available", lambda cmd: False)
    installed = []
    monkeypatch.setattr(
        asanai, "_pip_install", lambda package, quiet=False: installed.append(package)
    )
    monkeypatch.setattr(asanai.platform, "system", lambda: system)
    assert asanai._pip_install_tensorflowjs_converter_and_run_it(["x"]) is False
    assert installed == ["tensorflowjs"]


def test_find_model_zip_parse_error(monkeypatch, quiet_console, tmp_path):
    (tmp_path / "weird.zip").write_bytes(b"data")
    pattern = SimpleNamespace(match=lambda entry: SimpleNamespace(group=lambda index: "not-a-number"))
    monkeypatch.setattr(asanai.re, "compile", lambda pattern_string: pattern)
    assert asanai.find_model_zip() is None
    assert quiet_console.messages


def test_get_shape_file_not_found_error(tmp_path, monkeypatch, quiet_console):
    path = tmp_path / "valid.txt"
    path.write_text("shape: (3, 224)\n", encoding="utf-8")

    def fail(*args, **kwargs):
        raise FileNotFoundError("gone")

    monkeypatch.setattr(asanai.Path, "open", fail)
    assert asanai.get_shape(path) is None


def test_get_shape_value_error(tmp_path, quiet_console):
    path = tmp_path / "valid.txt"
    path.write_text("shape: (²)\n", encoding="utf-8")
    assert asanai.get_shape(path) is None


def test_model_is_simple_classification_shape_rejected(fake_tf):
    model = SimpleNamespace(
        layers=[fake_tf.Dense(fake_tf.softmax)], output_shape=(None, 1)
    )
    assert asanai.model_is_simple_classification(model) is False


def test_has_classification_output_shape_non_sequence():
    assert asanai._has_classification_output_shape(
        SimpleNamespace(output_shape="not-a-shape")
    ) is False


def test_visualize_webcam_varying_grayscale(monkeypatch, quiet_console):
    class Capture:
        def __init__(self):
            self.reads = [(True, np.zeros((8, 8, 3), dtype=np.uint8))]
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            if self.reads:
                return self.reads.pop(0)
            return (False, None)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(asanai.cv2, "VideoCapture", lambda *args: capture)
    monkeypatch.setattr(asanai, "load_frame", lambda *args: np.zeros((1, 4, 4, 3)))
    monkeypatch.setattr(asanai.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(asanai.cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(asanai.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)

    prediction = np.linspace(0, 1, 32).reshape(1, 4, 4, 2)
    asanai.visualize_webcam(SimpleNamespace(predict=lambda *args, **kwargs: prediction))
    assert capture.released


def test_module_import_failure_is_measured(monkeypatch, capsys):
    project_root = Path(__file__).resolve().parents[1]
    real_import = builtins.__import__

    def blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "psutil" or name.startswith("psutil."):
            raise ModuleNotFoundError("blocked for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocking_import)

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(
            str(project_root / "asanai.py"),
            run_name="asanai_import_failure",
        )

    assert exc.value.code == 1
    assert "Failed to load module" in capsys.readouterr().out


def test_module_import_failure_exits_cleanly():
    project_root = Path(__file__).resolve().parents[1]
    code = """
import sys

class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "skimage" or fullname.startswith("skimage."):
            raise ModuleNotFoundError("blocked for test")
        return None

sys.meta_path.insert(0, Blocker())
import asanai
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "Failed to load module" in result.stdout
