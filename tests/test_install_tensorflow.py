"""TensorFlow installation logic, including Windows/macOS/Linux differences."""

from __future__ import annotations

import types

import pytest

import asanai


def test_install_tensorflow_already_available(monkeypatch, quiet_console):
    fake_tf = types.ModuleType("tensorflow")
    monkeypatch.setattr(asanai.util, "find_spec", lambda name: object())
    monkeypatch.setattr(asanai, "import_module", lambda name: fake_tf)
    hinted = []
    monkeypatch.setattr(asanai, "_gpu_hint", lambda: hinted.append(True))

    assert asanai.install_tensorflow() is fake_tf
    assert hinted


def test_install_tensorflow_requires_virtualenv(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai.util, "find_spec", lambda name: None)
    monkeypatch.setattr(asanai, "_in_virtual_env", lambda: False)
    with pytest.raises(SystemExit) as exc:
        asanai.install_tensorflow()
    assert exc.value.code == 1


def test_install_tensorflow_windows_success(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai.util, "find_spec", lambda name: None)
    monkeypatch.setattr(asanai, "_in_virtual_env", lambda: True)
    monkeypatch.setattr(asanai, "_platform_wheel_warning", lambda: None)
    monkeypatch.setattr(asanai.platform, "system", lambda: "Windows")
    monkeypatch.setattr(asanai.platform, "machine", lambda: "AMD64")
    installed = []
    monkeypatch.setattr(asanai, "download_and_install_ms_visual_cpp", lambda: installed.append(True))
    calls = []
    monkeypatch.setattr(asanai, "_pip_install", lambda pkg, quiet=False: calls.append(pkg) or True)
    monkeypatch.setattr(asanai, "_gpu_hint", lambda: None)

    with pytest.raises(SystemExit) as exc:
        asanai.install_tensorflow()
    assert exc.value.code == 0
    assert installed
    assert calls == ["tensorflow"]


def test_install_tensorflow_darwin_arm64_restarts(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai.util, "find_spec", lambda name: None)
    monkeypatch.setattr(asanai, "_in_virtual_env", lambda: True)
    monkeypatch.setattr(asanai, "_platform_wheel_warning", lambda: None)
    monkeypatch.setattr(asanai.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(asanai.platform, "machine", lambda: "arm64")
    calls = []
    monkeypatch.setattr(asanai, "_pip_install", lambda pkg, quiet=False: calls.append(pkg) or True)
    monkeypatch.setattr(asanai, "_gpu_hint", lambda: None)
    exec_calls = []
    monkeypatch.setattr(asanai.os, "execv", lambda path, argv: exec_calls.append((path, argv)))
    monkeypatch.setattr(asanai.sys, "executable", "/usr/bin/python")

    result = asanai.install_tensorflow(["script.py", "--flag"])
    assert result is None
    assert calls == ["tensorflow-macos"]
    assert exec_calls == [("/usr/bin/python", ["/usr/bin/python", "script.py", "--flag"])]


def test_install_tensorflow_nightly_fallback(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai.util, "find_spec", lambda name: None)
    monkeypatch.setattr(asanai, "_in_virtual_env", lambda: True)
    monkeypatch.setattr(asanai, "_platform_wheel_warning", lambda: None)
    monkeypatch.setattr(asanai.platform, "system", lambda: "Linux")
    monkeypatch.setattr(asanai.platform, "machine", lambda: "x86_64")
    calls = []

    def pip_install(pkg, quiet=False):
        calls.append(pkg)
        return pkg == "tf_nightly"

    monkeypatch.setattr(asanai, "_pip_install", pip_install)
    monkeypatch.setattr(asanai, "_gpu_hint", lambda: None)

    with pytest.raises(SystemExit) as exc:
        asanai.install_tensorflow()
    assert exc.value.code == 0
    assert calls == ["tensorflow", "tf_nightly"]


@pytest.mark.parametrize(
    ("system", "expected_hint"),
    [("Windows", "Scripts"), ("Linux", "bin")],
)
def test_install_tensorflow_manual_fallback(monkeypatch, quiet_console, system, expected_hint):
    monkeypatch.setattr(asanai.util, "find_spec", lambda name: None)
    monkeypatch.setattr(asanai, "_in_virtual_env", lambda: True)
    monkeypatch.setattr(asanai, "_platform_wheel_warning", lambda: None)
    monkeypatch.setattr(asanai.platform, "system", lambda: system)
    monkeypatch.setattr(asanai.platform, "machine", lambda: "x86_64")
    if system == "Windows":
        monkeypatch.setattr(asanai, "download_and_install_ms_visual_cpp", lambda: None)
    monkeypatch.setattr(asanai, "_pip_install", lambda pkg, quiet=False: False)
    monkeypatch.setenv("VIRTUAL_ENV", "/venv")

    with pytest.raises(SystemExit) as exc:
        asanai.install_tensorflow()
    assert exc.value.code == 1
    rendered = " ".join(str(m) for m in quiet_console.messages)
    assert expected_hint in rendered


def test_install_tensorflow_fallback_uses_conda_prefix(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai.util, "find_spec", lambda name: None)
    monkeypatch.setattr(asanai, "_in_virtual_env", lambda: True)
    monkeypatch.setattr(asanai, "_platform_wheel_warning", lambda: None)
    monkeypatch.setattr(asanai.platform, "system", lambda: "Linux")
    monkeypatch.setattr(asanai.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(asanai, "_pip_install", lambda pkg, quiet=False: False)
    monkeypatch.setenv("CONDA_PREFIX", "/conda")
    monkeypatch.setattr(asanai.sys, "prefix", "/fallback")

    with pytest.raises(SystemExit) as exc:
        asanai.install_tensorflow()
    assert exc.value.code == 1
    assert any("/conda" in str(message) for message in quiet_console.messages)
