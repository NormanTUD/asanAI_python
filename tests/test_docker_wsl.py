"""Docker and WSL helpers with explicit Windows/macOS/Linux coverage."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

import asanai


def test_is_docker_installed(monkeypatch):
    monkeypatch.setattr(asanai.shutil, "which", lambda name: "/usr/bin/docker")
    assert asanai.is_docker_installed() is True
    monkeypatch.setattr(asanai.shutil, "which", lambda name: None)
    assert asanai.is_docker_installed() is False


@pytest.mark.parametrize("manager", ["apt", "dnf", "pacman", "unknown"])
def test_try_install_docker_linux(monkeypatch, quiet_console, manager):
    commands = []
    monkeypatch.setattr(
        asanai.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name == manager else None,
    )
    monkeypatch.setattr(
        asanai.subprocess,
        "run",
        lambda cmd, **kwargs: commands.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )
    asanai.try_install_docker_linux()
    if manager == "unknown":
        assert commands == []
    else:
        assert commands


def test_try_install_docker_windows_without_winget(monkeypatch, capsys):
    monkeypatch.setattr(asanai.shutil, "which", lambda name: None)
    asanai.try_install_docker_windows()
    assert "Winget not found" in capsys.readouterr().out


def test_try_install_docker_windows_success(monkeypatch):
    commands = []
    monkeypatch.setattr(asanai.shutil, "which", lambda name: "/winget")
    monkeypatch.setattr(
        asanai.subprocess,
        "run",
        lambda cmd, **kwargs: commands.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )
    started = []
    monkeypatch.setattr(asanai, "start_docker_if_not_running", lambda: started.append(True))

    with pytest.raises(SystemExit) as exc:
        asanai.try_install_docker_windows()
    assert exc.value.code == 0
    assert commands and started


def test_try_install_docker_windows_failure(monkeypatch, capsys):
    monkeypatch.setattr(asanai.shutil, "which", lambda name: "/winget")

    def fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="nope")

    monkeypatch.setattr(asanai.subprocess, "run", fail)
    asanai.try_install_docker_windows()
    assert "Docker installation failed" in capsys.readouterr().out


@pytest.mark.parametrize("returncode", [0, 1])
def test_try_install_docker_mac_brew(monkeypatch, quiet_console, returncode):
    monkeypatch.setattr(asanai.shutil, "which", lambda name: "/opt/homebrew/bin/brew")

    def run(cmd, **kwargs):
        assert cmd[:3] == ["brew", "install", "--cask"]
        assert kwargs["env"]["NONINTERACTIVE"] == "1"
        return subprocess.CompletedProcess(cmd, returncode, stdout="out", stderr="err")

    monkeypatch.setattr(asanai.subprocess, "run", run)
    asanai.try_install_docker_mac()
    assert quiet_console.messages


def test_try_install_docker_mac_without_brew_declined(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai.shutil, "which", lambda name: None)
    monkeypatch.setattr(asanai, "ask_yes_no", lambda prompt: False)
    asanai.try_install_docker_mac()
    assert quiet_console.messages


def test_try_install_docker_mac_curl_failure(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai.shutil, "which", lambda name: None)
    monkeypatch.setattr(asanai, "ask_yes_no", lambda prompt: True)
    monkeypatch.setattr(
        asanai.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="fail"),
    )
    asanai.try_install_docker_mac()
    assert quiet_console.messages


@pytest.mark.parametrize(
    ("existing_path", "expected_prefix"),
    [
        ("/opt/homebrew/bin/brew", "/opt/homebrew/bin/brew"),
        ("/usr/local/bin/brew", "/usr/local/bin/brew"),
        (None, ""),
    ],
)
def test_try_install_docker_mac_homebrew_install(
    monkeypatch, quiet_console, existing_path, expected_prefix
):
    monkeypatch.setattr(asanai.shutil, "which", lambda name: None)
    monkeypatch.setattr(asanai, "ask_yes_no", lambda prompt: True)

    def run(cmd, **kwargs):
        if cmd[0] == "curl":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="installed", stderr="")

    monkeypatch.setattr(asanai.subprocess, "run", run)
    monkeypatch.setattr(asanai.os, "chmod", lambda path, mode: None)
    monkeypatch.setattr(
        asanai.os.path,
        "exists",
        lambda path: path == existing_path if existing_path else False,
    )
    recursive = []
    monkeypatch.setattr(asanai, "try_install_docker_mac", lambda: recursive.append(True))

    # The captured original performs the recursive lookup through the module
    # globals, so the patched name is used for the nested call.
    _ORIGINAL_TRY_INSTALL_DOCKER_MAC()

    assert recursive == [True]
    if expected_prefix:
        assert asanai.os.environ["PATH"].startswith(expected_prefix)


# The original function must be captured once before any test patches it.
_ORIGINAL_TRY_INSTALL_DOCKER_MAC = asanai.try_install_docker_mac


def test_try_install_docker_mac_install_failure(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai.shutil, "which", lambda name: None)
    monkeypatch.setattr(asanai, "ask_yes_no", lambda prompt: True)

    def run(cmd, **kwargs):
        if cmd[0] == "curl":
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="out", stderr="err")

    monkeypatch.setattr(asanai.subprocess, "run", run)
    monkeypatch.setattr(asanai.os, "chmod", lambda path, mode: None)
    asanai.try_install_docker_mac()
    assert quiet_console.messages


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(1, "brew", stderr="err"),
        OSError("os error"),
    ],
)
def test_try_install_docker_mac_unexpected_errors(monkeypatch, quiet_console, error):
    monkeypatch.setattr(asanai.shutil, "which", lambda name: "/brew")

    def fail(cmd, **kwargs):
        raise error

    monkeypatch.setattr(asanai.subprocess, "run", fail)
    asanai.try_install_docker_mac()
    assert quiet_console.messages


def test_update_wsl_non_windows(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai.platform, "system", lambda: "Linux")
    asanai.update_wsl_if_windows()
    assert quiet_console.messages == []


def test_update_wsl_not_installed(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai.platform, "system", lambda: "Windows")
    monkeypatch.setattr(asanai, "check_wsl_installed", lambda: False)
    installed = []
    monkeypatch.setattr(asanai, "install_wsl", lambda: installed.append(True))
    asanai.update_wsl_if_windows()
    assert installed == [True]


def test_update_wsl_up_to_date(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai.platform, "system", lambda: "Windows")
    monkeypatch.setattr(asanai, "check_wsl_installed", lambda: True)
    monkeypatch.setattr(asanai, "check_wsl_update_available", lambda: False)
    updated = []
    monkeypatch.setattr(asanai, "run_wsl_update", lambda: updated.append(True))
    asanai.update_wsl_if_windows()
    assert updated == []


@pytest.mark.parametrize(("answer", "expected"), [(True, [True]), (False, [])])
def test_update_wsl_with_available_update(monkeypatch, quiet_console, answer, expected):
    monkeypatch.setattr(asanai.platform, "system", lambda: "Windows")
    monkeypatch.setattr(asanai, "check_wsl_installed", lambda: True)
    monkeypatch.setattr(asanai, "check_wsl_update_available", lambda: True)
    monkeypatch.setattr(asanai, "ask_yes_no", lambda prompt: answer)
    updated = []
    monkeypatch.setattr(asanai, "run_wsl_update", lambda: updated.append(True))
    asanai.update_wsl_if_windows()
    assert updated == expected


def test_check_wsl_installed(monkeypatch, quiet_console):
    monkeypatch.setattr(
        asanai.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0)
    )
    assert asanai.check_wsl_installed() is True

    def missing(*a, **k):
        raise FileNotFoundError("wsl")

    monkeypatch.setattr(asanai.subprocess, "run", missing)
    assert asanai.check_wsl_installed() is False

    def failed(*a, **k):
        raise subprocess.CalledProcessError(1, "wsl", stderr="err")

    monkeypatch.setattr(asanai.subprocess, "run", failed)
    assert asanai.check_wsl_installed() is False


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("wsl"),
        PermissionError("denied"),
        subprocess.CalledProcessError(1, "wsl", stderr="err"),
        OSError("os"),
    ],
)
def test_install_wsl_errors(monkeypatch, quiet_console, error):
    def fail(*a, **k):
        raise error

    monkeypatch.setattr(asanai.subprocess, "run", fail)
    asanai.install_wsl()
    assert quiet_console.messages


def test_install_wsl_success(monkeypatch, quiet_console):
    monkeypatch.setattr(
        asanai.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0)
    )
    asanai.install_wsl()
    assert quiet_console.messages


def test_check_wsl_update_available(monkeypatch, quiet_console):
    monkeypatch.setattr(
        asanai.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 0, stdout="The installed version is the same as the latest version"
        ),
    )
    assert asanai.check_wsl_update_available() is False

    monkeypatch.setattr(
        asanai.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="update available"),
    )
    assert asanai.check_wsl_update_available() is True


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("wsl"),
        PermissionError("denied"),
        subprocess.CalledProcessError(1, "wsl", stderr="err"),
        OSError("os"),
    ],
)
def test_check_wsl_update_available_errors(monkeypatch, quiet_console, error):
    def fail(*a, **k):
        raise error

    monkeypatch.setattr(asanai.subprocess, "run", fail)
    assert asanai.check_wsl_update_available() is False


def test_run_wsl_update(monkeypatch, quiet_console):
    monkeypatch.setattr(
        asanai.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0)
    )
    asanai.run_wsl_update()

    monkeypatch.setattr(
        asanai.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="err"),
    )
    asanai.run_wsl_update()


@pytest.mark.parametrize(
    "error",
    [
        subprocess.CalledProcessError(1, "wsl", stderr="err"),
        FileNotFoundError("wsl"),
        subprocess.TimeoutExpired("wsl", 10),
        OSError("os"),
    ],
)
def test_run_wsl_update_errors(monkeypatch, quiet_console, error):
    def fail(*a, **k):
        raise error

    monkeypatch.setattr(asanai.subprocess, "run", fail)
    asanai.run_wsl_update()
    assert quiet_console.messages


def test_try_install_docker_already_installed(monkeypatch, capsys):
    monkeypatch.setattr(asanai, "is_docker_installed", lambda: True)
    assert asanai.try_install_docker() is True
    assert "already installed" in capsys.readouterr().out


def test_try_install_docker_declined(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "is_docker_installed", lambda: False)
    monkeypatch.setattr(asanai, "ask_yes_no", lambda prompt: False)
    assert asanai.try_install_docker() is False


@pytest.mark.parametrize(
    ("system", "installer_name"),
    [
        ("Linux", "try_install_docker_linux"),
        ("Windows", "try_install_docker_windows"),
        ("Darwin", "try_install_docker_mac"),
    ],
)
def test_try_install_docker_per_platform(monkeypatch, quiet_console, system, installer_name):
    answers = iter([False, True])
    monkeypatch.setattr(asanai, "is_docker_installed", lambda: next(answers))
    monkeypatch.setattr(asanai, "ask_yes_no", lambda prompt: True)
    monkeypatch.setattr(asanai.platform, "system", lambda: system)
    calls = []
    monkeypatch.setattr(asanai, installer_name, lambda: calls.append(True))
    assert asanai.try_install_docker() is True
    assert calls


def test_try_install_docker_unsupported_os(monkeypatch, capsys):
    monkeypatch.setattr(asanai, "is_docker_installed", lambda: False)
    monkeypatch.setattr(asanai, "ask_yes_no", lambda prompt: True)
    monkeypatch.setattr(asanai.platform, "system", lambda: "Plan9")
    assert asanai.try_install_docker() is False
    assert "Unsupported OS" in capsys.readouterr().out


def test_try_install_docker_still_missing(monkeypatch, capsys):
    monkeypatch.setattr(asanai, "is_docker_installed", lambda: False)
    monkeypatch.setattr(asanai, "ask_yes_no", lambda prompt: True)
    monkeypatch.setattr(asanai.platform, "system", lambda: "Linux")
    monkeypatch.setattr(asanai, "try_install_docker_linux", lambda: None)
    assert asanai.try_install_docker() is False
    assert "still not found" in capsys.readouterr().out


def test_check_docker_and_try_to_install(monkeypatch):
    monkeypatch.setattr(asanai, "_is_command_available", lambda cmd: True)
    assert asanai.check_docker_and_try_to_install("json", "bin") is True

    monkeypatch.setattr(asanai, "_is_command_available", lambda cmd: False)
    deleted = []
    monkeypatch.setattr(asanai, "delete_tmp_files", lambda j, b: deleted.append((j, b)))
    monkeypatch.setattr(asanai, "try_install_docker", lambda: False)
    assert asanai.check_docker_and_try_to_install("json", "bin") is False
    assert deleted == [("json", "bin")]

    monkeypatch.setattr(asanai, "try_install_docker", lambda: True)
    assert asanai.check_docker_and_try_to_install("json", "bin") is False
    assert deleted == [("json", "bin"), ("json", "bin")]


def test_check_docker_and_try_to_install_after_install(monkeypatch):
    answers = iter([False, True])
    monkeypatch.setattr(asanai, "_is_command_available", lambda cmd: next(answers))
    monkeypatch.setattr(asanai, "try_install_docker", lambda: True)
    assert asanai.check_docker_and_try_to_install("json", "bin") is True


def test_is_windows(monkeypatch):
    monkeypatch.setattr(asanai.platform, "system", lambda: "Windows")
    assert asanai.is_windows() is True
    monkeypatch.setattr(asanai.platform, "system", lambda: "Linux")
    assert asanai.is_windows() is False


def test_get_program_files(monkeypatch):
    monkeypatch.setenv("ProgramW6432", "C:/Program Files")
    assert asanai.get_program_files() == "C:/Program Files"

    monkeypatch.delenv("ProgramW6432")
    monkeypatch.setenv("ProgramFiles", "C:/Program Files (x86)")
    assert asanai.get_program_files() == "C:/Program Files (x86)"

    monkeypatch.delenv("ProgramFiles")
    with pytest.raises(EnvironmentError):
        asanai.get_program_files()


def test_is_docker_running(monkeypatch):
    monkeypatch.setattr(
        asanai.psutil,
        "process_iter",
        lambda attrs: [SimpleNamespace(info={"name": "chrome"})],
    )
    assert asanai.is_docker_running() is False

    monkeypatch.setattr(
        asanai.psutil,
        "process_iter",
        lambda attrs: [SimpleNamespace(info={"name": "Docker Desktop.exe"})],
    )
    assert asanai.is_docker_running() is True


def test_start_docker_missing_program_files(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "get_program_files", lambda: "")
    assert asanai.start_docker() == 2


def test_start_docker_missing_executable(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "get_program_files", lambda: "/Program Files")
    monkeypatch.setattr(asanai.os.path, "isfile", lambda path: False)
    assert asanai.start_docker() == 3


def test_start_docker_success(monkeypatch):
    monkeypatch.setattr(asanai, "get_program_files", lambda: "/Program Files")
    monkeypatch.setattr(asanai.os.path, "isfile", lambda path: True)

    class Process:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(asanai.subprocess, "Popen", lambda *a, **k: Process())
    assert asanai.start_docker() == 0


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing"),
        PermissionError("denied"),
        OSError("os"),
        ValueError("bad"),
    ],
)
def test_start_docker_errors(monkeypatch, quiet_console, error):
    monkeypatch.setattr(asanai, "get_program_files", lambda: "/Program Files")
    monkeypatch.setattr(asanai.os.path, "isfile", lambda path: True)

    def fail(*a, **k):
        raise error

    monkeypatch.setattr(asanai.subprocess, "Popen", fail)
    assert asanai.start_docker() == 4


def test_start_docker_get_program_files_raises(monkeypatch, quiet_console):
    monkeypatch.setattr(
        asanai, "get_program_files", lambda: (_ for _ in ()).throw(EnvironmentError("no"))
    )
    assert asanai.start_docker() == 2


def test_start_docker_if_not_running(monkeypatch):
    monkeypatch.setattr(asanai, "is_windows", lambda: False)
    assert asanai.start_docker_if_not_running() is True

    monkeypatch.setattr(asanai, "is_windows", lambda: True)
    monkeypatch.setattr(asanai, "is_docker_running", lambda: True)
    assert asanai.start_docker_if_not_running() is False

    monkeypatch.setattr(asanai, "is_docker_running", lambda: False)
    monkeypatch.setattr(asanai, "start_docker", lambda: 0)
    assert asanai.start_docker_if_not_running() is True

    monkeypatch.setattr(asanai, "start_docker", lambda: 3)
    assert asanai.start_docker_if_not_running() is False
