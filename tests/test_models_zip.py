"""Model file discovery, ZIP handling and TensorFlow.js conversion tests."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

import asanai


def test_newest_match(tmp_path):
    (tmp_path / "model(1).json").write_text("{}", encoding="utf-8")
    (tmp_path / "model(3).json").write_text("{}", encoding="utf-8")
    newest = tmp_path / "model(10).json"
    newest.write_text("{}", encoding="utf-8")
    assert asanai._newest_match(tmp_path, r"model\((\d+)\)\.json") == newest
    assert asanai._newest_match(tmp_path, r"nothing-here\((\d+)\)") is None


def test_newest_match_rejects_names_without_number(tmp_path, monkeypatch):
    (tmp_path / "plain.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(asanai.re, "fullmatch", lambda pattern, name: True)
    with pytest.raises(ValueError):
        asanai._newest_match(tmp_path, r"model\((\d+)\)\.json")


def test_find_model_files_none_directory():
    assert asanai.find_model_files(None) == {}


def test_find_model_files_direct(tmp_path):
    (tmp_path / "model.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.weights.bin").write_bytes(b"data")
    found = asanai.find_model_files(tmp_path)
    assert found["model.json"] == tmp_path / "model.json"
    assert found["model.weights.bin"] == tmp_path / "model.weights.bin"


def test_find_model_files_numbered_and_missing(tmp_path):
    (tmp_path / "model(2).json").write_text("{}", encoding="utf-8")
    found = asanai.find_model_files(tmp_path)
    assert found["model.json"] == tmp_path / "model(2).json"
    assert found["model.weights.bin"] is None


def test_copy_and_patch_tfjs(tmp_path):
    model_json = tmp_path / "input.json"
    weights = tmp_path / "weights.bin"
    model_json.write_text(
        json.dumps({"weightsManifest": [{"paths": ["old.bin"]}]}), encoding="utf-8"
    )
    weights.write_bytes(b"weights")

    json_out, bin_out = asanai.copy_and_patch_tfjs(str(model_json), str(weights), "out")

    patched = json.loads(Path(json_out).read_text(encoding="utf-8"))
    assert patched["weightsManifest"][0]["paths"] == ["./out.bin"]
    assert Path(bin_out).read_bytes() == b"weights"


def test_delete_tmp_files(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("x", encoding="utf-8")
    second.write_text("y", encoding="utf-8")
    asanai.delete_tmp_files(str(first), str(second))
    assert not first.exists() and not second.exists()
    asanai.delete_tmp_files(str(first), str(second))


def test_find_model_zip_direct(tmp_path):
    (tmp_path / "model.zip").write_bytes(b"data")
    assert asanai.find_model_zip() == "model.zip"


def test_find_model_zip_numbered(tmp_path):
    (tmp_path / "model(3).zip").write_bytes(b"data")
    (tmp_path / "model (2).zip").write_bytes(b"data")
    assert asanai.find_model_zip() == "model (2).zip"


def test_find_model_zip_none(tmp_path):
    assert asanai.find_model_zip() is None


def test_find_model_zip_ignores_directories(tmp_path):
    (tmp_path / "model(1).zip").mkdir()
    assert asanai.find_model_zip() is None


def test_zip_file_would_overwrite(tmp_path):
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as zip_ref:
        zip_ref.writestr("existing.txt", "data")
    assert asanai.zip_file_would_overwrite(str(archive)) is False

    (tmp_path / "existing.txt").write_text("old", encoding="utf-8")
    assert asanai.zip_file_would_overwrite(str(archive)) is True


@pytest.mark.parametrize(
    "error",
    [
        zipfile.BadZipFile("bad"),
        FileNotFoundError("missing"),
        PermissionError("denied"),
        OSError("os"),
        zipfile.LargeZipFile("large"),
    ],
)
def test_zip_file_would_overwrite_errors(monkeypatch, error, quiet_console):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(asanai.zipfile, "ZipFile", fail)
    assert asanai.zip_file_would_overwrite("model.zip") is True


def test_extract_zip_file(tmp_path):
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as zip_ref:
        zip_ref.writestr("extracted.txt", "data")
    asanai.extract_zip_file(str(archive))
    assert (tmp_path / "extracted.txt").read_text(encoding="utf-8") == "data"


@pytest.mark.parametrize(
    "error",
    [
        zipfile.BadZipFile("bad"),
        zipfile.LargeZipFile("large"),
        FileNotFoundError("missing"),
        PermissionError("denied"),
        IsADirectoryError("dir"),
        OSError("os"),
    ],
)
def test_extract_zip_file_errors(monkeypatch, error, quiet_console):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(asanai.zipfile, "ZipFile", fail)
    asanai.extract_zip_file("model.zip")
    assert quiet_console.messages


def test_find_and_extract_model_zip_file_if_exists_none(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "find_model_zip", lambda: None)
    asanai.find_and_extract_model_zip_file_if_exists()
    assert quiet_console.messages


def test_find_and_extract_model_zip_file_if_exists_skips_overwrite(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "find_model_zip", lambda: "model.zip")
    monkeypatch.setattr(asanai, "zip_file_would_overwrite", lambda path: True)
    extracted = []
    monkeypatch.setattr(asanai, "extract_zip_file", lambda path: extracted.append(path))
    asanai.find_and_extract_model_zip_file_if_exists()
    assert extracted == []


def test_find_and_extract_model_zip_file_if_exists_extracts(monkeypatch):
    monkeypatch.setattr(asanai, "find_model_zip", lambda: "model.zip")
    monkeypatch.setattr(asanai, "zip_file_would_overwrite", lambda path: False)
    extracted = []
    monkeypatch.setattr(asanai, "extract_zip_file", lambda path: extracted.append(path))
    asanai.find_and_extract_model_zip_file_if_exists()
    assert extracted == ["model.zip"]


def test_locate_tfjs_model_files_none_directory():
    assert asanai.locate_tfjs_model_files(None) == (None, None)


def test_locate_tfjs_model_files_found_initially(tmp_path):
    (tmp_path / "model.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.weights.bin").write_bytes(b"data")
    assert asanai.locate_tfjs_model_files(tmp_path) == (
        str(tmp_path / "model.json"),
        str(tmp_path / "model.weights.bin"),
    )


def test_locate_tfjs_model_files_after_zip(tmp_path, monkeypatch):
    extracted = []

    def extract():
        extracted.append(True)
        (tmp_path / "model.json").write_text("{}", encoding="utf-8")
        (tmp_path / "model.weights.bin").write_bytes(b"data")

    monkeypatch.setattr(asanai, "find_and_extract_model_zip_file_if_exists", extract)
    assert asanai.locate_tfjs_model_files(tmp_path) == (
        str(tmp_path / "model.json"),
        str(tmp_path / "model.weights.bin"),
    )
    assert extracted == [True]


def test_locate_tfjs_model_files_not_found(tmp_path, monkeypatch):
    monkeypatch.setattr(asanai, "find_and_extract_model_zip_file_if_exists", lambda: None)
    assert asanai.locate_tfjs_model_files(tmp_path) == (None, None)


def test_convert_to_keras_existing_h5(tmp_path, quiet_console):
    (tmp_path / "model.h5").write_text("x", encoding="utf-8")
    assert asanai.convert_to_keras_if_needed(tmp_path) is True


def test_convert_to_keras_missing_tfjs(tmp_path, monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "locate_tfjs_model_files", lambda directory: (None, None))
    with pytest.raises(SystemExit) as exc:
        asanai.convert_to_keras_if_needed(tmp_path)
    assert exc.value.code == 1


def test_convert_to_keras_copy_failure(tmp_path, monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "locate_tfjs_model_files", lambda directory: ("m.json", "w.bin"))
    monkeypatch.setattr(asanai, "copy_and_patch_tfjs", lambda j, w: (None, None))
    deleted = []
    monkeypatch.setattr(asanai, "delete_tmp_files", lambda j, w: deleted.append((j, w)))
    assert asanai.convert_to_keras_if_needed(tmp_path) is False
    assert deleted == [(None, None)]


def test_convert_to_keras_local_success(tmp_path, monkeypatch):
    monkeypatch.setattr(asanai, "locate_tfjs_model_files", lambda directory: ("m.json", "w.bin"))
    monkeypatch.setattr(asanai, "copy_and_patch_tfjs", lambda j, w: ("tmp.json", "tmp.bin"))
    monkeypatch.setattr(
        asanai, "_pip_install_tensorflowjs_converter_and_run_it", lambda args: True
    )
    deleted = []
    monkeypatch.setattr(asanai, "delete_tmp_files", lambda j, w: deleted.append((j, w)))
    assert asanai.convert_to_keras_if_needed(tmp_path) is True
    assert deleted == [("tmp.json", "tmp.bin")]


@pytest.mark.parametrize(
    ("docker_available", "docker_success", "expected"),
    [(False, False, False), (True, False, False), (True, True, True)],
)
def test_convert_to_keras_docker_fallback(
    tmp_path, monkeypatch, quiet_console, docker_available, docker_success, expected
):
    monkeypatch.setattr(asanai, "locate_tfjs_model_files", lambda directory: ("m.json", "w.bin"))
    monkeypatch.setattr(asanai, "copy_and_patch_tfjs", lambda j, w: ("tmp.json", "tmp.bin"))
    monkeypatch.setattr(
        asanai, "_pip_install_tensorflowjs_converter_and_run_it", lambda args: False
    )
    monkeypatch.setattr(asanai, "update_wsl_if_windows", lambda: None)
    monkeypatch.setattr(
        asanai, "check_docker_and_try_to_install", lambda j, w: docker_available
    )
    monkeypatch.setattr(asanai, "run_docker_conversion", lambda args: docker_success)
    deleted = []
    monkeypatch.setattr(asanai, "delete_tmp_files", lambda j, w: deleted.append((j, w)))

    assert asanai.convert_to_keras_if_needed(tmp_path) is expected
    assert deleted


def test_run_docker_conversion_daemon_not_running(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "start_docker_if_not_running", lambda: False)

    def fail(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="no daemon")

    monkeypatch.setattr(asanai.subprocess, "run", fail)
    assert asanai.run_docker_conversion(["a"]) is False


def test_run_docker_conversion_daemon_timeout(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "start_docker_if_not_running", lambda: False)

    def timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(asanai.subprocess, "run", timeout)
    assert asanai.run_docker_conversion(["a"]) is False


def test_run_docker_conversion_build_failure(monkeypatch, quiet_console):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(asanai.subprocess, "run", run)
    monkeypatch.setattr(asanai, "write_dockerfile", lambda path: None)
    monkeypatch.setattr(asanai, "build_docker_image", lambda *args: False)
    assert asanai.run_docker_conversion(["a"]) is False
    assert calls and calls[0][:2] == ["docker", "info"]


def test_run_docker_conversion_success(monkeypatch, quiet_console):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="done", stderr="")

    monkeypatch.setattr(asanai.subprocess, "run", run)
    monkeypatch.setattr(asanai, "write_dockerfile", lambda path: None)
    monkeypatch.setattr(asanai, "build_docker_image", lambda *args: True)
    assert asanai.run_docker_conversion(["a", "b"]) is True
    assert calls[-1][:5] == ["docker", "run", "--rm", "-v", f"{Path.cwd()}:/app"]


def test_run_docker_conversion_failure(monkeypatch, quiet_console):
    def run(cmd, **kwargs):
        if cmd[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(cmd, 0)
        raise subprocess.CalledProcessError(1, cmd, stderr="failed")

    monkeypatch.setattr(asanai.subprocess, "run", run)
    monkeypatch.setattr(asanai, "write_dockerfile", lambda path: None)
    monkeypatch.setattr(asanai, "build_docker_image", lambda *args: True)
    assert asanai.run_docker_conversion(["a"]) is False


def test_run_docker_conversion_container_timeout(monkeypatch, quiet_console):
    def run(cmd, **kwargs):
        if cmd[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(cmd, 0)
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(asanai.subprocess, "run", run)
    monkeypatch.setattr(asanai, "write_dockerfile", lambda path: None)
    monkeypatch.setattr(asanai, "build_docker_image", lambda *args: True)
    assert asanai.run_docker_conversion(["a"]) is False


def test_run_docker_conversion_keyboard_interrupt(monkeypatch, quiet_console):
    def run(cmd, **kwargs):
        if cmd[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(cmd, 0)
        raise KeyboardInterrupt()

    monkeypatch.setattr(asanai.subprocess, "run", run)
    monkeypatch.setattr(asanai, "write_dockerfile", lambda path: None)
    monkeypatch.setattr(asanai, "build_docker_image", lambda *args: True)
    with pytest.raises(SystemExit) as exc:
        asanai.run_docker_conversion(["a"])
    assert exc.value.code == 0


def test_write_dockerfile(tmp_path):
    path = tmp_path / "Dockerfile"
    asanai.write_dockerfile(str(path))
    content = path.read_text(encoding="utf-8")
    assert "tensorflowjs" in content
    assert "FROM python:3.11-slim" in content


def test_build_docker_image_success(monkeypatch, fake_progress):
    monkeypatch.setattr(
        asanai.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0)
    )
    assert asanai.build_docker_image("image", "Dockerfile", ".") is True


def test_build_docker_image_failure(monkeypatch, quiet_console, fake_progress):
    def fail(*a, **k):
        raise subprocess.CalledProcessError(1, "docker", stderr="failed")

    monkeypatch.setattr(asanai.subprocess, "run", fail)
    assert asanai.build_docker_image("image", "Dockerfile", ".") is False


def test_build_docker_image_timeout(monkeypatch, quiet_console, fake_progress):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker", 1)

    monkeypatch.setattr(asanai.subprocess, "run", timeout)
    assert asanai.build_docker_image("image", "Dockerfile", ".") is False


def test_build_docker_image_keyboard_interrupt(monkeypatch, quiet_console, fake_progress):
    def interrupt(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(asanai.subprocess, "run", interrupt)
    with pytest.raises(SystemExit) as exc:
        asanai.build_docker_image("image", "Dockerfile", ".")
    assert exc.value.code == 0
