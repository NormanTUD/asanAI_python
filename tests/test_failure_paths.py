"""Failure-path and robustness tests: missing, broken and hostile inputs."""

from __future__ import annotations

import builtins
import json
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import psutil
import pytest

import asanai


class _Response:
    def __init__(self, status=200, data=b"data", include_status=True):
        self._data = data
        if include_status:
            self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self._data


def test_download_file_without_status_attribute(tmp_path, monkeypatch):
    destination = tmp_path / "file.bin"
    monkeypatch.setattr(
        asanai.urllib.request,
        "urlopen",
        lambda url: _Response(include_status=False),
    )
    assert asanai.download_file("https://example.invalid", str(destination)) is True
    assert destination.read_bytes() == b"data"


def test_download_file_disk_error(tmp_path, monkeypatch):
    destination = tmp_path / "file.bin"
    monkeypatch.setattr(asanai.urllib.request, "urlopen", lambda url: _Response())
    monkeypatch.setattr(
        builtins, "open", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full"))
    )
    assert asanai.download_file("https://example.invalid", str(destination)) is False


def test_run_ms_visual_cpp_installer_os_error(monkeypatch):
    monkeypatch.setattr(
        asanai.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("os")),
    )
    assert asanai.run_ms_visual_cpp_installer("installer.exe") is False


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
def test_ask_yes_no_aborts_on_missing_input(monkeypatch, quiet_console, error):
    monkeypatch.setattr(
        asanai,
        "Prompt",
        SimpleNamespace(ask=lambda *args, **kwargs: (_ for _ in ()).throw(error)),
    )
    assert asanai.ask_yes_no("question") is False


def test_newest_match_unreadable_directory(tmp_path):
    not_a_directory = tmp_path / "file.txt"
    not_a_directory.write_text("x", encoding="utf-8")
    assert asanai._newest_match(not_a_directory, r"model\((\d+)\)\.json") is None


def test_find_model_files_directory_is_a_file(tmp_path):
    not_a_directory = tmp_path / "file.txt"
    not_a_directory.write_text("x", encoding="utf-8")
    found = asanai.find_model_files(not_a_directory)
    assert found["model.json"] is None
    assert found["model.weights.bin"] is None


def test_find_model_files_missing_directory(tmp_path):
    assert asanai.find_model_files(tmp_path / "missing") == {
        "model.json": None,
        "model.weights.bin": None,
    }


def test_delete_tmp_files_tolerates_empty_and_unlink_errors(tmp_path, monkeypatch, quiet_console):
    asanai.delete_tmp_files(None, "")
    path = tmp_path / "file.txt"
    path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        asanai.os, "unlink", lambda target: (_ for _ in ()).throw(PermissionError("denied"))
    )
    asanai.delete_tmp_files(str(path), None)
    assert quiet_console.messages


def test_find_model_zip_listdir_error(monkeypatch, quiet_console):
    monkeypatch.setattr(
        asanai.os, "listdir", lambda target: (_ for _ in ()).throw(PermissionError("denied"))
    )
    assert asanai.find_model_zip() is None


def test_copy_and_patch_tfjs_invalid_json(tmp_path, quiet_console):
    model = tmp_path / "model.json"
    weights = tmp_path / "weights.bin"
    model.write_text("not json", encoding="utf-8")
    weights.write_bytes(b"data")
    assert asanai.copy_and_patch_tfjs(str(model), str(weights)) == ("", "")


def test_copy_and_patch_tfjs_non_object_json(tmp_path, quiet_console):
    model = tmp_path / "model.json"
    weights = tmp_path / "weights.bin"
    model.write_text("[]", encoding="utf-8")
    weights.write_bytes(b"data")
    assert asanai.copy_and_patch_tfjs(str(model), str(weights)) == ("", "")


def test_copy_and_patch_tfjs_missing_weights(tmp_path, quiet_console):
    model = tmp_path / "model.json"
    model.write_text("{}", encoding="utf-8")
    assert asanai.copy_and_patch_tfjs(str(model), str(tmp_path / "missing.bin")) == ("", "")


def test_copy_and_patch_tfjs_write_error(tmp_path, monkeypatch, quiet_console):
    model = tmp_path / "model.json"
    weights = tmp_path / "weights.bin"
    model.write_text(json.dumps({"weightsManifest": []}), encoding="utf-8")
    weights.write_bytes(b"data")
    monkeypatch.setattr(
        asanai.shutil,
        "copyfile",
        lambda source, target: (_ for _ in ()).throw(PermissionError("denied")),
    )
    assert asanai.copy_and_patch_tfjs(str(model), str(weights)) == ("", "")


def test_copy_and_patch_tfjs_manifest_not_a_list(tmp_path, quiet_console):
    model = tmp_path / "model.json"
    weights = tmp_path / "weights.bin"
    model.write_text(json.dumps({"weightsManifest": {"paths": []}}), encoding="utf-8")
    weights.write_bytes(b"data")
    assert asanai.copy_and_patch_tfjs(str(model), str(weights)) == ("", "")


def test_copy_and_patch_tfjs_manifest_entry_not_an_object(tmp_path, quiet_console):
    model = tmp_path / "model.json"
    weights = tmp_path / "weights.bin"
    model.write_text(json.dumps({"weightsManifest": [[]]}), encoding="utf-8")
    weights.write_bytes(b"data")
    assert asanai.copy_and_patch_tfjs(str(model), str(weights)) == ("", "")


def test_extract_zip_file_does_not_escape_working_directory(tmp_path):
    import zipfile

    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as zip_ref:
        zip_ref.writestr("../escaped.txt", "data")
    asanai.extract_zip_file(str(archive))
    assert not (tmp_path.parent / "escaped.txt").exists()
    assert (tmp_path / "escaped.txt").read_text(encoding="utf-8") == "data"


def test_zip_file_would_overwrite_corrupt_archive(tmp_path, quiet_console):
    import zipfile

    archive = tmp_path / "model.zip"
    archive.write_bytes(b"this is not a zip file")
    assert asanai.zip_file_would_overwrite(str(archive)) is True


def test_is_float_list_rejects_any_invalid_value():
    assert asanai._is_float_list(["1", "oops"]) is False
    assert asanai._is_float_list([None]) is False
    assert asanai._is_float_list([]) is True


def test_load_rejects_zero_division(tmp_path, quiet_console):
    from PIL import Image

    path = tmp_path / "image.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(path)
    assert asanai.load(path, divide_by=0) is None


def test_load_frame_rejects_zero_division(quiet_console):
    assert asanai.load_frame(np.zeros((4, 4, 3), dtype=np.uint8), divide_by=0) is None


@pytest.mark.parametrize(
    "model",
    [
        object(),
        SimpleNamespace(input_shape=()),
        SimpleNamespace(input_shape=(None, 0)),
        SimpleNamespace(input_shape=(None, -1)),
        SimpleNamespace(input_shape=(None, "x")),
    ],
)
def test_load_or_input_model_data_invalid_model(quiet_console, model):
    with pytest.raises(SystemExit) as exc:
        asanai.load_or_input_model_data(model, "missing.txt")
    assert exc.value.code == 1


def test_load_or_input_model_data_eof(monkeypatch, quiet_console):
    model = SimpleNamespace(input_shape=(None, 2))
    monkeypatch.setattr(
        builtins, "input", lambda: (_ for _ in ()).throw(EOFError())
    )
    with pytest.raises(SystemExit) as exc:
        asanai.load_or_input_model_data(model, "missing.txt")
    assert exc.value.code == 1


def test_annotate_frame_invalid_predictions(quiet_console):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    assert asanai.annotate_frame(frame, np.array([]), ["a"]) is None
    assert asanai.annotate_frame(frame, np.array([[]]), ["a"]) is None


def test_annotate_frame_overflow(monkeypatch, quiet_console):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    monkeypatch.setattr(asanai.np, "argmax", lambda values: float("inf"))
    assert asanai.annotate_frame(frame, np.array([[0.5]]), ["a"]) is None


def test_annotate_frame_argmax_type_error(monkeypatch, quiet_console):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    monkeypatch.setattr(
        asanai.np, "argmax", lambda values: (_ for _ in ()).throw(TypeError("bad"))
    )
    assert asanai.annotate_frame(frame, np.array([[0.5]]), ["a"]) is None


def test_load_and_prepare_image_model_without_shape(monkeypatch):
    monkeypatch.setattr(asanai, "load", lambda path: np.zeros((1, 4, 4, 3)))
    assert asanai._load_and_prepare_image("image.png", object()) is None
    assert asanai._load_and_prepare_image(
        "image.png", SimpleNamespace(input_shape=5)
    ) is None


def test_visualize_survives_prediction_error(monkeypatch, quiet_console):
    monkeypatch.setattr(
        asanai, "_load_and_prepare_image", lambda path, model: np.zeros((4, 4, 3))
    )
    model = SimpleNamespace(
        predict=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    asanai.visualize(model, "image.png")


def test_release_capture_handles_release_errors():
    class BrokenCapture:
        def release(self):
            raise OSError("already gone")

    asanai._release_capture(None)
    asanai._release_capture(BrokenCapture())


def test_visualize_webcam_capture_error(monkeypatch, quiet_console):
    monkeypatch.setattr(
        asanai.cv2,
        "VideoCapture",
        lambda *args: (_ for _ in ()).throw(cv2.error("no camera")),
    )
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)
    with pytest.raises(SystemExit) as exc:
        asanai.visualize_webcam(SimpleNamespace())
    assert exc.value.code == 1


def test_visualize_webcam_read_failure_releases_capture(monkeypatch, quiet_console):
    class Capture:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            return (False, None)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(asanai.cv2, "VideoCapture", lambda *args: capture)
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)
    with pytest.raises(SystemExit) as exc:
        asanai.visualize_webcam(SimpleNamespace())
    assert exc.value.code == 1
    assert capture.released


def test_classify_webcam_capture_error(monkeypatch, quiet_console):
    monkeypatch.setattr(
        asanai.cv2,
        "VideoCapture",
        lambda *args: (_ for _ in ()).throw(cv2.error("no camera")),
    )
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)
    with pytest.raises(SystemExit) as exc:
        asanai.classify_webcam(SimpleNamespace(), ["a"])
    assert exc.value.code == 1


def test_classify_webcam_prediction_error(monkeypatch, quiet_console):
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
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)
    model = SimpleNamespace(
        predict=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(SystemExit) as exc:
        asanai.classify_webcam(model, ["a"])
    assert exc.value.code == 1
    assert capture.released


def test_classify_webcam_read_failure_releases_capture(monkeypatch, quiet_console):
    class Capture:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            return (False, None)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(asanai.cv2, "VideoCapture", lambda *args: capture)
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)
    with pytest.raises(SystemExit) as exc:
        asanai.classify_webcam(SimpleNamespace(), ["a"])
    assert exc.value.code == 1
    assert capture.released


def test_classify_and_display_prediction_error(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "load", lambda *args, **kwargs: np.zeros((1, 2, 2, 3)))
    model = SimpleNamespace(
        predict=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    asanai.classify_and_display(model, "image.png", ["a", "b"])


@pytest.mark.parametrize(
    "error",
    [subprocess.CalledProcessError(1, "apt"), OSError("no sudo")],
)
def test_try_install_docker_linux_failure(monkeypatch, quiet_console, error):
    monkeypatch.setattr(asanai.shutil, "which", lambda name: "/usr/bin/apt")
    monkeypatch.setattr(
        asanai.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(error)
    )
    asanai.try_install_docker_linux()
    assert quiet_console.messages


def test_is_docker_running_psutil_error(monkeypatch):
    monkeypatch.setattr(
        asanai.psutil,
        "process_iter",
        lambda attrs: (_ for _ in ()).throw(psutil.AccessDenied(pid=1)),
    )
    assert asanai.is_docker_running() is False


def test_is_docker_running_process_without_name(monkeypatch):
    monkeypatch.setattr(
        asanai.psutil, "process_iter", lambda attrs: [SimpleNamespace(info={})]
    )
    assert asanai.is_docker_running() is False


def test_is_docker_running_broken_process_info(monkeypatch):
    class BrokenProcess:
        @property
        def info(self):
            raise AttributeError("gone")

    monkeypatch.setattr(
        asanai.psutil, "process_iter", lambda attrs: [BrokenProcess()]
    )
    assert asanai.is_docker_running() is False


def test_install_tensorflow_restart_failure(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai.util, "find_spec", lambda name: None)
    monkeypatch.setattr(asanai, "_in_virtual_env", lambda: True)
    monkeypatch.setattr(asanai, "_platform_wheel_warning", lambda: None)
    monkeypatch.setattr(asanai.platform, "system", lambda: "Linux")
    monkeypatch.setattr(asanai.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(asanai, "_pip_install", lambda pkg, quiet=False: True)
    monkeypatch.setattr(asanai, "_gpu_hint", lambda: None)
    monkeypatch.setattr(
        asanai.os, "execv", lambda *args: (_ for _ in ()).throw(OSError("exec failed"))
    )
    with pytest.raises(SystemExit) as exc:
        asanai.install_tensorflow(["script.py"])
    assert exc.value.code == 0


def test_converter_local_os_error(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "_is_command_available", lambda cmd: True)
    monkeypatch.setattr(
        asanai.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("gone")),
    )
    assert asanai._pip_install_tensorflowjs_converter_and_run_it(["x"]) is False


def test_run_docker_conversion_docker_missing(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "start_docker_if_not_running", lambda: False)
    monkeypatch.setattr(
        asanai.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("docker missing")),
    )
    assert asanai.run_docker_conversion(["x"]) is False


def test_run_docker_conversion_dockerfile_write_error(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "start_docker_if_not_running", lambda: False)
    monkeypatch.setattr(
        asanai.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0)
    )
    monkeypatch.setattr(
        asanai, "write_dockerfile", lambda path: (_ for _ in ()).throw(OSError("read-only"))
    )
    assert asanai.run_docker_conversion(["x"]) is False


def test_build_docker_image_os_error(monkeypatch, fake_progress, quiet_console):
    monkeypatch.setattr(
        asanai.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("docker missing")),
    )
    assert asanai.build_docker_image("image", "Dockerfile", ".") is False
