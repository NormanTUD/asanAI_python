"""Image loading, annotation and prediction-visualization tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image, UnidentifiedImageError

import asanai


def _model(input_shape=(None, 4, 4, 3), output_shape=(None, 2), prediction=None):
    class Model:
        pass

    model = Model()
    model.input_shape = input_shape
    model.output_shape = output_shape

    def predict(data, verbose=0):
        if callable(prediction):
            return prediction(data)
        if prediction is not None:
            return prediction
        return np.zeros((1, 2), dtype="float32")

    model.predict = predict
    return model


def _image(path, size=(8, 8, 3)):
    array = np.zeros(size, dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path


def test_load_missing_file(tmp_path, quiet_console):
    assert asanai.load(tmp_path / "missing.png") is None


def test_load_success(tmp_path, quiet_console):
    path = _image(tmp_path / "image.png")
    result = asanai.load(path, height=4, width=4)
    assert result is not None
    assert result.shape == (1, 4, 4, 3)


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("denied"),
        UnidentifiedImageError("bad image"),
        OSError("os error"),
    ],
)
def test_load_open_errors(tmp_path, monkeypatch, quiet_console, error):
    path = _image(tmp_path / "image.png")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(asanai.Image, "open", fail)
    assert asanai.load(path) is None


@pytest.mark.parametrize("error", [ValueError("bad"), TypeError("type"), OSError("os")])
def test_load_processing_errors(tmp_path, monkeypatch, quiet_console, error):
    path = _image(tmp_path / "image.png")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(asanai.transform, "resize", fail)
    assert asanai.load(path) is None


def test_load_keyboard_interrupt(tmp_path, monkeypatch, quiet_console):
    path = _image(tmp_path / "image.png")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(asanai.Image, "open", interrupt)
    with pytest.raises(SystemExit) as exc:
        asanai.load(path)
    assert exc.value.code == 0


def test_load_frame_success():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result = asanai.load_frame(frame, height=4, width=4)
    assert result is not None
    assert result.shape == (1, 4, 4, 3)


@pytest.mark.parametrize(
    "error",
    [
        cv2.error("bad"),
        ValueError("bad"),
        TypeError("bad"),
        OSError("bad"),
    ],
)
def test_load_frame_errors(monkeypatch, quiet_console, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(asanai.cv2, "cvtColor", fail)
    assert asanai.load_frame(np.zeros((4, 4, 3), dtype=np.uint8)) is None


def test_load_frame_keyboard_interrupt(monkeypatch, quiet_console):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(asanai.cv2, "cvtColor", interrupt)
    with pytest.raises(SystemExit) as exc:
        asanai.load_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    assert exc.value.code == 0


def test_format_probabilities():
    assert asanai._format_probabilities(np.array([0.1, 0.2])) == ["0.100", "0.200"]
    assert asanai._format_probabilities(np.array([0.5, 0.5])) == ["0.5000000000", "0.5000000000"]


def test_annotate_frame_success(quiet_console):
    frame = np.zeros((120, 600, 3), dtype=np.uint8)
    predictions = np.array([[0.25, 0.75]])
    result = asanai.annotate_frame(frame, predictions, ["cat", "dog"])
    assert result is not None
    assert result.shape == frame.shape


def test_annotate_frame_missing_font_falls_back(monkeypatch):
    frame = np.zeros((120, 600, 3), dtype=np.uint8)
    predictions = np.array([[0.25, 0.75]])
    monkeypatch.setattr(asanai.os.path, "exists", lambda path: False)
    put_text = []
    monkeypatch.setattr(
        asanai.cv2,
        "putText",
        lambda *args, **kwargs: put_text.append(args),
    )
    result = asanai.annotate_frame(frame, predictions, ["cat", "dog"])
    assert result is not None
    assert put_text


def test_annotate_frame_label_mismatch(quiet_console):
    with pytest.raises(SystemExit) as exc:
        asanai.annotate_frame(
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.array([[0.2, 0.8]]),
            ["only-one"],
        )
    assert exc.value.code == 0


def test_get_shape(tmp_path, quiet_console):
    assert asanai.get_shape(tmp_path / "missing.txt") is None

    directory = tmp_path / "dir"
    directory.mkdir()
    assert asanai.get_shape(directory) is None

    valid = tmp_path / "valid.txt"
    valid.write_text("shape: (3, 224, 224)\n", encoding="utf-8")
    assert asanai.get_shape(valid) == [3, 224, 224]

    no_pattern = tmp_path / "no_pattern.txt"
    no_pattern.write_text("nothing here\n", encoding="utf-8")
    assert asanai.get_shape(no_pattern) is None

    no_ints = tmp_path / "no_ints.txt"
    no_ints.write_text("shape: (a, b)\n", encoding="utf-8")
    assert asanai.get_shape(no_ints) is None


def test_get_shape_encoding_error(tmp_path, quiet_console):
    path = tmp_path / "binary.txt"
    path.write_bytes(b"\xff\xfe\x00\x00")
    assert asanai.get_shape(path) is None


def test_get_shape_regex_error(tmp_path, monkeypatch, quiet_console):
    path = tmp_path / "valid.txt"
    path.write_text("shape: (3, 224)\n", encoding="utf-8")
    monkeypatch.setattr(
        asanai.re, "search", lambda *args, **kwargs: (_ for _ in ()).throw(asanai.re.error("bad"))
    )
    assert asanai.get_shape(path) is None


def test_get_shape_io_error(tmp_path, monkeypatch, quiet_console):
    path = tmp_path / "valid.txt"
    path.write_text("shape: (3, 224)\n", encoding="utf-8")

    def fail(*args, **kwargs):
        raise IOError("io")

    monkeypatch.setattr(asanai.Path, "open", fail)
    assert asanai.get_shape(path) is None


def test_load_and_prepare_image_branches(monkeypatch):
    model = _model(input_shape=(None, 4, 4, 3))
    monkeypatch.setattr(asanai, "load", lambda path: None)
    assert asanai._load_and_prepare_image("x.png", model) is None

    monkeypatch.setattr(asanai, "load", lambda path: np.zeros((4, 4)))
    assert asanai._load_and_prepare_image("x.png", model) is None

    monkeypatch.setattr(asanai, "load", lambda path: np.zeros((1, 4, 4, 3)))
    bad_model = _model(input_shape=(4, 4, 3))
    assert asanai._load_and_prepare_image("x.png", bad_model) is None

    monkeypatch.setattr(asanai, "load", lambda path: np.zeros((1, 8, 8, 3)))
    resized = asanai._load_and_prepare_image("x.png", model)
    assert resized.shape == (4, 4, 3)

    channel_model = _model(input_shape=(None, 4, 4, 1))
    assert asanai._load_and_prepare_image("x.png", channel_model) is None


def test_load_and_prepare_image_resize_error(monkeypatch):
    model = _model(input_shape=(None, 4, 4, 3))
    monkeypatch.setattr(asanai, "load", lambda path: np.zeros((1, 8, 8, 3)))

    def fail(*args, **kwargs):
        raise cv2.error("bad")

    monkeypatch.setattr(asanai.cv2, "resize", fail)
    assert asanai._load_and_prepare_image("x.png", model) is None


def test_normalize_and_convert_to_uint8():
    normal = asanai._normalize_and_convert_to_uint8(np.array([[0.0, 0.5, 1.0]]))
    assert normal.dtype == np.uint8
    assert normal.max() == 255

    flat = asanai._normalize_and_convert_to_uint8(np.array([1.0, 1.0]))
    assert np.array_equal(flat, np.zeros(2, dtype=np.uint8))


def test_show_image_in_window(monkeypatch):
    shown = []
    monkeypatch.setattr(asanai.cv2, "imshow", lambda name, image: shown.append(name))
    monkeypatch.setattr(asanai.cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(asanai.cv2, "getWindowProperty", lambda *args: 0)
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)
    asanai._show_image_in_window(np.zeros((2, 2, 3), dtype=np.uint8), "window")
    assert shown == ["window"]

    monkeypatch.setattr(asanai.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(asanai.cv2, "waitKey", lambda delay: 113)
    asanai._show_image_in_window(np.zeros((2, 2, 3), dtype=np.uint8), "window")


def test_visualize_color_and_grayscale(monkeypatch):
    shown = []
    monkeypatch.setattr(
        asanai, "_show_image_in_window", lambda image, name: shown.append((image, name))
    )
    asanai._visualize_color_image(np.zeros((4, 4, 3), dtype=np.float32))
    asanai._visualize_grayscale_channels(np.zeros((4, 4, 2), dtype=np.float32))
    assert len(shown) == 2
    assert shown[1][0].shape == (4, 8)


def test_visualize_branches(monkeypatch, quiet_console):
    model = _model()
    monkeypatch.setattr(asanai, "_load_and_prepare_image", lambda path, model: None)
    asanai.visualize(model, "x.png")

    monkeypatch.setattr(
        asanai, "_load_and_prepare_image", lambda path, model: np.zeros((4, 4, 3))
    )
    color = []
    gray = []
    monkeypatch.setattr(asanai, "_visualize_color_image", lambda image: color.append(True))
    monkeypatch.setattr(asanai, "_visualize_grayscale_channels", lambda image: gray.append(True))

    asanai.visualize(_model(prediction=np.zeros((1, 2, 2, 3))), "x.png")
    assert color == [True]

    asanai.visualize(_model(prediction=np.zeros((1, 2, 2, 1))), "x.png")
    assert gray == [True]

    asanai.visualize(_model(prediction=np.zeros((1, 2))), "x.png")
    asanai.visualize(_model(prediction=np.zeros((1, 2, 2))), "x.png")


def test_visualize_handles_cv2_error(monkeypatch, quiet_console):
    model = _model()
    monkeypatch.setattr(
        asanai, "_load_and_prepare_image", lambda path, model: np.zeros((4, 4, 3))
    )

    def fail(*args, **kwargs):
        raise cv2.error("bad")

    model.predict = fail
    asanai.visualize(model, "x.png")


class _Capture:
    def __init__(self, reads, opened=True):
        self.reads = list(reads)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.reads:
            return (False, None)
        item = self.reads.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def release(self):
        self.released = True


def _patch_capture(monkeypatch, capture):
    monkeypatch.setattr(asanai.cv2, "VideoCapture", lambda *args: capture)


def test_visualize_webcam_not_opened(monkeypatch, quiet_console):
    _patch_capture(monkeypatch, _Capture([], opened=False))
    asanai.visualize_webcam(_model())


def test_visualize_webcam_read_failure(monkeypatch, quiet_console):
    _patch_capture(monkeypatch, _Capture([]))
    with pytest.raises(SystemExit) as exc:
        asanai.visualize_webcam(_model())
    assert exc.value.code == 1


def test_visualize_webcam_color_path(monkeypatch, quiet_console):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = _Capture([(True, frame)])
    _patch_capture(monkeypatch, capture)
    monkeypatch.setattr(asanai, "load_frame", lambda *args: np.zeros((1, 4, 4, 3)))
    monkeypatch.setattr(asanai.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(asanai.cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(asanai.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)

    asanai.visualize_webcam(_model(prediction=np.zeros((1, 4, 4, 3))))
    assert capture.released


def test_visualize_webcam_grayscale_and_unknown_paths(monkeypatch, quiet_console):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for prediction in (
        np.zeros((1, 4, 4, 1)),
        np.zeros((1, 4, 4, 2)),
        np.zeros((1, 4)),
    ):
        capture = _Capture([(True, frame)])
        _patch_capture(monkeypatch, capture)
        monkeypatch.setattr(asanai, "load_frame", lambda *args: np.zeros((1, 4, 4, 3)))
        monkeypatch.setattr(asanai.cv2, "imshow", lambda *args: None)
        monkeypatch.setattr(asanai.cv2, "waitKey", lambda delay: ord("q"))
        monkeypatch.setattr(asanai.cv2, "getWindowProperty", lambda *args: 1)
        monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)
        asanai.visualize_webcam(_model(prediction=prediction))


def test_visualize_webcam_frame_preprocess_failure(monkeypatch, quiet_console):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = _Capture([(True, frame)])
    _patch_capture(monkeypatch, capture)
    monkeypatch.setattr(asanai, "load_frame", lambda *args: None)
    monkeypatch.setattr(asanai.cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(asanai.cv2, "getWindowProperty", lambda *args: 0)
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)
    asanai.visualize_webcam(_model())


def test_visualize_webcam_window_closed(monkeypatch, quiet_console):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = _Capture([(True, frame)])
    _patch_capture(monkeypatch, capture)
    monkeypatch.setattr(asanai, "load_frame", lambda *args: np.zeros((1, 4, 4, 3)))
    monkeypatch.setattr(asanai.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(asanai.cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(asanai.cv2, "getWindowProperty", lambda *args: 0)
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)
    asanai.visualize_webcam(_model(prediction=np.zeros((1, 4, 4, 3))))
    assert capture.released


def test_visualize_webcam_keyboard_interrupt(monkeypatch, quiet_console):
    capture = _Capture([KeyboardInterrupt()])
    _patch_capture(monkeypatch, capture)
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)
    with pytest.raises(SystemExit) as exc:
        asanai.visualize_webcam(_model())
    assert exc.value.code == 0
    assert capture.released


def test_visualize_webcam_error(monkeypatch, quiet_console):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = _Capture([(True, frame)])
    _patch_capture(monkeypatch, capture)
    monkeypatch.setattr(asanai, "load_frame", lambda *args: np.zeros((1, 4, 4, 3)))
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)

    model = _model()
    model.predict = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(SystemExit) as exc:
        asanai.visualize_webcam(model)
    assert exc.value.code == 1
    assert capture.released


def test_auto_wrap_namespace():
    def plain(value: int) -> int:
        return value

    namespace = {"plain": plain, "auto_wrap_namespace": plain, "value": 3}
    wrapped = asanai.auto_wrap_namespace(namespace)
    assert wrapped["plain"](4) == 4
    assert wrapped["auto_wrap_namespace"] is plain
    assert wrapped["value"] == 3
    with pytest.raises(Exception):
        wrapped["plain"]("wrong")


def test_show_result(capsys):
    asanai.show_result({"answer": 42})
    assert "42" in capsys.readouterr().out
