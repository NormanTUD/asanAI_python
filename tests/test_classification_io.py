"""Model input loading, output-shape detection and classification display."""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import asanai


def test_is_float_list():
    assert asanai._is_float_list(["1", "2.5"]) is True
    assert asanai._is_float_list(["not-a-number"]) is False


def test_convert_to_ndarray():
    result = asanai._convert_to_ndarray(["1", "2", "3", "4"], (2, 2))
    assert result.shape == (2, 2)
    assert result.dtype == float


def test_exit_with_error(quiet_console):
    with pytest.raises(SystemExit) as exc:
        asanai._exit_with_error("no")
    assert exc.value.code == 1


def test_load_or_input_model_data_from_file(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("1.0 2.0\n", encoding="utf-8")
    model = SimpleNamespace(input_shape=(None, 2))
    result = asanai.load_or_input_model_data(model, str(path))
    assert result.shape == (2,)
    assert result[0] == 1.0


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing"),
        IsADirectoryError("dir"),
        ValueError("bad"),
        OSError("io"),
    ],
)
def test_load_or_input_model_data_file_errors(tmp_path, monkeypatch, error):
    path = tmp_path / "data.txt"
    path.write_text("data", encoding="utf-8")
    model = SimpleNamespace(input_shape=(None, 2))
    monkeypatch.setattr(asanai.os.path, "isfile", lambda filename: True)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(asanai.np, "loadtxt", fail)
    with pytest.raises(SystemExit) as exc:
        asanai.load_or_input_model_data(model, str(path))
    assert exc.value.code == 1


def test_load_or_input_model_data_size_mismatch(tmp_path, quiet_console):
    path = tmp_path / "data.txt"
    path.write_text("1 2 3\n", encoding="utf-8")
    model = SimpleNamespace(input_shape=(None, 2))
    with pytest.raises(SystemExit) as exc:
        asanai.load_or_input_model_data(model, str(path))
    assert exc.value.code == 1


def test_load_or_input_model_data_reshape_error(tmp_path, monkeypatch, quiet_console):
    path = tmp_path / "data.txt"
    path.write_text("data", encoding="utf-8")
    model = SimpleNamespace(input_shape=(None, 2, 2))

    class FakeData:
        size = 4

        def reshape(self, shape):
            raise ValueError("cannot reshape")

    monkeypatch.setattr(asanai.np, "loadtxt", lambda filename: FakeData())
    with pytest.raises(SystemExit) as exc:
        asanai.load_or_input_model_data(model, str(path))
    assert exc.value.code == 1


def test_load_or_input_model_data_non_float(tmp_path, monkeypatch, quiet_console):
    path = tmp_path / "data.txt"
    path.write_text("1 2\n", encoding="utf-8")
    model = SimpleNamespace(input_shape=(None, 2))
    monkeypatch.setattr(asanai.np, "loadtxt", lambda filename: np.array([1, 2], dtype=int))
    with pytest.raises(SystemExit) as exc:
        asanai.load_or_input_model_data(model, str(path))
    assert exc.value.code == 1


def test_load_or_input_model_data_manual_success(monkeypatch):
    model = SimpleNamespace(input_shape=(None, 2))
    monkeypatch.setattr(builtins, "input", lambda: "1.0 2.0")
    result = asanai.load_or_input_model_data(model, "missing.txt")
    assert list(result) == [1.0, 2.0]


def test_load_or_input_model_data_manual_retries(monkeypatch, quiet_console):
    model = SimpleNamespace(input_shape=(None, 2))
    answers = iter(["1.0", "a b", "1.0 2.0"])
    monkeypatch.setattr(builtins, "input", lambda: next(answers))
    result = asanai.load_or_input_model_data(model, "missing.txt")
    assert list(result) == [1.0, 2.0]


def test_load_or_input_model_data_manual_conversion_error(monkeypatch, quiet_console):
    model = SimpleNamespace(input_shape=(None, 2))
    answers = iter(["1.0 2.0", "3.0 4.0"])
    monkeypatch.setattr(builtins, "input", lambda: next(answers))
    calls = []

    def convert(values, shape):
        calls.append(values)
        if len(calls) == 1:
            raise ValueError("bad")
        return np.array([3.0, 4.0])

    monkeypatch.setattr(asanai, "_convert_to_ndarray", convert)
    result = asanai.load_or_input_model_data(model, "missing.txt")
    assert list(result) == [3.0, 4.0]
    assert len(calls) == 2


def test_load_or_input_model_data_manual_keyboard_interrupt(monkeypatch, quiet_console):
    model = SimpleNamespace(input_shape=(None, 2))
    monkeypatch.setattr(
        builtins, "input", lambda: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(SystemExit) as exc:
        asanai.load_or_input_model_data(model, "missing.txt")
    assert exc.value.code == 1


def test_model_is_simple_classification_invalid(fake_tf):
    class Empty:
        layers = []

    assert asanai.model_is_simple_classification(object()) is False
    assert asanai.model_is_simple_classification(Empty()) is False

    class Broken:
        @property
        def layers(self):
            raise ValueError("broken")

    assert asanai.model_is_simple_classification(Broken()) is False


def test_model_is_simple_classification_dense(fake_tf):
    softmax_layer = fake_tf.Dense(fake_tf.softmax)
    model = SimpleNamespace(layers=[softmax_layer], output_shape=(None, 3))
    assert asanai.model_is_simple_classification(model) is True

    relu_layer = fake_tf.Dense(fake_tf.relu)
    model = SimpleNamespace(layers=[relu_layer], output_shape=(None, 3))
    assert asanai.model_is_simple_classification(model) is False

    no_activation = fake_tf.Dense()
    del no_activation.activation
    model = SimpleNamespace(layers=[no_activation], output_shape=(None, 3))
    assert asanai.model_is_simple_classification(model) is False


def test_model_is_simple_classification_activation_layer(fake_tf):
    model = SimpleNamespace(
        layers=[fake_tf.Activation(fake_tf.softmax)], output_shape=(None, 3)
    )
    assert asanai.model_is_simple_classification(model) is True

    model = SimpleNamespace(
        layers=[fake_tf.Activation(fake_tf.relu)], output_shape=(None, 3)
    )
    assert asanai.model_is_simple_classification(model) is False

    layer = fake_tf.Activation()
    del layer.activation
    model = SimpleNamespace(layers=[layer], output_shape=(None, 3))
    assert asanai.model_is_simple_classification(model) is False


def test_model_is_simple_classification_softmax_and_other(fake_tf):
    model = SimpleNamespace(layers=[fake_tf.Softmax()], output_shape=(None, 3))
    assert asanai.model_is_simple_classification(model) is True

    model = SimpleNamespace(layers=[SimpleNamespace()], output_shape=(None, 3))
    assert asanai.model_is_simple_classification(model) is False


@pytest.mark.parametrize(
    ("output_shape", "expected"),
    [
        (None, False),
        ((None,), False),
        ((None, 2, 3), False),
        ((2, 3), False),
        ((None, "3"), False),
        ((None, 1), False),
        ((None, 2), True),
        ((-1, 5), True),
    ],
)
def test_has_classification_output_shape(output_shape, expected):
    model = SimpleNamespace() if output_shape is None else SimpleNamespace(output_shape=output_shape)
    if output_shape is None:
        model = object()
    assert asanai._has_classification_output_shape(model) is expected


class _LenNotIterable:
    def __len__(self):
        return 4

    def __iter__(self):
        raise TypeError("not iterable")


class _LenMismatch:
    def __len__(self):
        return 4

    def __iter__(self):
        return iter((None, 2, 2))


def test_output_is_simple_image_errors():
    assert asanai.output_is_simple_image(object()) is False
    assert asanai.output_is_simple_image(SimpleNamespace(output_shape=5)) is False
    assert asanai.output_is_simple_image(SimpleNamespace(output_shape=(None, 2, 2))) is False
    assert (
        asanai.output_is_simple_image(SimpleNamespace(output_shape=_LenNotIterable()))
        is False
    )
    assert (
        asanai.output_is_simple_image(SimpleNamespace(output_shape=_LenMismatch())) is False
    )


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((2, 2, 2, 3), False),
        ((None, 0, 2, 3), False),
        ((None, "2", 2, 3), False),
        ((None, 2, 0, 3), False),
        ((None, 2, "2", 3), False),
        ((None, 2, 2, 1), False),
        ((None, 2, 2, 3), True),
        ((-1, 2, 2, 3), True),
    ],
)
def test_output_is_simple_image_shapes(shape, expected):
    assert asanai.output_is_simple_image(SimpleNamespace(output_shape=shape)) is expected


def test_output_is_complex_image_errors():
    with pytest.raises(AttributeError):
        asanai.output_is_complex_image(object())
    assert asanai.output_is_complex_image(SimpleNamespace(output_shape=(None, 2, 2))) is False
    assert (
        asanai.output_is_complex_image(SimpleNamespace(output_shape=_LenNotIterable()))
        is False
    )
    assert (
        asanai.output_is_complex_image(SimpleNamespace(output_shape=_LenMismatch()))
        is False
    )


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((2, 2, 2, 3), False),
        ((None, 0, 2, 3), False),
        ((None, "2", 2, 3), False),
        ((None, 2, 0, 3), False),
        ((None, 2, "2", 3), False),
        ((None, 2, 2, 0), False),
        ((None, 2, 2, "3"), False),
        ((None, 2, 2, 1), True),
        ((-1, 2, 2, 4), True),
    ],
)
def test_output_is_complex_image_shapes(shape, expected):
    assert asanai.output_is_complex_image(SimpleNamespace(output_shape=shape)) is expected


def test_classify_and_display_load_failure(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "load", lambda *args, **kwargs: None)
    asanai.classify_and_display(SimpleNamespace(), "missing.png", ["a"])


def test_classify_and_display_label_shortage(monkeypatch, quiet_console):
    monkeypatch.setattr(asanai, "load", lambda *args, **kwargs: np.zeros((1, 2, 2, 3)))
    model = SimpleNamespace(predict=lambda image, verbose=0: np.array([[0.2, 0.8]]))
    with pytest.raises(SystemExit) as exc:
        asanai.classify_and_display(model, "image.png", ["only-one"])
    assert exc.value.code == 1


def test_classify_and_display_success(monkeypatch):
    monkeypatch.setattr(asanai, "load", lambda *args, **kwargs: np.zeros((1, 2, 2, 3)))
    model = SimpleNamespace(
        predict=lambda image, verbose=0: np.array([[0.2, 0.8], [0.4, 0.6]])
    )
    asanai.classify_and_display(model, "image.png", ["cat", "dog"])


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


def test_classify_webcam_not_opened(monkeypatch, quiet_console):
    _patch_capture(monkeypatch, _Capture([], opened=False))
    asanai.classify_webcam(SimpleNamespace(), ["a"])


def test_classify_webcam_read_failure(monkeypatch, quiet_console):
    _patch_capture(monkeypatch, _Capture([]))
    with pytest.raises(SystemExit) as exc:
        asanai.classify_webcam(SimpleNamespace(), ["a"])
    assert exc.value.code == 1


def test_classify_webcam_success(monkeypatch, quiet_console):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = _Capture([(True, frame)])
    _patch_capture(monkeypatch, capture)
    monkeypatch.setattr(asanai, "load_frame", lambda *args: np.zeros((1, 4, 4, 3)))
    monkeypatch.setattr(asanai, "annotate_frame", lambda *args: frame)
    monkeypatch.setattr(asanai, "print_predictions_line", lambda *args: None)
    monkeypatch.setattr(asanai.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(asanai.cv2, "waitKey", lambda delay: ord("q"))
    monkeypatch.setattr(asanai.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)

    model = SimpleNamespace(predict=lambda image, verbose=0: np.array([[0.5, 0.5]]))
    asanai.classify_webcam(model, ["a", "b"])
    assert capture.released


def test_classify_webcam_window_closed(monkeypatch, quiet_console):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = _Capture([(True, frame)])
    _patch_capture(monkeypatch, capture)
    monkeypatch.setattr(asanai, "load_frame", lambda *args: np.zeros((1, 4, 4, 3)))
    monkeypatch.setattr(asanai, "annotate_frame", lambda *args: frame)
    monkeypatch.setattr(asanai, "print_predictions_line", lambda *args: None)
    monkeypatch.setattr(asanai.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(asanai.cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(asanai.cv2, "getWindowProperty", lambda *args: 0)
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)

    model = SimpleNamespace(predict=lambda image, verbose=0: np.array([[0.5, 0.5]]))
    asanai.classify_webcam(model, ["a", "b"])
    assert capture.released


def test_classify_webcam_frame_none(monkeypatch, quiet_console):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = _Capture([(True, frame), (False, None)])
    _patch_capture(monkeypatch, capture)
    monkeypatch.setattr(asanai, "load_frame", lambda *args: np.zeros((1, 4, 4, 3)))
    monkeypatch.setattr(asanai, "annotate_frame", lambda *args: None)
    monkeypatch.setattr(asanai, "print_predictions_line", lambda *args: None)
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)

    model = SimpleNamespace(predict=lambda image, verbose=0: np.array([[0.5, 0.5]]))
    with pytest.raises(SystemExit):
        asanai.classify_webcam(model, ["a", "b"])


def test_classify_webcam_preprocess_none(monkeypatch, quiet_console):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = _Capture([(True, frame), (False, None)])
    _patch_capture(monkeypatch, capture)
    monkeypatch.setattr(asanai, "load_frame", lambda *args: None)
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)
    with pytest.raises(SystemExit):
        asanai.classify_webcam(SimpleNamespace(), ["a"])


def test_classify_webcam_imshow_error(monkeypatch, quiet_console):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = _Capture([(True, frame)])
    _patch_capture(monkeypatch, capture)
    monkeypatch.setattr(asanai, "load_frame", lambda *args: np.zeros((1, 4, 4, 3)))
    monkeypatch.setattr(asanai, "annotate_frame", lambda *args: frame)
    monkeypatch.setattr(asanai, "print_predictions_line", lambda *args: None)
    monkeypatch.setattr(
        asanai.cv2,
        "imshow",
        lambda *args: (_ for _ in ()).throw(cv2.error("bad")),
    )
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)

    model = SimpleNamespace(predict=lambda image, verbose=0: np.array([[0.5, 0.5]]))
    with pytest.raises(SystemExit) as exc:
        asanai.classify_webcam(model, ["a", "b"])
    assert exc.value.code == 1


def test_classify_webcam_keyboard_interrupt(monkeypatch, quiet_console):
    capture = _Capture([KeyboardInterrupt()])
    _patch_capture(monkeypatch, capture)
    monkeypatch.setattr(asanai.cv2, "destroyAllWindows", lambda: None)
    with pytest.raises(SystemExit) as exc:
        asanai.classify_webcam(SimpleNamespace(), ["a"])
    assert exc.value.code == 0
    assert capture.released
