"""Shared fixtures and lightweight fakes for the asanai test suite.

The tests deliberately avoid importing TensorFlow/Keras at collection time:
those packages are huge, optional at runtime for most helpers, and are only
imported lazily inside ``asanai._is_softmax_output_layer``.  The ``fake_tf``
fixture installs a minimal stand-in so that branch can be exercised even on
machines (and in CI jobs) where TensorFlow is not installed.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import asanai


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch, tmp_path):
    """Keep tests independent from the caller's cwd and proxy/CI settings."""
    monkeypatch.chdir(tmp_path)
    for name in ("CI", "HTTP_PROXY", "HTTPS_PROXY", "CONDA_PREFIX", "VIRTUAL_ENV"):
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def fake_tf(monkeypatch):
    """Install a minimal fake ``tensorflow.keras.layers`` module tree."""

    class Dense:
        def __init__(self, activation=None):
            self.activation = activation

    class Activation:
        def __init__(self, activation=None):
            self.activation = activation

    class Softmax:
        pass

    def softmax(x):
        return x

    def relu(x):
        return x

    layers = types.ModuleType("tensorflow.keras.layers")
    layers.Dense = Dense
    layers.Activation = Activation
    layers.Softmax = Softmax

    keras = types.ModuleType("tensorflow.keras")
    keras.layers = layers

    tensorflow = types.ModuleType("tensorflow")
    tensorflow.keras = keras

    for name, module in (
        ("tensorflow", tensorflow),
        ("tensorflow.keras", keras),
        ("tensorflow.keras.layers", layers),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    return SimpleNamespace(
        Dense=Dense,
        Activation=Activation,
        Softmax=Softmax,
        softmax=softmax,
        relu=relu,
    )


@pytest.fixture
def quiet_console(monkeypatch):
    """Replace rich's Console with a no-op recorder so tests stay readable."""

    class _Status:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    class QuietConsole:
        def __init__(self, *args, **kwargs):
            self.messages = []
            self._live = None

        def print(self, *args, **kwargs):
            self.messages.append(args)

        def log(self, *args, **kwargs):
            self.messages.append(args)

        def rule(self, *args, **kwargs):
            self.messages.append(args)

        def status(self, *args, **kwargs):
            return _Status()

        def get_time(self):
            return time.monotonic()

        def set_live(self, live):
            self._live = live

        def clear_live(self):
            self._live = None

        def is_terminal(self):
            return False

    console = QuietConsole()
    monkeypatch.setattr(asanai, "console", console)
    return console


@pytest.fixture
def fake_progress(monkeypatch):
    """Replace rich's Progress with a tiny context manager.

    ``_pip_install`` nests ``console.status`` and ``Progress`` on the same
    console, which tripping rich's "only one live display" guard in a
    non-interactive test run.  The tests care about the control flow, not rich's
    rendering, so a stub keeps them deterministic on every platform.
    """

    class FakeProgress:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def add_task(self, *args, **kwargs):
            return 1

        def update(self, *args, **kwargs):
            pass

        def start_task(self, *args, **kwargs):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(asanai, "Progress", FakeProgress)
    return FakeProgress


def make_model(output_shape=None, input_shape=None, layers=None, prediction=None):
    """Create a tiny object that quacks like a Keras model."""

    class Model:
        pass

    model = Model()
    if output_shape is not None:
        model.output_shape = output_shape
    if input_shape is not None:
        model.input_shape = input_shape
    if layers is not None:
        model.layers = layers

    def predict(data, verbose=0):
        if callable(prediction):
            return prediction(data)
        if prediction is not None:
            return prediction
        return np.zeros((1, 2), dtype="float32")

    model.predict = predict
    return model


def make_capture(read_results=None, opened=True):
    """Fake ``cv2.VideoCapture`` with scripted ``read()`` results."""

    results = list(read_results or [(True, np.zeros((10, 10, 3), dtype=np.uint8))])

    class Capture:
        def __init__(self, *args, **kwargs):
            self.released = False

        def isOpened(self):
            return opened

        def read(self):
            if results:
                return results.pop(0)
            return (False, None)

        def release(self):
            self.released = True

    return Capture()


def write_image(path: Path, size=(8, 8, 3)):
    from PIL import Image

    array = np.zeros(size, dtype=np.uint8)
    Image.fromarray(array).save(path)
    return path
