"""Real end-to-end test: TensorFlow.js model -> Keras -> prediction.

These tests are marked ``integration`` and are skipped unless TensorFlow and
tensorflowjs are installed. CI runs them in a separate job with real wheels.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import asanai

pytestmark = pytest.mark.integration

EXAMPLE = Path(__file__).with_name("example_script")


def test_tfjs_model_converts_to_keras_and_predicts(tmp_path):
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflowjs")

    for name in ("model(2).json", "model.weights(2).bin", "slimer.png"):
        shutil.copy(EXAMPLE / name, tmp_path / name)

    assert asanai.convert_to_keras_if_needed(tmp_path) is True
    model_path = tmp_path / "model.h5"
    assert model_path.is_file()

    model = tf.keras.models.load_model(model_path)
    assert asanai.model_is_simple_classification(model) is True

    image = asanai.load(tmp_path / "slimer.png", height=40, width=40)
    assert image is not None
    assert image.shape == (1, 40, 40, 3)

    prediction = model.predict(image, verbose=0)
    assert prediction.shape == (1, 5)
    assert np.all(np.isfinite(prediction))
    assert np.allclose(prediction.sum(axis=1), 1.0, atol=1e-5)
