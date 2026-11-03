"""Property-based and fuzz-style tests for parsers, shapes and ZIP safety."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import asanai

SETTINGS = settings(max_examples=60, deadline=None)


@SETTINGS
@given(st.lists(st.floats(allow_nan=True, allow_infinity=True), max_size=20))
def test_is_float_list_accepts_float_strings(values):
    assert asanai._is_float_list([str(value) for value in values]) is True


@SETTINGS
@given(st.text(min_size=1, alphabet="abcdefghijklmnopqrstuvwxyz ?!"))
def test_is_float_list_rejects_non_numeric_text(value):
    assert asanai._is_float_list([value]) is False


@SETTINGS
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=20))
def test_format_probabilities_preserves_length(values):
    formatted = asanai._format_probabilities(np.array(values))
    assert len(formatted) == len(values)
    assert all(isinstance(item, str) for item in formatted)


@SETTINGS
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=20))
def test_convert_to_ndarray_round_trip(values):
    result = asanai._convert_to_ndarray([str(value) for value in values], (len(values),))
    assert result.shape == (len(values),)
    assert np.allclose(result, np.array(values))


@SETTINGS
@given(
    st.lists(
        st.floats(allow_nan=False, allow_infinity=False, width=32),
        min_size=1,
        max_size=20,
    )
)
def test_normalize_and_convert_to_uint8_range(values):
    result = asanai._normalize_and_convert_to_uint8(np.array(values))
    assert result.dtype == np.uint8
    assert result.shape == (len(values),)
    assert int(result.min()) >= 0
    assert int(result.max()) <= 255


@SETTINGS
@given(
    st.integers(min_value=1, max_value=10000),
    st.integers(min_value=1, max_value=10000),
    st.integers(min_value=1, max_value=10000),
)
def test_get_shape_round_trip(height, width, channels):
    path = Path("shape.txt")
    expected = [height, width, channels]
    path.write_text(f"shape: ({height}, {width}, {channels})\n", encoding="utf-8")
    assert asanai.get_shape(path) == expected


@SETTINGS
@given(st.integers(min_value=2, max_value=1000))
def test_has_classification_output_shape_accepts_class_dimension(class_dim):
    model = type("Model", (), {"output_shape": (None, class_dim)})()
    assert asanai._has_classification_output_shape(model) is True


@SETTINGS
@given(
    st.integers(min_value=2, max_value=1000),
    st.sampled_from([None, -1]),
)
def test_output_is_simple_image_accepts_rgb(height, batch):
    shape = (batch, height, height, 3)
    model = type("Model", (), {"output_shape": shape})()
    assert asanai.output_is_simple_image(model) is True


@SETTINGS
@given(
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=1000),
    st.integers(min_value=1, max_value=1000),
    st.sampled_from([None, -1]),
)
def test_output_is_complex_image_accepts_positive_channels(n, m, a, batch):
    shape = (batch, n, m, a)
    model = type("Model", (), {"output_shape": shape})()
    assert asanai.output_is_complex_image(model) is True


@SETTINGS
@given(
    st.lists(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
            min_size=1,
            max_size=12,
        ),
        min_size=1,
        max_size=4,
    )
)
def test_safe_zip_target_stays_inside_working_directory(parts):
    name = "/".join(parts)
    target = asanai._safe_zip_target(name)
    assert target is not None
    assert target.startswith(asanai.os.path.abspath(".") + asanai.os.sep)


@SETTINGS
@given(st.text(min_size=1, max_size=30))
def test_copy_and_patch_tfjs_never_raises_on_random_json(text):
    model = Path("model.json")
    weights = Path("weights.bin")
    weights.write_bytes(b"weights")

    data = {"weightsManifest": [{"paths": [text], "weights": text}], "meta": text}
    model.write_text(json.dumps(data), encoding="utf-8")

    result = asanai.copy_and_patch_tfjs(str(model), str(weights), "fuzz")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(item, str) for item in result)
