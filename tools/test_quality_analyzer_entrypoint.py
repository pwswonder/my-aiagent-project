# tests/test_quality_analyzer_entrypoint.py
import json
from services.code_quality_analyzer import analyze_quality

SRC_BUILD = """
def build():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential([layers.Dense(2)])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
"""

SRC_BUILD_MODEL = SRC_BUILD.replace("def build():", "def build_model():")

SPEC = {
    "optimizer_name": "adam",
    "loss": "sparse_categorical_crossentropy",
    "metrics": ["accuracy"],
}


def _codes(issues):
    return {i["code"] for i in issues}


def test_nonstandard_only_low():
    out = analyze_quality(SRC_BUILD, SPEC)
    codes = _codes(out["issues"])
    assert "NO_BUILD_MODEL" not in codes
    assert "ENTRYPOINT_NONSTANDARD" in codes
    assert out["score"] >= 90  # 여유 점수(필요시 조정)


def test_canonical_no_issue():
    out = analyze_quality(SRC_BUILD_MODEL, SPEC)
    codes = _codes(out["issues"])
    assert "NO_BUILD_MODEL" not in codes
    assert "ENTRYPOINT_NONSTANDARD" not in codes
    assert out["score"] == 100
