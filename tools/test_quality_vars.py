# -*- coding: utf-8 -*-
import json, ast
from services.code_quality_analyzer import analyze_quality

SNIPPET = r"""
from tensorflow import keras
def build():
    optimizer = keras.optimizers.get('adam')
    loss_fn   = 'sparse_categorical_crossentropy'
    metrics   = ['accuracy']
    model = type('M', (), {'compile': lambda *a, **k: None})()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model
"""

SPEC = {
    "optimizer_name": "adam",
    "loss": "sparse_categorical_crossentropy",
    "metrics": ["accuracy"],
}


def main():
    report = analyze_quality(SNIPPET, SPEC)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    issues = {i["code"] for i in report["issues"]}
    assert "OPT_MISMATCH" not in issues
    assert "LOSS_MISMATCH" not in issues
    assert "METRICS_MISMATCH" not in issues
    print("✅ variable-aware compile check OK")


if __name__ == "__main__":
    main()
