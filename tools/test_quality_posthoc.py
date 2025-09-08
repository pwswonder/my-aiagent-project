# -*- coding: utf-8 -*-
"""
회귀 방지: compile에 변수로 인자가 전달되는 경우(중첩 변수 포함)도
analyzer가 정확히 스펙과 일치 판정하는지 테스트.
"""
from services.code_quality_analyzer import analyze_quality

SRC = """
from tensorflow import keras
import numpy as np
import random

random.seed(seed)
np.random.seed(seed)

def build():
    optimizer_name = "adam"
    optimizer = keras.optimizers.get(optimizer_name)  # 중첩 변수 전달 케이스
    loss_fn = "sparse_categorical_crossentropy"
    metrics = ["accuracy"]
    class M:
        def compile(self, **kw): pass
    model = M()
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model
"""
SPEC = {
    "optimizer_name": "adam",
    "loss": "sparse_categorical_crossentropy",
    "metrics": ["accuracy"],
}

if __name__ == "__main__":
    report = analyze_quality(SRC, SPEC)
    print(report["score"], report["issues"])
    assert all(
        i["code"] not in {"OPT_MISMATCH", "LOSS_MISMATCH", "METRICS_MISMATCH"}
        for i in report["issues"]
    )
    print("✅ variable/deep-resolve post-hoc: PASS")
