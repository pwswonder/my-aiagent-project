# -*- coding: utf-8 -*-
"""
demo_codegen_quick.py

용도
- 15개 템플릿에 대해 spec을 자동으로 구성해 빠르게 스모크 테스트
- build_model() 호출 및 model.summary()까지 실행하여 컴파일 성공을 확인

주의
- 텐서플로 미설치 시 ImportError 발생 가능
- 무거운 백본(ResNet/Etc.)도 weights=None이므로 네트워크 다운로드는 없음
"""

import os
import traceback

from services.template_registry import build_manifest
from services import codegen

# ---- 1) 매니페스트 갱신 ----
manifest = build_manifest(write_file=True)
models = sorted(manifest["models"].keys())


# ---- 2) 빠른 테스트용 기본 스펙(작게) ----
def tiny_spec_for(model_key: str):
    """
    각 모델별로 빠른 스모크 테스트에 적합한 작은 입력 형태/파라미터를 지정
    - metrics는 codegen에서 ['mse']로 자동 하드닝됨(없을 때)
    """
    # 공통 기본
    spec = {
        "learning_rate": 1e-3,
        "optimizer_name": "adam",
        # "metrics": ["accuracy"]  # 굳이 지정 안하면 ['mse'] 기본
    }

    # 이미지 계열(분류)
    if model_key in [
        "cnn",
        "resnet",
        "vgg",
        "mobilenet",
        "efficientnet",
        "densenet",
    ]:
        spec.update(
            {
                "input_shape": [64, 64, 3],
                "num_classes": 3,
                "conv_filters": [16, 32],  # cnn, densenet 등 일부 템플릿에서 사용
                "dense_units": [64],
                "dropout_rate": 0.1,
                "growth_rate": 16,  # densenet
                "blocks": [2, 2],  # densenet
                "alpha": 0.5,  # mobilenet
                "version": "B0",  # efficientnet
            }
        )
        return spec
    
    elif model_key=="inception":
        spec |= {
            "input_shape":[96,96,3],   # ★ 기존 64→96
            "num_classes":3
        }
    # 나머지 공통 분류 파라미터는 상단 이미지계열에서 이미 spec에 들어있음
        return spec

    # 세그멘테이션(UNet)
    if model_key == "unet":
        spec.update(
            {
                "input_shape": [128, 128, 3],
                "num_classes": 2,
                "base_filters": 16,
                "depth": 3,
            }
        )
        return spec

    # 오토인코딩 계열
    if model_key == "autoencoder":
        spec.update(
            {
                "input_shape": [32, 32, 3],
                "num_classes": 1,
                "encoder_filters": [16, 32],
                "decoder_filters": [32, 16],
                "latent_dim": 16,
            }
        )
        return spec

    if model_key == "vae":
        spec.update(
            {
                "input_shape": [32, 32, 3],
                "num_classes": 1,
                "latent_dim": 8,
                "conv_filters": [16, 32],
                "decoder_units": [64, 128],
            }
        )
        return spec

    # GAN
    if model_key == "gan":
        spec.update(
            {
                "input_shape": [32, 32, 3],
                "num_classes": 1,
                "noise_dim": 32,
                "gen_units": [64, 128],
                "disc_units": [128, 64],
            }
        )
        return spec

    # 시계열 계열
    if model_key in ["gru", "lstm", "transformer"]:
        spec.update(
            {
                "input_shape": [16, 8],  # (timesteps=16, features=8)
                "num_classes": 1,
                "rnn_units": [32, 16],  # gru
                "lstm_units": [32, 16],  # lstm
                "bidirectional": False,
                "return_sequences": False,
                "d_model": 32,  # transformer
                "num_heads": 4,
                "ff_dim": 64,
                "num_layers": 1,
            }
        )
        return spec

    # MLP
    if model_key == "mlp":
        spec.update(
            {
                "input_shape": [32],  # features
                "num_classes": 4,
                "hidden_units": [64, 32],
                "dropout_rate": 0.1,
            }
        )
        return spec

    # 폴백
    spec.update(
        {
            "input_shape": [32, 32, 3],
            "num_classes": 3,
        }
    )
    return spec


# ---- 3) 실행 ----
ok, fail = [], []
for m in models:
    try:
        spec = tiny_spec_for(m)
        model = codegen.build_compiled_model(m, spec)
        # 간단 요약 출력(필요 시 주석 처리)
        model.summary(print_fn=lambda s: None)  # 화면 지저분 방지
        ok.append(m)
    except Exception as e:
        traceback.print_exc()
        fail.append((m, str(e)))

print("\n=== RESULT ===")
print("OK :", ok)
print("FAIL:", fail)
print(f"Generated py files are under: {codegen.GENERATED_DIR}")
