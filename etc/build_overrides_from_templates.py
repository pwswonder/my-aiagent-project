# build_overrides_from_templates.py
# -*- coding: utf-8 -*-
"""
전 패밀리(Transformer/RNN/CNN/UNet/GAN/VAE/Autoencoder/Generic) 대상:
- templates_dir 아래 *.j2를 스캔해 CUSTOM_BLOCK/AUTOBLOCK 추출
- slot_registry.get_slot_registry()의 내장 슬롯과 비교
- 누락 슬롯을 family별로 slot_registry_overrides.json에 자동 추가(병합)
- 자주 쓰는 슬롯명은 family-aware 예시 코드 스니펫을 자동 채움
"""

from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List, Any, Set
from slot_registry import get_slot_registry, infer_family_from_source

# 1) 블록 추출 정규식(대/소문자 무시, 콜론/괄호/따옴표 모두 허용)
BLOCK_REGEXES = [
    re.compile(
        r"\{\{\s*CUSTOM_BLOCK\s*[:(]\s*['\"]?(?P<name>[A-Za-z0-9_]+)['\"]?\s*\)?\s*\}\}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\{\{\s*AUTOBLOCK\s*[:(]\s*['\"]?(?P<name>[A-Za-z0-9_]+)['\"]?\s*\)?\s*\}\}",
        re.IGNORECASE,
    ),
    re.compile(
        r"CUSTOM_BLOCK\s*[:(]\s*['\"]?(?P<name>[A-Za-z0-9_]+)['\"]?\s*\)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"AUTOBLOCK\s*[:(]\s*['\"]?(?P<name>[A-Za-z0-9_]+)['\"]?\s*\)?", re.IGNORECASE
    ),
]


def extract_blocks(text: str) -> List[str]:
    """템플릿에서 커스텀 슬롯명 추출(순서보존, 중복제거)."""
    hits: List[str] = []
    for rx in BLOCK_REGEXES:
        for m in rx.finditer(text):
            hits.append(m.group("name"))
    seen: Set[str] = set()
    uniq: List[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq


# 2) 자주 쓰는 슬롯명 → family-aware 예시 코드 사전
def snippet_library() -> Dict[str, Dict[str, str]]:
    """
    key=slot_name, value=family->example_code 매핑
    - 일치하는 family가 없으면 Generic을 사용
    - 예시는 최소 형태(필요시 팀에 맞게 확장)
    """
    lib: Dict[str, Dict[str, str]] = {
        # CNN 계열
        "imports_extra": {"CNN": "import math\n"},
        "se_squeeze_excitation": {
            "CNN": (
                "se = layers.GlobalAveragePooling2D()(x)\n"
                "se = layers.Dense(int(x.shape[-1]*{se_ratio}), activation='relu')(se)\n"
                "se = layers.Dense(int(x.shape[-1]), activation='sigmoid')(se)\n"
                "se = layers.Reshape((1,1,int(x.shape[-1])))(se)\n"
                "x = layers.Multiply()([x, se])\n"
            )
        },
        "stochastic_depth": {
            "CNN": (
                "if {drop_rate} > 0:\n"
                "    import tensorflow as tf\n"
                "    keep = 1.0 - {drop_rate}\n"
                "    shape = (tf.shape(x)[0],) + (1,)* (len(x.shape)-1)\n"
                "    rnd = tf.random.uniform(shape, dtype=x.dtype)\n"
                "    mask = tf.floor(rnd + keep) / keep\n"
                "    x = x * mask\n"
                "x = layers.Add()([x, shortcut])\n"
            )
        },
        "stages": {
            "CNN": (
                "for si, (filters, blocks) in enumerate(zip({filters_list}, {blocks_per_stage})):\n"
                "    for bi in range(blocks):\n"
                "        sc = x\n"
                "        x = layers.Conv2D(filters, 3, padding='same', use_bias=False, name=f's{si}_b{bi}_conv1')(x)\n"
                "        x = layers.BatchNormalization(name=f's{si}_b{bi}_bn1')(x)\n"
                "        x = layers.ReLU(name=f's{si}_b{bi}_relu1')(x)\n"
                "        x = layers.Conv2D(filters, 3, padding='same', use_bias=False, name=f's{si}_b{bi}_conv2')(x)\n"
                "        x = layers.BatchNormalization(name=f's{si}_b{bi}_bn2')(x)\n"
                "        x = layers.Add(name=f's{si}_b{bi}_add')([x, sc])\n"
                "        x = layers.ReLU(name=f's{si}_b{bi}_out')(x)\n"
            )
        },
        "inception_mixed": {
            "CNN": (
                "b1 = layers.Conv2D({filters}, 1, padding='same', activation='relu')(x)\n"
                "b2 = layers.Conv2D({filters}, 1, padding='same', activation='relu')(x)\n"
                "b2 = layers.Conv2D({filters}, 3, padding='same', activation='relu')(b2)\n"
                "b3 = layers.Conv2D({filters}, 1, padding='same', activation='relu')(x)\n"
                "b3 = layers.Conv2D({filters}, 5, padding='same', activation='relu')(b3)\n"
                "b4 = layers.MaxPool2D(3, strides=1, padding='same')(x)\n"
                "b4 = layers.Conv2D({filters}, 1, padding='same', activation='relu')(b4)\n"
                "x = layers.Concatenate()([b1,b2,b3,b4])\n"
            )
        },
        # Transformer 계열
        "positional_encoding": {
            "Transformer": (
                "import tensorflow as tf\n"
                "def sinusoidal_positional_encoding(length, d_model):\n"
                "    pos = tf.range(length)[:, None]\n"
                "    i = tf.range(d_model)[None, :]\n"
                "    angle_rates = 1 / tf.pow(10000.0, (2*(i//2))/tf.cast(d_model, tf.float32))\n"
                "    angles = tf.cast(pos, tf.float32) * angle_rates\n"
                "    sines = tf.sin(angles[:, 0::2]); cosines = tf.cos(angles[:, 1::2])\n"
                "    pe = tf.concat([sines, cosines], axis=-1)\n"
                "    return pe\n"
            )
        },
        "causal_mask": {
            "Transformer": (
                "import tensorflow as tf\n"
                "def make_causal_mask(seq_len):\n"
                "    i = tf.range(seq_len)[:, None]\n"
                "    j = tf.range(seq_len)[None, :]\n"
                "    return tf.cast(i >= j, tf.float32)\n"
            )
        },
        "label_smoothing": {
            "Transformer": (
                "from tensorflow.keras.losses import CategoricalCrossentropy\n"
                "loss_fn = CategoricalCrossentropy(label_smoothing={eps})\n"
            )
        },
        # RNN 계열
        "attention_bahdanau": {
            "RNN": (
                "def bahdanau_attention(query, values, units):\n"
                "    score = layers.Dense(units)(layers.Activation('tanh')(layers.Add()([query, values])))\n"
                "    weights = tf.nn.softmax(score, axis=1)\n"
                "    context = tf.reduce_sum(weights * values, axis=1, keepdims=True)\n"
                "    return context, weights\n"
            )
        },
        "attention_luong": {
            "RNN": (
                "def luong_dot_attention(query, values):\n"
                "    scores = tf.matmul(query, values, transpose_b=True)\n"
                "    weights = tf.nn.softmax(scores, axis=-1)\n"
                "    context = tf.matmul(weights, values)\n"
                "    return context, weights\n"
            )
        },
        # UNet/VAE/GAN/Autoencoder 공통 후보
        "encoder_block": {
            "VAE": "z_mean = layers.Dense({z_dim})(x)\nz_logvar = layers.Dense({z_dim})(x)\n",
            "Autoencoder": "# build encoder stack here\n",
            "UNet": "# downsampling conv block here\n",
            "GAN": "# generator encoder stem here\n",
            "Generic": "# encoder block\n",
        },
        "decoder_block": {
            "VAE": "x = layers.Dense({proj_dim}, activation='relu')(z)\n",
            "Autoencoder": "# build decoder stack here\n",
            "UNet": "# upsampling conv block here\n",
            "GAN": "# generator upsampling here\n",
            "Generic": "# decoder block\n",
        },
        "sampling_latent": {
            "VAE": (
                "import tensorflow as tf\n"
                "def sampling(args):\n"
                "    z_mean, z_logvar = args\n"
                "    eps = tf.random.normal(shape=tf.shape(z_mean))\n"
                "    return z_mean + tf.exp(0.5*z_logvar) * eps\n"
                "z = layers.Lambda(sampling)([z_mean, z_logvar])\n"
            )
        },
        "generator_block": {
            "GAN": "x = layers.Conv2DTranspose({filters}, 4, strides=2, padding='same', activation='relu')(x)\n",
        },
        "discriminator_block": {
            "GAN": "x = layers.Conv2D({filters}, 4, strides=2, padding='same', activation='leaky_relu')(x)\n",
        },
        "bottleneck": {
            "UNet": "x = layers.Conv2D({filters}, 3, padding='same', activation='relu')(x)\n",
            "Autoencoder": "x = layers.Dense({latent_dim}, activation='relu')(x)\n",
            "Generic": "# bottleneck\n",
        },
        "skip_connect": {
            "UNet": "x = layers.Concatenate()([x, skip])\n",
        },
    }
    return lib


def main(
    templates_dir: str = "templates_scaffolded/services",
    out_path: str = "slot_registry_overrides.json",
) -> None:
    tdir = Path(templates_dir)
    assert tdir.exists(), f"templates_dir not found: {tdir}"
    reg = get_slot_registry()  # 내장 + 기존 오버라이드 병합본

    # family -> known slot names 세트
    known: Dict[str, Set[str]] = {
        fam: {s["name"] for s in conf.get("slots", [])} for fam, conf in reg.items()
    }

    # 새로 발견된(=오버라이드로 추가할) 슬롯 누적
    new_overrides: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    lib = snippet_library()

    for path in sorted(tdir.glob("**/*.j2")):
        src = path.read_text(encoding="utf-8")
        family = infer_family_from_source(src)
        blocks = extract_blocks(src)

        fam_known = known.get(family, set())
        for name in blocks:
            if name in fam_known:
                continue  # 이미 등록됨
            # 예시 코드 우선순위: (slot, family) → (slot, Generic) → 기본 TODO
            code = (
                lib.get(name, {}).get(family)
                or lib.get(name, {}).get("Generic")
                or f"# TODO: provide code for slot {name}\n"
            )
            entry = {
                "name": name,
                "contract": {"inputs": [], "provides": [], "forbidden": []},
                "example_code": code,
            }
            new_overrides.setdefault(family, {"slots": []})["slots"].append(entry)
            fam_known.add(name)  # 중복 방지

    # 기존 overrides가 있으면 병합
    out_fp = Path(out_path)
    merged: Dict[str, Any] = {}
    if out_fp.exists():
        try:
            merged = json.loads(out_fp.read_text(encoding="utf-8"))
        except Exception:
            merged = {}

    # family별로 병합(슬롯명 기준 de-dup)
    for fam, conf in new_overrides.items():
        if fam not in merged:
            merged[fam] = {"slots": conf["slots"]}
        else:
            by_name = {s["name"]: s for s in merged[fam].get("slots", [])}
            for s in conf["slots"]:
                by_name[s["name"]] = s
            merged[fam]["slots"] = list(by_name.values())

    out_fp.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] wrote overrides → {out_fp.resolve()}")


if __name__ == "__main__":
    main()
