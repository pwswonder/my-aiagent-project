# tools/test_basecode_transformer_mt.py
# -*- coding: utf-8 -*-
"""
목적
- transformer_mt 템플릿 라우팅 + 슬롯 주입(LLM→autoblock→reflection→최종봉합) 경로 단독 검증
- 파이프라인(services/pipeline_basecode.py)의 generate_high_quality_basecode()만 호출

전제
- OpenAI GPT-4.1 (API 2024-10-21) 사용 가정(본 스크립트는 직접 호출 X)
- 기존 모듈: services/spec_schema.py, services/spec_verifier.py, services/spec_hardener.py,
             services/codegen.py, services/codegen_autoblocks.py, services/basecode_service.py,
             services/code_quality_analyzer.py, services/quality_reflection.py,
             services/langgraph_reflection.py, services/llm_codegen_assist.py
"""

import os
import sys
import json
import traceback
import re

# 0) 실행 컨텍스트: 프로젝트 루트에서
sys.path.append(os.getcwd())

# 1) 환경 변수 (원하면 DEBUG on)
os.environ.setdefault("USE_LLM_ASSIST", "true")
os.environ.setdefault("USE_QUALITY_REFLECTION", "true")
os.environ.setdefault("DEBUG_BASECODE", "true")  # 내부 로그 보고 싶을 때 true

from services.pipeline_basecode import generate_high_quality_basecode


def _spec_transformer_mt() -> dict:
    """
    ModelSpec 스키마를 만족하도록 'MT(번역)' 특화 최소 스펙 구성.
    - 라우팅을 transformer_mt로 유도:
      * task_type='translation' (혹은 'seq2seq')
      * proposed_model_family='TransformerMT'
      * model_family='transformer_mt'
      * key_blocks에 Encoder-Decoder/CrossAttention 힌트
    - dims: MT 템플릿에서 자주 쓰는 키를 충분히 제공
    - custom_blocks: 템플릿 슬롯과 '정확히 같은 이름'으로 제공
      (encoder_layers/decoder_layers는 안전하게 'pass'로 시작)
    """
    return {
        # (A) 메타
        "title": "Dummy MT Paper for Smoke Test",
        "task_type": "machine_translation",  # 또는 "seq2seq" (라우터 규칙에 맞게 조정)
        "data_modality": "text",
        # (B) 라우팅 힌트 (아주 중요)
        "proposed_model_family": "TransformerMT",
        "model_family": "transformer_mt",  # 일부 라우터는 이 값을 직접 봄
        "subtype": "machine_translation",  # 선택: 라우터에 추가 힌트
        # (C) 핵심 블록 시그널
        "key_blocks": [
            "EncoderDecoder",
            "CrossAttention",
            "MultiHeadAttention",
            "LayerNorm",
        ],
        # (D) 검증기 초반 체크 통과 플래그
        "is_proposed_clearly_identified": True,
        # (E) 차원/하이퍼 파라미터 (하드너/템플릿에서 평탄화 사용)
        "dims": {
            "d_model": 64,
            "ffn_dim": 128,
            "num_heads": 4,
            "num_layers": 2,  # 인코더/디코더 공통 레이어 수 (템플릿이 따로쓰면 encoder_layers/decoder_layers 참조)
            "dropout_rate": 0.1,
            "src_vocab_size": 5000,
            "tgt_vocab_size": 6000,
            "max_len_src": 128,
            "max_len_tgt": 128,
            # 일부 템플릿이 루트키를 직접 읽을 수 있어 안전하게 중복 제공(파이프라인이 setdefault로 평탄화함)
            "input_dim": 64,
        },
        # (F) 근거 스니펫 (EvidenceSnippet 리스트)
        "evidence": [
            {
                "text": "This paper proposes an encoder-decoder Transformer for machine translation with cross-attention.",
                "section": "Model",
                "page": 4,
            }
        ],
        # (G) 슬롯 페이로드 (템플릿 슬롯명과 '정확히' 일치)
        "custom_blocks": {
            # 임포트/시드(quality 경고 완화)
            "imports_extra": """\
import random
import numpy as np
import tensorflow as tf
seed = int(globals().get('seed', 42))
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)""",
            # compile 변수만 정의(템플릿에서 실제 model.compile을 호출하는 형태를 권장)
            "compile_override": """\
# apply spec-aligned compile options for translation
opt = keras.optimizers.get('adam')
loss_fn = 'sparse_categorical_crossentropy'
metrics = ['accuracy']""",
            # 안전 시동: 템플릿 내부 컨텍스트(x/y 변수명 등)를 모르면 'pass'부터 시작
            "encoder_layers": "pass",
            "decoder_layers": "pass",
        },
    }


def main():
    try:
        spec = _spec_transformer_mt()
        out = generate_high_quality_basecode(
            spec, doc_id="step1_mt_smoke", max_reflect_rounds=2
        )

        # --- 결과 검증/출력 ---
        print("\n[StepMT: RESULT KEYS]")
        print(list(out.keys()))

        tkey = out.get("template_key")
        print("\n[StepMT: TEMPLATE KEY]", tkey)
        # 라우팅 강제 확인 (다르면 스펙 튜닝)
        assert (
            str(tkey).strip().lower() == "transformer_mt"
        ), f"라우팅 실패: 기대='transformer_mt' 실제='{tkey}' → spec의 task_type/proposed_model_family/subtype/evidence를 강화하세요."

        print("\n[StepMT: PY PATH]")
        print(out.get("py_path"))

        py_src = out.get("py_src", "")
        print("\n[StepMT: PY SRC LEN]", len(py_src))

        # 슬롯 사용 현황
        print("\n[StepMT: SLOTS_USED]")
        print(out.get("slots_used"))

        # 남은 슬롯(any-style) 검사
        slot_any_re = re.compile(
            r"(?m)^(?:\s*)(?:#\s*)?(?:\{\%\s*raw\s*\%\}\s*)?"
            r"(?:\{\{CUSTOM_BLOCK:\s*([A-Za-z0-9_\-]+)\s*\}\}|\{CUSTOM_BLOCK:\s*([A-Za-z0-9_\-]+)\s*\})"
            r"(?:\s*\{\%\s*endraw\s*\%\})?\s*$"
        )
        leftover = []
        for m in slot_any_re.finditer(py_src):
            leftover.append(m.group(1) or m.group(2) or "")
        print("\n[StepMT: LEFTOVER_SLOTS]")
        print(sorted(set(filter(None, leftover))))

        # 간단 품질 확인: compile 변수 정의
        # print(
        #     "\n[Quick check] has opt/loss_fn/metrics? ->",
        #     ("opt =" in py_src) and ("loss_fn =" in py_src) and ("metrics =" in py_src),
        # )

        has_opt = ("opt =" in py_src) or ("optimizer =" in py_src)

        has_loss = "loss_fn =" in py_src
        has_mets = "metrics =" in py_src
        print(
            "[Quick check] has opt/loss_fn/metrics? ->",
            has_opt and has_loss and has_mets,
        )

        # 파이썬 구문 체크
        try:
            compile(py_src, "<gen>", "exec")
            print("[Quick check] Python compile() -> OK")
        except Exception as ce:
            print("[Quick check] Python compile() -> FAIL:", type(ce).__name__, str(ce))

        # 품질/리플렉션 요약
        print("\n[StepMT: QUALITY SUMMARY]")
        q = out.get("quality", {})
        print(json.dumps(q, ensure_ascii=False, indent=2)[:1000])

        print("\n[StepMT: REFLECTION SUMMARY]")
        r = out.get("reflection", {})
        # 길어질 수 있으니 주요 키/길이만
        small_r = {
            k: (len(v) if isinstance(v, str) else v)
            for k, v in r.items()
            if k in ("src", "score", "issues", "payloads")
        }
        print(small_r)

        print(
            "[StepMT: QUALITY_AFTER_COMPILE_ALIGN]",
            out.get("quality_after_compile_align", {}),
        )

        print("\n✅ Step MT PASS: 단독 드라이런 성공")

    except Exception as e:
        print("\n❌ Step MT FAIL")
        print(type(e).__name__, str(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()
