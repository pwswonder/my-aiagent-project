# tools/test_basecode_step1.py
# -*- coding: utf-8 -*-
"""
Step 1 단독 드라이런:
- services/pipeline_basecode.py의 generate_high_quality_basecode()를
  프로젝트 외부(그래프/스트림릿 미연동)에서 직접 호출해 base code 생성 경로를 검증한다.
- 실패 시에도 오류 원인을 표준출력으로 최대한 설명.

전제:
- OpenAI GPT-4.1 (API version: 2024-10-21) 사용 가정(코드 상 직접 호출 X).
- 프로젝트에 다음 모듈들이 존재한다고 가정:
  services/spec_verifier.py, services/spec_hardener.py, services/codegen.py,
  services/codegen_autoblocks.py, services/basecode_service.py,
  services/code_quality_analyzer.py, services/quality_reflection.py,
  services/langgraph_reflection.py, services/llm_codegen_assist.py
"""

import os
import json
import traceback

# 1) PYTHONPATH 보정: "프로젝트 루트"에서 실행한다고 가정
#    필요 시: export PYTHONPATH=$(pwd)
try:
    import sys

    sys.path.append(os.getcwd())
except Exception:
    pass

# 2) 환경변수: LLM/Reflection on (없어도 기본값은 True)
os.environ.setdefault("USE_LLM_ASSIST", "true")
os.environ.setdefault("USE_QUALITY_REFLECTION", "true")

# 3) 오케스트레이터 import
from services.pipeline_basecode import generate_high_quality_basecode


def _minimal_spec() -> dict:
    """
    ModelSpec 스키마를 만족하도록 최소 스펙 구성.
    - verify_and_normalize는 ModelSpec을 기대하므로, 파이프라인에서 ModelSpec(**dict)를 만들 수 있게 키명을 정확히 맞춘다.
    - evidence는 EvidenceSnippet 리스트 타입( {text, section, page} ) 이어야 한다.
    - proposed_model_family는 Enum이지만 문자열로 주면 Pydantic이 변환한다.
    """
    return {
        # (A) 메타
        "title": "Dummy Paper for Smoke Test",
        "task_type": "classification",  # 예: classification/regression/...
        "data_modality": "text",  # 예: text/image/time_series/...
        # (B) 제안 모델 계열
        "proposed_model_family": "Transformer",  # 정확한 필드명! (family 아님)
        # (C) 핵심 블록 시그니처 (검증기에서 필수 블록 확인에 사용)
        "key_blocks": ["MultiHeadAttention", "LayerNorm", "FeedForward"],
        # (D) 플래그 (검증기의 첫 체크를 통과시키기 위해 True)
        "is_proposed_clearly_identified": True,
        # (E) 차원/하이퍼파라 (DimensionConfig로 파싱됨)
        "dims": {
            "input_dim": 32,
            "hidden_dim": 64,
            "ffn_dim": 128,
            "num_heads": 4,
            "max_len": 128,
            "vocab_size": 3000,
            # 필요 시 time_series면 seq_len/pred_len 등
        },
        # (F) 근거 스니펫 (EvidenceSnippet 리스트)
        "evidence": [
            {
                "text": "Transformer encoder with multi-head attention and layer normalization.",
                "section": "Model",
                "page": 3,
            }
        ],
        # (G) 슬롯 힌트(선택): 템플릿 내 CUSTOM_BLOCK 있으면 LLM 호출을 줄일 수 있음
        "custom_blocks": {
            "optimizer": "return keras.optimizers.Adam(learning_rate=1e-3)"
        },
        # (H) 과거 호환(혹시 라우터가 model_family 키도 참고한다면 함께 제공)
        "model_family": "transformer",
    }


def main():
    try:
        spec = _minimal_spec()
        # doc_id를 주면 base code 영속화 시 디렉토리 구분에 사용됩니다.
        out = generate_high_quality_basecode(
            spec, doc_id="step1_smoke", max_reflect_rounds=2
        )

        # --- 확인 블록 시작 ---
        # 1) 생성된 소스 읽기
        py_src = out.get("py_src", "")  # ✅ 항상 먼저 정의
        print("\n[Step1: PY SRC LEN]", len(py_src))

        # 2) 슬롯 사용/잔여 검사(이미 있으시면 유지)
        print("\n[Step1: SLOTS_USED]")
        print(out.get("slots_used"))

        import re

        slot_any_re = re.compile(
            r"(?m)^(?:\s*)(?:#\s*)?(?:\{\%\s*raw\s*\%\}\s*)?"
            r"(?:\{\{CUSTOM_BLOCK:\s*([A-Za-z0-9_\-]+)\s*\}\}|\{CUSTOM_BLOCK:\s*([A-Za-z0-9_\-]+)\s*\})"
            r"(?:\s*\{\%\s*endraw\s*\%\})?\s*$"
        )
        leftover = []
        for m in slot_any_re.finditer(py_src):
            leftover.append(m.group(1) or m.group(2) or "")
        print("\n[Step1: LEFTOVER_SLOTS]")
        print(sorted(set(filter(None, leftover))))

        # 3) compile_override 변수 정의가 들어갔는지 간단 체크
        print(
            "\n[Quick check] has opt/loss_fn/metrics? ->",
            ("opt =" in py_src) and ("loss_fn =" in py_src) and ("metrics =" in py_src),
        )

        # 4) (선택) 템플릿이 실제로 compile을 호출하는지 확인
        has_compile_call = "model.compile(" in py_src
        print("[Quick check] has model.compile call? ->", has_compile_call)

        # 5) (강력 권장) 파이썬 구문 검사
        try:
            compile(py_src, "<gen>", "exec")
            print("[Quick check] Python compile() -> OK")
        except Exception as ce:
            print("[Quick check] Python compile() -> FAIL:", type(ce).__name__, str(ce))
        # --- 확인 블록 끝 ---

        print("\n[Step1: RESULT KEYS]")
        print(list(out.keys()))

        print("\n[Step1: TEMPLATE KEY]")
        print(out.get("template_key"))

        print("\n[Step1: PY PATH]")
        print(out.get("py_path"))

        print("\n[Step1: PY SRC HEAD]")
        print(py_src.splitlines()[:25])

        print("\n[Step1: QUALITY SUMMARY]")
        q = out.get("quality", {})
        print(json.dumps(q, ensure_ascii=False, indent=2)[:1000])

        print("\n[Step1: REFLECTION SUMMARY]")
        r = out.get("reflection", {})
        # src는 길 수 있으니 길이만 출력
        print({k: (len(v) if isinstance(v, str) else v) for k, v in r.items()})

        # 간단한 성공 조건: 최소한 소스코드에 model/build 흔적이 있어야 함
        assert (
            "def build_model" in py_src or "keras.Model(" in py_src
        ), "build_model 함수/모델 정의 미검출"
        print("\n✅ Step 1 PASS: 단독 드라이런 성공")

    except Exception as e:
        print("\n❌ Step 1 FAIL")
        print(type(e).__name__, str(e))
        traceback.print_exc()


if __name__ == "__main__":
    main()
