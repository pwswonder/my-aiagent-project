# tools/patch_routing_transformer_mt.py
# -*- coding: utf-8 -*-
"""
라우팅 테이블에 'transformer_mt' 규칙을 안전하게 삽입합니다.
- 이미 존재하면 중복 삽입하지 않습니다.
- 'transformer' 규칙 바로 앞에 삽입하여 우선순위 확보.
"""

import io, sys, os, re, textwrap

ROUTING_PATH = os.path.join("services", "routing.py")

MT_RULE = textwrap.dedent(
    r"""
    {
        "template": "transformer_mt",
        "weight": 120,
        "family_any": ["transformer_mt", "transformer", "TransformerMT", "transformermt"],
        "task_any": [
            "machine_translation",
            "translation",
            "sequence_to_sequence",
            "seq2seq"
        ],
        "subtype_any": ["machine_translation", "translation", "encoderdecoder", "seq2seq"],
        "keywords_any": [
            "machine translation", "mt", "encoder-decoder", "encoder decoder",
            "cross-attention", "cross attention", "autoregressive", "teacher forcing",
            "beam search", "bleu", "sacrebleu", "source to target", "src→tgt", "src->tgt"
        ],
    },
"""
).strip("\n")


def main():
    if not os.path.exists(ROUTING_PATH):
        print(f"❌ {ROUTING_PATH} not found. Run from project root.")
        sys.exit(1)

    src = open(ROUTING_PATH, "r", encoding="utf-8").read()

    # 이미 규칙이 있으면 종료
    if '"template": "transformer_mt"' in src:
        print("✅ transformer_mt rule already exists. No changes.")
        return

    # ROUTE_RULES 리스트 내부에서 'transformer' 규칙의 시작 위치 찾기
    m_rules = re.search(r"ROUTE_RULES\s*:\s*List\[.*?\]\s*=\s*\[", src, flags=re.S)
    if not m_rules:
        print("❌ ROUTE_RULES list not found.")
        sys.exit(2)

    # transformer 규칙 시작 블록 탐지
    m_tf = re.search(
        r'\{\s*?"template"\s*:\s*"transformer"\s*,', src[m_rules.end() :], flags=re.S
    )
    if not m_tf:
        print("❌ 'transformer' rule not found. Insert at start of ROUTE_RULES.")
        insert_pos = m_rules.end()
    else:
        insert_pos = m_rules.end() + m_tf.start()

    new_src = src[:insert_pos] + "\n" + MT_RULE + "\n" + src[insert_pos:]

    # 백업 저장
    backup_path = ROUTING_PATH + ".bak"
    open(backup_path, "w", encoding="utf-8").write(src)
    open(ROUTING_PATH, "w", encoding="utf-8").write(new_src)

    print("✅ transformer_mt rule inserted before 'transformer' rule.")
    print(f"Backup written to: {backup_path}")


if __name__ == "__main__":
    main()
