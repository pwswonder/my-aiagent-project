# -*- coding: utf-8 -*-
"""
tools/strip_build_model_tail.py
- 02_langgraph_reflected.py처럼 build_model() 내부에
  'return ' 이후가 중복/오염되어 문법오류가 나는 경우,
  첫 'return ' 라인까지를 남기고 나머지(동일 들여쓰기 레벨의 연속 블록)를 잘라냅니다.
- *주의*: 산출물 자체를 수정하므로 'changed_only_slots' 체크에는 불리할 수 있음.
"""

import sys, re, io, os


def strip_after_first_return_in_build_model(text: str) -> str:
    # build_model() 함수 영역 식별: def build_model(): ~ 다음 비인덴트(혹은 EOF)
    m = re.search(r"(?m)^def\s+build_model\s*\(\s*\)\s*:\s*$", text)
    if not m:
        return text
    start = m.end()
    # 함수 본문 시작(다음 줄부터). 들여쓰기 수준 파악
    # 다음 비어있지 않은 줄의 선행 공백 수를 함수 인덴트로 간주
    rest = text[start:]
    lines = rest.splitlines(True)
    if not lines:
        return text
    # 함수 인덴트 계산
    func_indent = None
    for ln in lines:
        if ln.strip():
            func_indent = len(ln) - len(ln.lstrip(" "))
            break
    if func_indent is None:
        return text

    # 첫 return 라인을 찾고, 그 라인 이후로 같은/더 깊은 인덴트의 연속 블록을 모두 제거
    out = []
    seen_return = False
    for ln in lines:
        cur_indent = len(ln) - len(ln.lstrip(" "))
        if not seen_return:
            out.append(ln)
            if ln.lstrip().startswith("return "):
                seen_return = True
            continue
        # seen_return 이후: 함수 영역에 해당(동일/더 깊은 인덴트)하면 skip
        if cur_indent >= func_indent or ln.strip() == "":
            # skip(오염 블록/빈 줄)
            continue
        else:
            # 함수 영역을 벗어남 → 여기서부터는 그대로 이어붙임
            out.append(ln)
            out.extend(lines[lines.index(ln) + 1 :])
            break

    return text[:start] + "".join(out)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python tools/strip_build_model_tail.py <path/to/02_langgraph_reflected.py>"
        )
        sys.exit(1)
    path = sys.argv[1]
    with io.open(path, "r", encoding="utf-8") as f:
        src = f.read()
    fixed = strip_after_first_return_in_build_model(
        src.replace("\r\n", "\n").replace("\t", "    ")
    )
    if fixed != src:
        with io.open(path, "w", encoding="utf-8") as f:
            f.write(fixed)
        print(f"[OK] Tail stripped: {path}")
    else:
        print("[INFO] No change:", path)


if __name__ == "__main__":
    main()
