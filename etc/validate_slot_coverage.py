# validate_slot_coverage.py
# -*- coding: utf-8 -*-
"""
각 템플릿에서 발견된 슬롯이 레지스트리(+오버라이드)에 모두 존재하는지 점검.
누락이 있으면 리스트업하고 종료코드 1을 반환(옵션).
"""
from __future__ import annotations
import sys, json, re
from pathlib import Path
from typing import Dict, List, Set
from slot_registry import get_slot_registry, infer_family_from_source

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
    hits: List[str] = []
    for rx in BLOCK_REGEXES:
        hits += [m.group("name") for m in rx.finditer(text)]
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def main(
    templates_dir: str = "templates_scaffolded/services",
    report_path: str = "slot_coverage_report.json",
    strict_exit: bool = False,
) -> None:
    tdir = Path(templates_dir)
    reg = get_slot_registry()
    # family -> known slots
    known: Dict[str, Set[str]] = {
        fam: {s["name"] for s in conf.get("slots", [])} for fam, conf in reg.items()
    }

    report: Dict[str, Dict] = {}
    missing_total = 0

    for p in sorted(tdir.glob("**/*.j2")):
        src = p.read_text(encoding="utf-8")
        family = infer_family_from_source(src)
        present = extract_blocks(src)
        fam_known = known.get(family, set())
        missing = [s for s in present if s not in fam_known]
        report[p.name] = {
            "family": family,
            "found": present,
            "missing": missing,
            "ok": len(missing) == 0,
        }
        missing_total += len(missing)

    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] wrote report → {Path(report_path).resolve()}")
    if strict_exit and missing_total > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
