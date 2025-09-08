# -*- coding: utf-8 -*-
"""
See previous cell for full docstring. Kept minimal to avoid duplication.
"""
from __future__ import annotations
import os, sys, json, difflib
from pathlib import Path
from typing import Dict, Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def _read_json(p: Optional[str]) -> Dict[str, Any]:
    if not p: return {}
    J = Path(p)
    if not J.exists(): return {}
    try:
        return json.loads(J.read_text(encoding='utf-8'))
    except Exception:
        return {}

def _default_spec(template_key: str) -> Dict[str, Any]:
    return {
        'proposed_model_family': template_key,
        'task_type': 'Image_Classification',
        'dims': {'image_size':224,'in_channels':3,'num_classes':10,'hidden_dim':256,'num_heads':8,'ffn_dim':1024,'num_layers':4,'dropout':0.1,'vocab_size':32000,'max_len':256,'latent_dim':64},
        'optimizer_name':'adam','loss':'sparse_categorical_crossentropy','metrics':['accuracy'],'seed':42,'dropout':0.1,
    }

def _preflight(src: str):
    try:
        compile(src, '<gen.py>', 'exec'); return True, ''
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}: {e.text}"
    except Exception as e:
        return True, f'Non-fatal: {type(e).__name__}: {e}'

def _write(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(s, encoding='utf-8')

def _diff(a: str, b: str, an: str, bn: str) -> str:
    return ''.join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), fromfile=an, tofile=bn))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--template-key', required=True)
    ap.add_argument('--spec-json', default=None)
    ap.add_argument('--outdir', default='./e2e_artifacts')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    spec = _default_spec(args.template_key); spec.update(_read_json(args.spec_json))

    from services.basecode_service import generate_base_code
    py_path, py_src, msum = None, '', {}
    try:
        py_path, py_src, msum = generate_base_code(args.template_key, spec)
    except Exception as e:
        _write(outdir/'00_codegen_error.log', f'{type(e).__name__}: {e}')
    _write(outdir/'01_codegen_source.py', py_src)
    ok, log = _preflight(py_src); _write(outdir/'01_codegen_preflight.log', log or 'OK')

    # LangGraph reflection (optional)
    if str(os.getenv('USE_LANGGRAPH_REFLECTION','false')).lower() in ('1','true','yes'):
        try:
            from services.langgraph_reflection import run_langgraph_reflection
            rounds = int(os.getenv('LG_REFLECTION_ROUNDS','2'))
            lg = run_langgraph_reflection(py_src, spec, max_rounds=rounds)
            py_src_lg = lg.get('src', py_src)
            _write(outdir/'02_langgraph_reflected.py', py_src_lg)
            _write(outdir/'02_langgraph_diff.patch', _diff(py_src, py_src_lg, 'codegen.py','langgraph.py'))
            ok2, log2 = _preflight(py_src_lg); _write(outdir/'02_langgraph_preflight.log', log2 or 'OK')
            py_src = py_src_lg
        except Exception as e:
            _write(outdir/'02_langgraph_error.log', f'{type(e).__name__}: {e}')

    # Classic reflection_loop (optional)
    if str(os.getenv('USE_REFLECTION','false')).lower() in ('1','true','yes'):
        try:
            from services.reflection_loop import run_reflection_rounds
            rounds = int(os.getenv('REFLECTION_ROUNDS','2'))
            r = run_reflection_rounds(args.template_key, spec, py_src, warnings=[], max_rounds=rounds, enable=True)
            py_src_rl = r.get('src', py_src)
            _write(outdir/'03_reflection_loop.py', py_src_rl)
            _write(outdir/'03_reflection_loop_diff.patch', _diff(py_src, py_src_rl, 'prev.py','classic_reflection.py'))
            ok3, log3 = _preflight(py_src_rl); _write(outdir/'03_reflection_loop_preflight.log', log3 or 'OK')
            py_src = py_src_rl
        except Exception as e:
            _write(outdir/'03_reflection_loop_error.log', f'{type(e).__name__}: {e}')

    # Quality reflection (optional)
    if str(os.getenv('USE_QUALITY_REFLECTION','false')).lower() in ('1','true','yes'):
        try:
            from services.quality_reflection import run_quality_reflection
            rounds = int(os.getenv('QUALITY_REFLECTION_ROUNDS','1'))
            qr = run_quality_reflection(args.template_key, spec, py_src, enable=True, max_rounds=rounds)
            py_src_qr = qr.get('src', py_src)
            _write(outdir/'04_quality_reflection.py', py_src_qr)
            _write(outdir/'04_quality_reflection_diff.patch', _diff(py_src, py_src_qr, 'prev.py','quality_reflection.py'))
            ok4, log4 = _preflight(py_src_qr); _write(outdir/'04_quality_reflection_preflight.log', log4 or 'OK')
            _write(outdir/'04_quality_issues.json', json.dumps(qr.get('issues', []), ensure_ascii=False, indent=2))
            py_src = py_src_qr
        except Exception as e:
            _write(outdir/'04_quality_reflection_error.log', f'{type(e).__name__}: {e}')

    _write(outdir/'ZZ_final_source.py', py_src)
    okf, logf = _preflight(py_src); _write(outdir/'ZZ_final_preflight.log', logf or 'OK')
    print('[DONE] artifacts under:', outdir)

if __name__ == '__main__':
    main()
