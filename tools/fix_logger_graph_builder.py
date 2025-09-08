# tools/fix_logger_graph_builder.py
# ----------------------------------------------------------------------
# graph_builder.py의 logger 호출을 안전한 log로 자동 교체합니다.
# ----------------------------------------------------------------------
import pathlib
import re

p = pathlib.Path("services/graph_builder.py")
src = p.read_text(encoding="utf-8")

# 1) logger. → log.
src = src.replace("logger.", "log.")

# 2) import/초기화 없으면 주입
if "from services.logging_utils import get_logger" not in src:
    # import logging 뒤에 붙여 넣기 (없으면 파일 맨 앞에 추가)
    if "import logging" in src:
        src = re.sub(
            r"(?m)^(import logging.*)$",
            r"\1\nfrom services.logging_utils import get_logger\nlog = get_logger('services.graph_builder')",
            src,
            count=1,
        )
    else:
        src = (
            "from services.logging_utils import get_logger\nlog = get_logger('services.graph_builder')\n"
            + src
        )

# 3) 방어적 assert 추가(중복 삽입 방지)
if "log = get_logger(" in src and "logger가 올바른 logger가 아닙니다." not in src:
    src = src.replace(
        "log = get_logger('services.graph_builder')",
        "log = get_logger('services.graph_builder')\nassert hasattr(log, 'info') and callable(log.info), \"log가 올바른 logger가 아닙니다.\"",
        1,
    )

p.write_text(src, encoding="utf-8")
print("✔ graph_builder.py patched")
