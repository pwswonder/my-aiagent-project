#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_HISTORY_REWRITE:-}" != "I_ROTATED_THE_CREDENTIAL" ]]; then
  echo "Refusing: rotate the exposed credential, then set CONFIRM_HISTORY_REWRITE=I_ROTATED_THE_CREDENTIAL."
  exit 2
fi
if ! git filter-repo --version >/dev/null 2>&1; then
  echo "git-filter-repo is required. Install it before continuing."
  exit 2
fi

git filter-repo --force --invert-paths \
  --path backend/.env \
  --path .generated \
  --path backup \
  --path uploaded_docs \
  --path test_result \
  --path readme_pack.zip \
  --path etc/slot_tools_bundle.zip \
  --path test_router.ipynb

echo "History rewritten locally. Inspect it and run gitleaks before force-pushing all refs."
