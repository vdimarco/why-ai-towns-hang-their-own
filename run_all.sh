#!/usr/bin/env bash
# Reproduce every number and figure in the paper from the public data. About 5 minutes, no API keys.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; DATA="${1:-$HERE/data}"; OUT="$HERE/results"; mkdir -p "$OUT"; cd "$OUT"
[ -n "${SKIP_DOWNLOAD:-}" ] || bash "$HERE/analysis/00_download.sh" "$DATA"
python3 "$HERE/analysis/01_probes.py" "$DATA"
python3 "$HERE/analysis/02_independence.py" "$DATA" | tee independence.txt
python3 "$HERE/analysis/03_executions_lineage.py" "$DATA" | tee executions_lineage.txt
python3 "$HERE/analysis/04_stats.py" "$DATA" | tee stats.txt
python3 "$HERE/analysis/05_aggregation.py" "$DATA" | tee aggregation.txt
python3 "$HERE/analysis/06_figures.py" "$OUT"
python3 "$HERE/analysis/07_robustness.py" "$DATA" | tee robustness.txt
