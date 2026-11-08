#!/usr/bin/env bash
# Download every public Clocktower Bench game file (the files the clocktowerbench.com replay viewer loads).
set -euo pipefail
OUT="$1"; mkdir -p "$OUT"
curl -sS -o "$OUT/index.json" https://clocktowerbench.com/data/index.json
python3 -c "
import json, sys
for g in json.load(open(sys.argv[1])):
    s = g['id'].replace('/', '--')
    print(f'url = \"https://clocktowerbench.com/data/games/{s}.json.gz\"'); print(f'output = \"{sys.argv[2]}/{s}.json.gz\"')
" "$OUT/index.json" "$OUT" > "$OUT/urls.txt"
curl -sS --parallel --parallel-max 16 -K "$OUT/urls.txt"
echo "$(ls "$OUT"/*.json.gz | wc -l) games in $OUT"
