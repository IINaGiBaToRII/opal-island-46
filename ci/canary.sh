#!/usr/bin/env bash
# Эталон тождества: два коротких матча на фиксированных узлах служебными сетями weights/t-*.dw (случайные веса из зерна,
# по одной на каждую форму сети кругов). Результат при узловом контроле не зависит от числа потоков
# и машины, если сборка считает так же. Печатает строки «имя FINAL-без-времени» и «имя journal sha256 журнала»;
# сверка: ci/canary.sh BIN | diff ci/canary.expected -
set -euo pipefail
BIN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
cd "$(dirname "$0")/.."
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
run() { # имя оценкаA оценкаB партий зерно
  "$BIN" match --inproc --variant russian --eval-a "$2" --eval-b "$3" --nodes-a 20000 --nodes-b 20000 \
    --games "$4" --seed "$5" --concurrency "$NCPU" --state "$T/$1.jsonl" > "$T/$1.log" 2>&1 || { echo "$1 ошибка: $(tail -n 3 "$T/$1.log")"; return; }
  echo "$1 $(grep '^FINAL' "$T/$1.log" | sed -E 's/^FINAL \[[0-9]+s\] //; s/  resumed pairs [0-9]+$//')"
  echo "$1 journal $(python3 ci/merge.py canon "$T/$1.jsonl")"
}
run c1 weights/t-nn64.dw simple 16 1
run c2 weights/t-pnn.dw weights/t-nn64.dw 8 2
