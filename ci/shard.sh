#!/usr/bin/env bash
# Один осколок матча (или работы KIND=selfplay|relabel — ci/work.sh). Окружение: BIN SHARD SEED GAMES(0 = по времени) MINUTES NODES_A NODES_B
# EVAL_A EVAL_B VARIANT EXTRA OUT [KIND]
# Матч: таблица — флаг --hash N во входе EXTRA (без него — 16 МБ, умолчание runner match), таблицы сторон — --hash-a/--hash-b N
# (без них — как --hash); по реальному времени — --time T во входе
# EXTRA и узлы 0 (merge.py plan): тогда без --nodes-a/--nodes-b и с параллельностью max(1, физических ядер / 2), чтобы
# партии не делили ядра (время на ход не искажается). Параллельность уменьшается, если таблицы обеих сторон партии не помещаются
# в 60 % доступной памяти.
# Порядок: проверка тождества (ci/canary.sh против ci/canary.expected) -> матч с журналом пар под timeout -> OUT/shard.json.
# Расхождение тождества: матч не играется, осколок помечается canary=mismatch и в сумму не идёт.
set -uo pipefail
BIN=$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
cd "$(dirname "$0")/.."
T0=$(date +%s)
ev() { case "$1" in simple|SIMPLE) echo simple ;; *) if [ -f "weights/$1" ]; then echo "weights/$1"; else echo "weights/$1.dw"; fi ;; esac; }
EA=$(ev "$EVAL_A"); EB=$(ev "$EVAL_B")
CONC=$(nproc)
G=$GAMES; [ "$G" = 0 ] && G=2000000           # по времени: заведомо больше, чем успеет; остановит timeout
python3 - "$OUT/meta.json" <<PY
import json, sys
json.dump({"shard": $SHARD, "seed": $SEED, "games_target": $GAMES, "minutes": $MINUTES, "nodes_a": $NODES_A, "nodes_b": $NODES_B,
           "eval_a": "$EVAL_A", "eval_b": "$EVAL_B", "variant": "$VARIANT", "extra": "$EXTRA", "experiment": "${EXPERIMENT:-}", "kind": "${KIND:-match}",
           "commit": "${GITHUB_SHA:-}", "runner": "${RUNNER_LABEL:-}", "cpus": $CONC, "canary": "pending", "rc": None, "secs": 0},
          open(sys.argv[1], "w"), indent=1)
PY
setmeta() { python3 -c 'import json,sys; f=sys.argv[1]; m=json.load(open(f)); m[sys.argv[2]]=json.loads(sys.argv[3]); json.dump(m,open(f,"w"),indent=1)' "$OUT/meta.json" "$1" "$2"; }
lscpu > "$OUT/cpu.txt" 2>/dev/null || true
ci/canary.sh "$BIN" > "$OUT/canary.txt" 2>&1
if diff -u ci/canary.expected "$OUT/canary.txt" > "$OUT/canary.diff"; then
  setmeta canary '"ok"'
else
  setmeta canary '"mismatch"'
  echo "::error::осколок $SHARD: тождество не совпало с эталоном — матч не играется"
  cat "$OUT/canary.diff"
  setmeta secs "$(( $(date +%s) - T0 ))"
  python3 ci/merge.py shard "$OUT" > /dev/null
  exit 0
fi
if [ "${KIND:-match}" != match ]; then
  KIND="$KIND" BIN="$BIN" OUT="$OUT" T0="$T0" EA="$EA" CONC="$CONC" SHARD="$SHARD" SEED="$SEED" GAMES="$GAMES" MINUTES="$MINUTES" \
    NODES_A="$NODES_A" EVAL_B="$EVAL_B" VARIANT="$VARIANT" EXTRA="$EXTRA" ci/work.sh
  RC=$?
  setmeta rc "$RC"
  setmeta secs "$(( $(date +%s) - T0 ))"
  python3 ci/merge.py work "$OUT"
  tail -n 3 "$OUT/work.log" 2>/dev/null
  case "$RC" in 0|124|137) exit 0 ;; *) echo "::error::осколок $SHARD: код $RC"; exit 1 ;; esac
fi
HASH=$(printf ' %s ' "$EXTRA" | sed -nE 's/.* --hash ([0-9]+) .*/\1/p'); [ -n "$HASH" ] || HASH=16
HASH_A=$(printf ' %s ' "$EXTRA" | sed -nE 's/.* --hash-a ([0-9]+) .*/\1/p'); [ -n "$HASH_A" ] || HASH_A=$HASH
HASH_B=$(printf ' %s ' "$EXTRA" | sed -nE 's/.* --hash-b ([0-9]+) .*/\1/p'); [ -n "$HASH_B" ] || HASH_B=$HASH
MCONC=$CONC
CTRL="--nodes-a $NODES_A --nodes-b $NODES_B"
case " $EXTRA " in
  *" --time "*)
    CTRL=""
    PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
    [ "${PHYS:-0}" -ge 1 ] 2>/dev/null || PHYS=$CONC
    MCONC=$(( PHYS / 2 )); [ "$MCONC" -ge 1 ] || MCONC=1
    ;;
esac
MEM=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null)
if [ -n "$MEM" ]; then
  while [ "$MCONC" -gt 1 ] && [ $(( MCONC * (HASH_A + HASH_B) )) -gt $(( MEM * 6 / 10 )) ]; do MCONC=$(( MCONC - 1 )); done
fi
setmeta concurrency "$MCONC"
setmeta hash_mb "$HASH"
if [ "$HASH_A" != "$HASH" ] || [ "$HASH_B" != "$HASH" ]; then setmeta hash_mb_ab "[$HASH_A, $HASH_B]"; fi
LEFT=$(( MINUTES * 60 - ($(date +%s) - T0) ))
[ "$LEFT" -lt 60 ] && LEFT=60
( while sleep 60; do echo "$(( $(date +%s) - T0 )) $(( $(wc -l < "$OUT/pairs.jsonl" 2>/dev/null || echo 1) - 1 ))"; done ) > "$OUT/progress.txt" &
MON=$!
# shellcheck disable=SC2086  # EXTRA проверен по шаблону в merge.py plan; CTRL — пусто или узлы сторон (целые, merge.py plan)
timeout --signal=TERM --kill-after=30 "$LEFT" "$BIN" match --inproc --variant "$VARIANT" --eval-a "$EA" --eval-b "$EB" \
  $CTRL --games "$G" --seed "$SEED" --concurrency "$MCONC" --state "$OUT/pairs.jsonl" $EXTRA \
  > "$OUT/match.log" 2>&1
RC=$?
kill "$MON" 2>/dev/null
setmeta rc "$RC"
setmeta secs "$(( $(date +%s) - T0 ))"
python3 ci/merge.py shard "$OUT"
tail -n 2 "$OUT/match.log"
# 0 — доиграл; 124/137 — остановлен по времени (журнал годен); иное — ошибка
case "$RC" in 0|124|137) exit 0 ;; *) echo "::error::осколок $SHARD: код $RC"; exit 1 ;; esac
