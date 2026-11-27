#!/usr/bin/env bash
# Осколок вида train (вызывается из ci/work.sh после проверки тождества): одно обучение с зерном осколка SEED.
# Окружение — как у ci/work.sh (OUT, SEED, MINUTES, T0, CONC, TMPROOT) и параметры манифеста (ci/manifest.py open):
#   IN_TRAIN_BIN   b-<32 hex>        закрытый исполняемый вход (зашифрован; sha256 сверяется с манифестом, ci/fetch_inputs.sh)
#   IN_TRAIN_DATA  d-<32 hex>.tar    набор: только плоские имена games-<цифры>.dgz
#   IN_TRAIN_ARCH  a-<32 hex>.json   форма сети
#   IN_TRAIN_ANCHOR f-<32 hex>.dfen  набор якорей (5е; только у кругов с якорями, иначе пусто) — подстановка @ANCHOR@
#   IN_TRAIN_ARGS  команда входа целиком, с подстановками @DATA@ @ARCH@ @OUT@ @SEED@ @THREADS@ [@ANCHOR@] (шаблон — ci/merge.py plan)
# Вход и набор расшифровываются во временный каталог TMPROOT (вне OUT: в артефакт не попадают), убираются ci/work.sh.
# Итог: OUT/train/net.dw, net.dw.train.json, train.log (вывод входа — только в файл), inputs.json (sha256 входов, код, время).
# Ограничение по времени — остаток минут осколка минус запас на запечатывание; оборванное обучение оставляет контрольную
# точку (OUT/train/net.dw.ckpt) — её запечатывает шаг Seal, как у матчей. Потоки — nproc (итог от числа потоков не зависит).
set -uo pipefail
cd "$(dirname "$0")/.."
TD=$OUT/train
mkdir -p "$TD" "$TMPROOT/data" "$TMPROOT/home"
fetch() { ci/fetch_inputs.sh --to "$1" "$2" || { echo "::error::вход круга не получен"; exit 3; }; }
fetch "$TMPROOT/x" "$IN_TRAIN_BIN"
fetch "$TMPROOT/d.tar" "$IN_TRAIN_DATA"
fetch "$TMPROOT/arch.json" "$IN_TRAIN_ARCH"
ANCHOR_SHA=""
if [ -n "${IN_TRAIN_ANCHOR:-}" ]; then
  fetch "$TMPROOT/anchor.dfen" "$IN_TRAIN_ANCHOR"
  ANCHOR_SHA=$(sha256sum "$TMPROOT/anchor.dfen" | cut -d' ' -f1)
fi
chmod 700 "$TMPROOT/x"
if tar -tf "$TMPROOT/d.tar" | grep -vqE '^(\./)?games-[0-9]{6,7}\.dgz$'; then echo "::error::посторонние имена в наборе"; exit 5; fi
if tar -tvf "$TMPROOT/d.tar" | grep -qE '^[lh]'; then echo "::error::ссылки в наборе"; exit 5; fi
tar -xf "$TMPROOT/d.tar" -C "$TMPROOT/data" --no-same-owner --no-same-permissions || exit 6
rm -f "$TMPROOT/d.tar"
sum() { sha256sum "$1" | cut -d' ' -f1; }
python3 - "$TD/inputs.json" "$(sum "$TMPROOT/x")" "$(python3 ci/manifest.py sha "$ROUND_MANIFEST" "$IN_TRAIN_DATA")" "$(sum "$TMPROOT/arch.json")" "$(ls "$TMPROOT/data" | wc -l)" "$CONC" "$ANCHOR_SHA" <<'PY'
import json, sys
f, b, d, a, n, t, k = sys.argv[1:]
m = {"bin_sha256": b, "data_sha256": d, "arch_sha256": a, "files": int(n), "threads": int(t)}
if k:
    m["anchor_sha256"] = k
json.dump(m, open(f, "w"), indent=1)
PY
ARGS=()
for w in $IN_TRAIN_ARGS; do
  case "$w" in
    @DATA@) ARGS+=("$TMPROOT/data") ;;
    @ARCH@) ARGS+=("$TMPROOT/arch.json") ;;
    @OUT@) ARGS+=("$TD/net.dw") ;;
    @SEED@) ARGS+=("$SEED") ;;
    @THREADS@) ARGS+=("$CONC") ;;
    @ANCHOR@) ARGS+=("$TMPROOT/anchor.dfen") ;;
    *) ARGS+=("$w") ;;
  esac
done
LEFT=$(( MINUTES * 60 - ($(date +%s) - T0) - 120 ))
[ "$LEFT" -lt 60 ] && LEFT=60
t=$(date +%s)
# HOME — временный: вход не читает и не пишет ничего вне своих путей на машине
HOME="$TMPROOT/home" timeout --signal=TERM --kill-after=60 "$LEFT" "$TMPROOT/x" "${ARGS[@]}" > "$TD/train.log" 2>&1
RC=$?
python3 - "$TD/inputs.json" "$RC" "$(( $(date +%s) - t ))" <<'PY'
import json, sys
f = sys.argv[1]; m = json.load(open(f)); m.update({"rc": int(sys.argv[2]), "secs": int(sys.argv[3])}); json.dump(m, open(f, "w"), indent=1)
PY
# итог есть — контрольная точка не нужна (весит в разы больше весов); нет — она остаётся для продолжения
# (с усреднением весов --avg рядом файл среднего net.dw.ckpt.avg — с ней же)
[ "$RC" = 0 ] && [ -s "$TD/net.dw" ] && rm -f "$TD/net.dw.ckpt" "$TD/net.dw.ckpt.avg"
exit "$RC"
