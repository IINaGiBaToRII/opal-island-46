#!/usr/bin/env bash
# Осколок вида train (вызывается из ci/work.sh после проверки тождества): одно обучение с зерном осколка SEED.
# Окружение — как у ci/work.sh (OUT, SEED, MINUTES, T0, CONC, TMPROOT) и параметры манифеста (ci/manifest.py open):
#   IN_TRAIN_BIN   b-<32 hex>        закрытый исполняемый вход (зашифрован; sha256 сверяется с манифестом, ci/fetch_inputs.sh)
#   IN_TRAIN_DATA  d-<32 hex>.tar[,…] набор: части одного tar (каждая < 2 ГиБ — файл выпуска), склеиваются по порядку;
#                                    только плоские имена games-<цифры>.dgz
#   IN_TRAIN_CLOUD l-<32 hex>.txt     облачный набор вместо IN_TRAIN_DATA: список кусков из пачек-артефактов (ci/cloudset.py)
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
fetch "$TMPROOT/arch.json" "$IN_TRAIN_ARCH"
ANCHOR_SHA=""
if [ -n "${IN_TRAIN_ANCHOR:-}" ]; then
  fetch "$TMPROOT/anchor.dfen" "$IN_TRAIN_ANCHOR"
  ANCHOR_SHA=$(sha256sum "$TMPROOT/anchor.dfen" | cut -d' ' -f1)
fi
chmod 700 "$TMPROOT/x"
if [ -n "${IN_TRAIN_CLOUD:-}" ]; then
  # облачный набор: список кусков l-….txt, куски — из пачек-артефактов самоигры этого репозитория
  # (REST по id, права actions: read у задания); пачка качается целиком, нужные куски вырезаются по смещению,
  # расшифровываются, у каждого сверяются sha256 и заголовок (вариант, правила, учитель, узлы); пачка сразу убирается.
  # sha256 набора пересчитывается по кускам и сверяется со списком (Mac сверит его с заказом по inputs.json)
  fetch "$TMPROOT/set.list" "$IN_TRAIN_CLOUD"
  python3 ci/cloudset.py check "$TMPROOT/set.list" > /dev/null || exit 5
  API="${GITHUB_API_URL:-https://api.github.com}"
  NPARTS=0
  for id in $(python3 ci/cloudset.py packs "$TMPROOT/set.list"); do
    curl -fsSL --retry 3 -H "Authorization: Bearer ${GH_TOKEN:-}" -H "Accept: application/vnd.github+json" \
      -o "$TMPROOT/pack" "$API/repos/${GITHUB_REPOSITORY:-}/actions/artifacts/$id/zip" || { echo "::error::пачка $id не получена"; exit 3; }
    python3 ci/cloudset.py take "$TMPROOT/set.list" "$id" "$TMPROOT/pack" "$TMPROOT/data" || exit 5
    rm -f "$TMPROOT/pack"
    NPARTS=$((NPARTS + 1))
  done
  python3 ci/cloudset.py seal "$TMPROOT/set.list" "$TMPROOT/data" > "$TMPROOT/dsha" || exit 5
else
  # набор: части получаются по одной (каждая сверена с манифестом), склеиваются потоком в tar -x и сразу убираются — на диске
  # раннера не бывает набора дважды; sha256 склейки считается здесь же и уходит в inputs.json (Mac сверяет с заказом)
  IFS=, read -r -a DPARTS <<< "$IN_TRAIN_DATA"
  { for n in "${DPARTS[@]}"; do fetch "$TMPROOT/dp" "$n" >&2; cat "$TMPROOT/dp"; rm -f "$TMPROOT/dp"; done; } \
    | python3 -c 'import hashlib, sys
h = hashlib.sha256()
for b in iter(lambda: sys.stdin.buffer.read(1 << 20), b""):
    h.update(b)
    sys.stdout.buffer.write(b)
sys.stdout.buffer.flush()
open(sys.argv[1], "w").write(h.hexdigest())' "$TMPROOT/dsha" \
    | tar -x -f - -C "$TMPROOT/data" --no-same-owner --no-same-permissions
  RCS=("${PIPESTATUS[@]}")
  [ "${RCS[0]}" = 0 ] || { echo "::error::вход круга не получен"; exit 3; }
  [ "${RCS[1]}" = 0 ] && [ "${RCS[2]}" = 0 ] || exit 6
  NPARTS=${#DPARTS[@]}
fi
# только плоские файлы games-<6–7 цифр>.dgz, без ссылок и каталогов (проверка после распаковки: набор потоком)
if find "$TMPROOT/data" -mindepth 1 \( ! -type f -o -links +1 \) | grep -q .; then echo "::error::ссылки или каталоги в наборе"; exit 5; fi
if ls -A "$TMPROOT/data" | grep -vqE '^games-[0-9]{6,7}\.dgz$'; then echo "::error::посторонние имена в наборе"; exit 5; fi
sum() { sha256sum "$1" | cut -d' ' -f1; }
python3 - "$TD/inputs.json" "$(sum "$TMPROOT/x")" "$(cat "$TMPROOT/dsha")" "$(sum "$TMPROOT/arch.json")" "$(ls "$TMPROOT/data" | wc -l)" "$CONC" "$ANCHOR_SHA" "$NPARTS" <<'PY'
import json, sys
f, b, d, a, n, t, k, p = sys.argv[1:]
m = {"bin_sha256": b, "data_sha256": d, "arch_sha256": a, "files": int(n), "threads": int(t), "data_parts": int(p)}
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
