#!/usr/bin/env bash
# Работа осколка, не матч (вызывается из ci/shard.sh после проверки тождества). Окружение — как у shard.sh, плюс KIND, EA (путь
# весов), CONC, T0. Итог: OUT/chunks/games-*.dgz (и relabel.json у переоценки), журнал OUT/work.log; код — как у runner.
#   selfplay: куски по GPC партий самоигры сетью EA на NODES_A узлов, зерно осколка SEED (seed_base круга + номер осколка:
#             зёрна не пересекаются ни между осколками, ни между кругами), без баз; по одному куску, пока хватает времени
#             (кусок пишется только целиком) и пока не набран потолок GAMES партий (0 — по времени).
#   relabel:  архив кусков этого осколка — строка SHARD+1 списка weights/EVAL_B (c-<хеш>.tar.<kfp>.enc из выпуска
#             rounds-inputs, ci/fetch_inputs.sh: расшифровка, sha256 по манифесту); переоценка сетью EA на NODES_A узлов без баз (runner learn relabel --no-egdb --any-eval).
#   bookleaf: листья дебютной книги — архив заданий этого осколка (строка SHARD+1 списка weights/EVAL_B, в архиве один файл
#             tasks.tsv); расчёт сетью EA: runner booktask, 1 поток на задание, таблица и опции — из EXTRA (--hash N --opts S),
#             без баз; итог OUT/chunks/book.jsonl дописывается по заданию — срок обрывает только незаконченные. В журнал — счётчики.
#   train:    одно обучение с зерном SEED закрытым исполняемым входом круга (ci/train.sh); итог — OUT/train/.
set -uo pipefail
TMPROOT=$(mktemp -d)
trap 'rm -rf "$TMPROOT"' EXIT
export RUNNER_DATA="$TMPROOT"      # пустой корень данных: ни реестра, ни баз этой машины
mkdir -p "$OUT/chunks"
left() { echo $(( MINUTES * 60 - ($(date +%s) - T0) )); }
case "$KIND" in
  selfplay)
    GPC=${GPC:-2000}
    # Рецепт самоигры (дебют --book-min/--book-max, случайный ход --rnd-prob/--rnd-min/--rnd-max) приходит во входе extra
    # (шаблон — merge.py plan). Без рецепта дебюта — прежние 6–12 случайных полуходов.
    BOOK="--book-min 6 --book-max 12"
    case " $EXTRA " in *" --book-min "*) BOOK="" ;; esac
    i=0; made=0; last=0; rc=0
    while :; do
      [ "$GAMES" -gt 0 ] && [ "$made" -ge "$GAMES" ] && break
      L=$(left)
      [ "$L" -lt $(( last + last / 5 + 30 )) ] && break
      n=$GPC
      [ "$GAMES" -gt 0 ] && [ $(( GAMES - made )) -lt "$n" ] && n=$(( GAMES - made ))
      t=$(date +%s)
      # shellcheck disable=SC2086  # EXTRA проверен по шаблону в merge.py plan (--free и флаги рецепта; BOOK — пусто или ниже)
      timeout --signal=TERM --kill-after=20 "$L" "$BIN" selfplay-data --variant "$VARIANT" $EXTRA --out "$OUT/chunks" --chunks 1 \
        --first-chunk "$i" --games-per-chunk "$n" --nodes "$NODES_A" --seed "$SEED" --threads "$CONC" --no-egdb --eval "$EA" \
        $BOOK --hash 8 --max-plies 300 >> "$OUT/work.log" 2>&1
      rc=$?
      [ "$rc" -ne 0 ] && break
      last=$(( $(date +%s) - t )); i=$(( i + 1 )); made=$(( made + n ))
      echo "$(( $(date +%s) - T0 )) $made" >> "$OUT/progress.txt"
    done
    exit "$rc"
    ;;
  relabel)
    NAME=$(sed -n "$(( SHARD + 1 ))p" "weights/$EVAL_B" | tr -d '[:space:]')
    [[ "$NAME" =~ ^c-([0-9a-f]{32})\.tar$ ]] || { echo "::error::в списке $EVAL_B нет архива для осколка $SHARD" | tee -a "$OUT/work.log"; exit 2; }
    want=${BASH_REMATCH[1]}
    # зашифрованный вход <имя>.<kfp>.enc, sha256 сверяется с манифестом круга (ci/fetch_inputs.sh)
    ci/fetch_inputs.sh --to "$TMPROOT/in.tar" "$NAME" >> "$OUT/work.log" 2>&1 || exit 3
    : "$want"
    # в архиве — только плоские имена games-NNNNNN.dgz
    if tar -tf "$TMPROOT/in.tar" | grep -vqE '^games-[0-9]{6}\.dgz$'; then echo "::error::$NAME: посторонние имена в архиве" | tee -a "$OUT/work.log"; exit 5; fi
    mkdir -p "$TMPROOT/in" && tar -xf "$TMPROOT/in.tar" -C "$TMPROOT/in" --no-same-owner --no-same-permissions || exit 6
    echo "вход $NAME: $(ls "$TMPROOT/in" | wc -l) кусков" >> "$OUT/work.log"
    # shellcheck disable=SC2086
    timeout --signal=TERM --kill-after=30 "$(left)" "$BIN" learn relabel --variant "$VARIANT" $EXTRA --from "$TMPROOT/in" --out "$OUT/chunks" \
      --eval "$EA" --nodes "$NODES_A" --threads "$CONC" --no-egdb --any-eval >> "$OUT/work.log" 2>&1
    exit $?
    ;;
  bookleaf)
    NAME=$(sed -n "$(( SHARD + 1 ))p" "weights/$EVAL_B" | tr -d '[:space:]')
    [[ "$NAME" =~ ^c-[0-9a-f]{32}\.tar$ ]] || { echo "::error::в списке нет архива заданий для осколка $SHARD" | tee -a "$OUT/work.log"; exit 2; }
    ci/fetch_inputs.sh --to "$TMPROOT/in.tar" "$NAME" >> "$OUT/work.log" 2>&1 || exit 3
    if [ "$(tar -tf "$TMPROOT/in.tar")" != "tasks.tsv" ]; then echo "::error::архив заданий: посторонние имена" | tee -a "$OUT/work.log"; exit 5; fi
    mkdir -p "$TMPROOT/in" && tar -xf "$TMPROOT/in.tar" -C "$TMPROOT/in" --no-same-owner --no-same-permissions || exit 6
    HASH=$(printf ' %s ' "$EXTRA" | sed -nE 's/.* --hash ([0-9]+) .*/\1/p')
    OPTS=$(printf ' %s ' "$EXTRA" | sed -nE 's/.* --opts ([a-z0-9=,]+) .*/\1/p')
    [ -n "$HASH" ] && [ -n "$OPTS" ] || { echo "::error::нет таблицы или опций расчёта" | tee -a "$OUT/work.log"; exit 2; }
    FREE=""; case " $EXTRA " in *" --free "*) FREE="--free" ;; esac
    L=$(left); S=$(( L - 120 )); [ "$S" -ge 60 ] || S=60
    # shellcheck disable=SC2086
    RUNNER_EGDB=off timeout --signal=TERM --kill-after=30 "$L" "$BIN" booktask --variant "$VARIANT" $FREE --tasks "$TMPROOT/in/tasks.tsv" \
      --out "$OUT/chunks/book.jsonl" --eval "$EA" --hash "$HASH" --opts "$OPTS" --threads "$CONC" --seconds "$S" >> "$OUT/work.log" 2>&1
    exit $?
    ;;
  train)
    TMPROOT="$TMPROOT" ci/train.sh >> "$OUT/work.log" 2>&1
    exit $?
    ;;
  *) echo "::error::неизвестный вид работы $KIND"; exit 2 ;;
esac
