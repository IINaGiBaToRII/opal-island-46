#!/usr/bin/env bash
# Круг на раннере: ci/round.sh prepare | shard | seal | merge. Вход запуска — только ROUND (непрозрачный id круга) и секрет
# ROUNDS_KEY; параметры — из зашифрованного манифеста m-<round>.<kfp>.enc (ci/lib.sh manifest_open, ci/manifest.py) и в
# журнал не печатаются: вывод скриптов и движка — в файлы (out/ или каталог RUNNER_TEMP), в журнал — только постоянные фразы
# и причины отказа (имена входов сокращены до w-…/l-…/c-…/d-…/a-…/b-…).
# Вид круга train (обучение) идёт тем же этапом shard: ci/shard.sh -> ci/work.sh -> ci/train.sh (закрытый исполняемый вход b-…).
#   prepare: входы, план (merge.py plan) -> GITHUB_OUTPUT: matrix (только номера осколков), count, label, arch
#   shard:   SHARD, BIN; входы, план, ci/shard.sh -> out/shard-K/ -> pack/shard-K.<kfp>.enc (зашифрованный tar каталога shard-K)
#   seal:    SHARD; без манифеста — запечатать out/shard-K, если shard не успел (шаг if: always())
#   merge:   COUNT; sealed/shard-*.<kfp>.enc -> shards/shard-K/ -> merge.py merge -> result/ -> pack/result.<kfp>.enc;
#            самоигра с cloud=1 — ещё облачная пачка ci/store.sh -> cloudpack/pack, cloudpack/store.<kfp>.enc, result/store.json
set -euo pipefail
cd "$(dirname "$0")/.."
. ci/lib.sh
STAGE=${1:-}
quiet() { # файл-журнал: в открытый журнал — только строки ::error:: без значений входов
  grep -h '^::error::' "$@" 2>/dev/null | sed -E 's/([wlcdab])-[0-9a-f]{32}/\1-…/g' | head -n 3 || true
}
if [ "$STAGE" = seal ]; then
  # шаг if: always() после Shard: запечатать то, что есть в out/shard-K (обрыв, отмена), если Shard не успел; ключ — свой
  need_key
  [[ "${SHARD:-}" =~ ^[0-9]{1,3}$ ]] || die "нет номера осколка SHARD"
  KFP=$(kfp)
  RT=${RUNNER_TEMP:-${TMPDIR:-/tmp}}/round-seal; mkdir -p "$RT"
else
  manifest_open
fi
LOG=$RT/run.log
inputs() {
  ci/fetch_inputs.sh "$IN_EVAL_A" "$IN_EVAL_B" ${IN_BOOK_A:+"$IN_BOOK_A"} >> "$LOG" 2>&1 || { quiet "$LOG"; echo "::error::входы круга не получены"; exit 1; }
}
plan() { # $1 — файл для GITHUB_OUTPUT
  if ! PLAN_JSON="$RT/plan.json" GITHUB_OUTPUT="$1" python3 ci/merge.py plan >> "$LOG" 2>&1; then
    # сообщение плана — «вход X: …значение…»; в журнал — только имя входа
    echo "::error::план круга отклонён: $(tail -n 1 "$LOG" | grep -oE '^вход [A-Za-z_]+' || echo 'проверка параметров') (подробности не печатаются)"
    exit 1
  fi
}
seal() { # корень что артефакт: tar корень/что -> pack/артефакт.<kfp>.enc
  mkdir -p pack
  tar -C "$1" -cf "$RT/$3.tar" "${@:4}" "$2"
  # во временный файл и переименование: оборванное шифрование не оставляет полуфайла под именем артефакта
  enc "$RT/$3.tar" "pack/$3.$KFP.enc.part" || die "шифрование итога не удалось"
  mv -f "pack/$3.$KFP.enc.part" "pack/$3.$KFP.enc"
  rm -f "$RT/$3.tar"
}
case "$STAGE" in
  prepare)
    inputs
    plan "${GITHUB_OUTPUT:-/dev/null}"
    echo "круг $ROUND: манифест, входы и план — готово"
    ;;
  shard)
    [[ "${SHARD:-}" =~ ^[0-9]{1,3}$ ]] || die "нет номера осколка SHARD"
    [ -x "${BIN:-}" ] || die "нет бинаря BIN"
    inputs
    plan "$RT/plan.out"
    read -r SEED GAMES < <(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); r=[x for x in p["include"] if x["shard"]==int(sys.argv[2])]; print(r[0]["seed"], r[0]["games"]) if r else sys.exit(1)' "$RT/plan.json" "$SHARD") \
      || die "осколка $SHARD нет в плане круга"
    O=out/shard-$SHARD
    mkdir -p "$O"
    cp "$LOG" "$O/inputs.log"
    set +e
    BIN="$BIN" OUT="$O" SHARD="$SHARD" SEED="$SEED" GAMES="$GAMES" EXPERIMENT="$IN_EXPERIMENT" NODES_A="$IN_NODES_A" NODES_B="$IN_NODES_B" \
      EVAL_A="$IN_EVAL_A" EVAL_B="$IN_EVAL_B" VARIANT="$IN_VARIANT" EXTRA="$IN_EXTRA" MINUTES="$IN_MINUTES" KIND="$IN_KIND" \
      BOOK_A="$IN_BOOK_A" BOOK_PLIES="$IN_BOOK_PLIES" ci/shard.sh > "$O/run.log" 2>&1
    RC=$?
    set -e
    seal out "shard-$SHARD" "shard-$SHARD"
    CAN=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("canary","?"))' "$O/meta.json" 2>/dev/null || echo "?")
    [ "$CAN" = ok ] || echo "::error::осколок $SHARD: тождество с эталоном — $CAN"
    [ "$RC" = 0 ] || { echo "::error::осколок $SHARD: ошибка (код $RC; подробности — в зашифрованном артефакте)"; exit 1; }
    echo "осколок $SHARD — готово"
    ;;
  seal)
    # каталог осколка есть — запечатать заново (в нём может быть больше, чем в прежнем артефакте); нет — оставить прежний
    if [ -d "out/shard-$SHARD" ]; then seal out "shard-$SHARD" "shard-$SHARD"; echo "осколок $SHARD: запечатано то, что есть"
    elif [ -f "pack/shard-$SHARD.$KFP.enc" ]; then echo "осколок $SHARD: артефакт уже запечатан"
    else echo "осколок $SHARD: итога нет — запечатывать нечего"; fi
    ;;
  merge)
    [[ "${COUNT:-}" =~ ^[0-9]{1,3}$ ]] || die "нет числа осколков COUNT"
    mkdir -p shards
    n=0
    for f in sealed/shard-*."$KFP".enc; do
      [ -f "$f" ] || continue
      b=$(basename "$f" ".$KFP.enc")
      [[ "$b" =~ ^shard-[0-9]{1,3}$ ]] || die "артефакт с посторонним именем"
      # плохой осколок — «нет итога» у него одного, сложение остальных идёт дальше
      if ! dec "$f" "$RT/$b.tar"; then echo "осколок $b: нет итога (не расшифровался)"; rm -f "$RT/$b.tar"; continue; fi
      if tar -tf "$RT/$b.tar" | grep -vqE "^$b(/[A-Za-z0-9._-]+)*/?$" || tar -tf "$RT/$b.tar" | grep -qE '(^|/)\.\.?(/|$)' || tar -tvf "$RT/$b.tar" | grep -qE '^[lh]'; then
        echo "осколок $b: нет итога (посторонние имена или ссылки в архиве)"; rm -f "$RT/$b.tar"; continue
      fi
      tar -xf "$RT/$b.tar" -C shards --no-same-owner --no-same-permissions
      rm -f "$RT/$b.tar"
      n=$((n + 1))
    done
    echo "осколков получено: $n из $COUNT"
    mkdir -p result
    set +e
    python3 ci/merge.py merge shards "$COUNT" "$IN_KIND" > "$RT/merge.log" 2>&1
    RC=$?
    set -e
    [ -f result.json ] && mv result.json summary.md result/
    # облачная пачка: сбой — круг принимается прежним путём (куски — в артефактах осколков)
    if [ "${IN_CLOUD:-0}" = 1 ] && [ "$IN_KIND" = selfplay ]; then
      ci/store.sh result cloudpack shards/shard-* >> "$RT/merge.log" 2>&1 || { rm -rf cloudpack result/store.json; echo "::error::облачная пачка не собрана — круг принимается прежним путём"; }
    fi
    if [ -f result/result.json ]; then
      # shellcheck disable=SC2046
      seal result result.json result summary.md $([ -f result/store.json ] && echo store.json)
    elif [ -f result/store.json ]; then
      seal result store.json result
    fi
    [ "$RC" = 0 ] || { echo "::error::сложение итога не удалось (код $RC)"; exit 1; }
    echo "итог — готово"
    ;;
  *) die "ci/round.sh prepare | shard | seal | merge" ;;
esac
