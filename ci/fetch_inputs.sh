#!/usr/bin/env bash
# Входы круга из выпуска rounds-inputs этого репозитория (вызывается из ci/round.sh после ci/lib.sh manifest_open):
#   ci/fetch_inputs.sh NAME...          веса w-<32 hex>.dw, списки кусков l-<32 hex>.txt и таблица книги стороны A
#                                       k-<32 hex>.json -> weights/ (прочие имена пропускаются)
#   ci/fetch_inputs.sh --to FILE NAME   архив кусков c-<32 hex>.tar -> FILE (ci/work.sh, переоценка); вход круга train —
#                                       d-<32 hex>.tar, a-<32 hex>.json, b-<32 hex>, якоря f-<32 hex>.dfen (ci/train.sh)
# Каждый вход лежит только зашифрованным: <имя>.<kfp>.enc; открытых входов нет (файла .enc нет — ошибка круга).
# Открытый файл сверяется с sha256 из манифеста круга (ROUND_MANIFEST); имя — не хеш файла (HMAC ключа аккаунта); входа нет в манифесте — ошибка.
set -euo pipefail
cd "$(dirname "$0")/.."
. ci/lib.sh
[ -n "${ROUND_MANIFEST:-}" ] && [ -f "$ROUND_MANIFEST" ] || die "нет расшифрованного манифеста круга (входы берёт ci/round.sh)"
need_key
KFP=$(kfp)
one() { # имя путь
  local n=$1 dst=$2 want
  want=$(python3 ci/manifest.py sha "$ROUND_MANIFEST" "$n") || die "вход $n не указан в манифесте круга"
  rm -f "$dst.enc" "$dst.part"
  asset_get "$n.$KFP.enc" "$dst.enc" || die "вход $n: в rounds-inputs нет $n.$KFP.enc (открытые входы не принимаются)"
  dec "$dst.enc" "$dst.part" || { rm -f "$dst.enc" "$dst.part"; die "вход $n: расшифровка не удалась — ключ репозитория не тот (kfp $KFP)"; }
  rm -f "$dst.enc"
  [ "$(sha "$dst.part")" = "$want" ] || { rm -f "$dst.part"; die "вход $n: sha256 не совпал с манифестом"; }
  mv "$dst.part" "$dst"
  echo "вход $n: $(wc -c < "$dst") байт, sha256 совпал"
}
if [ "${1:-}" = --to ]; then
  [ $# -eq 3 ] && [[ "$3" =~ ^([cd]-[0-9a-f]{32}\.tar|a-[0-9a-f]{32}\.json|f-[0-9a-f]{32}\.dfen|l-[0-9a-f]{32}\.txt|b-[0-9a-f]{32})$ ]] || die "fetch_inputs --to FILE c-|d-<хеш>.tar | a-<хеш>.json | f-<хеш>.dfen | b-<хеш>"
  one "$3" "$2"
  exit 0
fi
for n in "$@"; do
  [[ "$n" =~ ^([wl]-[0-9a-f]{32}\.(dw|txt)|k-[0-9a-f]{32}\.json)$ ]] || continue
  [ -f "weights/$n" ] && continue
  one "$n" "weights/$n"
done
