# shellcheck shell=bash
# Общее для сценариев круга (подключается: . ci/lib.sh). Шифр — как у Mac: openssl enc AES-256-CBC, ключ из пароля
# PBKDF2-HMAC-SHA256 (200000 итераций), соль в заголовке; пароль — секрет ROUNDS_KEY (только через окружение).
# Отпечаток ключа kfp — первые 12 знаков sha256 байтов ROUNDS_KEY (без перевода строки); зашифрованный файл — <имя>.<kfp>.enc.
# Выпуск rounds-inputs: ROUNDS_INPUTS_URL (по умолчанию выпуск этого репозитория; file://КАТАЛОГ — для проверки на месте).
U=${ROUNDS_INPUTS_URL:-https://github.com/${GITHUB_REPOSITORY:-}/releases/download/rounds-inputs}
die() { echo "::error::$*"; exit 1; }
need_key() { [ -n "${ROUNDS_KEY:-}" ] || die "секрета ROUNDS_KEY в этом репозитории нет — круг не играется"; }
kfp() { printf %s "$ROUNDS_KEY" | sha256sum | cut -c1-12; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
dec() { openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -md sha256 -in "$1" -out "$2" -pass env:ROUNDS_KEY 2>/dev/null; }
enc() { openssl enc -e -aes-256-cbc -pbkdf2 -iter 200000 -md sha256 -salt -in "$1" -out "$2" -pass env:ROUNDS_KEY 2>/dev/null; }
asset_get() { # имя путь: файл выпуска rounds-inputs; код не 0 — файла нет
  case "$U" in
    file://*) cp "${U#file://}/$1" "$2" 2>/dev/null ;;
    *) curl -fsSL --retry 3 --max-time 600 -o "$2" "$U/$1" 2>/dev/null ;;
  esac
}
asset_list() { # имена файлов выпуска rounds-inputs, по одному в строке (только для объяснения отказа)
  case "$U" in
    file://*) ls "${U#file://}" 2>/dev/null ;;
    *) python3 - "${GITHUB_REPOSITORY:-}" <<'PY' 2>/dev/null || true
import json, os, sys, urllib.request
api = "https://api.github.com/repos/" + sys.argv[1]
h = {"Accept": "application/vnd.github+json"}
if os.environ.get("GH_TOKEN"):
    h["Authorization"] = "Bearer " + os.environ["GH_TOKEN"]
get = lambda u: json.load(urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=60))
rid = get(api + "/releases/tags/rounds-inputs")["id"]
page = 1
while True:
    a = get(f"{api}/releases/{rid}/assets?per_page=100&page={page}")
    if not a:
        break
    for x in a:
        print(x["name"])
    page += 1
PY
    ;;
  esac
}
manifest_open() { # m-<ROUND>.<kfp>.enc -> $RT/manifest.json; проверки ключа, круга, версии; параметры -> IN_* (не в журнал)
  need_key
  [[ "${ROUND:-}" =~ ^[0-9a-f]{12}$ ]] || die "вход round: нужен id круга из 12 знаков 0-9a-f"
  KFP=$(kfp); export KFP
  RT=${RUNNER_TEMP:-${TMPDIR:-/tmp}}/round-$ROUND
  mkdir -p "$RT"; chmod 700 "$RT"
  local m="m-$ROUND.$KFP.enc" other
  if ! asset_get "$m" "$RT/m.enc"; then
    other=$(asset_list | sed -nE "s/^m-$ROUND\.([0-9a-f]{12})\.enc$/\1/p" | head -n 1)
    [ -n "$other" ] && die "ключ репозитория не тот (kfp $KFP, нужен $other)"
    die "манифест круга не найден: в rounds-inputs нет m-$ROUND.$KFP.enc (открытые входы не принимаются)"
  fi
  dec "$RT/m.enc" "$RT/manifest.json" || die "манифест m-$ROUND.$KFP.enc не расшифровался — ключ репозитория не тот (kfp $KFP)"
  [ -f ci/EXPORT_ID ] || die "в дереве нет ci/EXPORT_ID — версия кода не та"
  python3 ci/manifest.py open "$RT/manifest.json" "$ROUND" "$(tr -d '[:space:]' < ci/EXPORT_ID)" > "$RT/params.sh" 2> "$RT/manifest.err" \
    || { head -n 3 "$RT/manifest.err"; exit 1; }
  # shellcheck disable=SC1091
  . "$RT/params.sh"
  ROUND_MANIFEST=$RT/manifest.json; export ROUND_MANIFEST
}
