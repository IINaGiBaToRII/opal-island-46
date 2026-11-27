#!/usr/bin/env bash
# Расшифровка входа круга: ci/decrypt.sh IN OUT. Ключ — секрет ROUNDS_KEY репозитория (переменная окружения, не аргумент).
# Параметры — те же, что у движка, который выкладывает вход: openssl enc AES-256-CBC, ключ из пароля PBKDF2-HMAC-SHA256
# (200000 итераций), соль в заголовке. Нет ключа или ключ не тот — код не 0 (файл OUT не годен).
set -euo pipefail
[ -n "${ROUNDS_KEY:-}" ] || { echo "нет ROUNDS_KEY" >&2; exit 1; }
openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -md sha256 -in "$1" -out "$2" -pass env:ROUNDS_KEY
