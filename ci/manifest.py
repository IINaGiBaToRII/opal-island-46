#!/usr/bin/env python3
"""Манифест круга (расшифрованный m-<round>.<kfp>.enc): {v:1, round, export_id, params:{...}, inputs:{имя: sha256}}.

  manifest.py open FILE ROUND EXPORT_ID   проверка; печатает export IN_<КЛЮЧ>=... для sh (значения не в журнал)
  manifest.py sha FILE NAME               sha256 открытого входа NAME из inputs (нет — код 1)
params — те же ключи и значения, что прежние входы workflow match (кроме round); нет ключа — прежнее умолчание.
Ошибки печатаются без значений параметров (журнал открыт).
"""
import json
import re
import shlex
import sys

DEFAULTS = {"experiment": None, "nodes_a": "20000", "nodes_b": "20000", "games": "4000", "shards": "20", "seed_base": "1",
            "variant": "russian", "eval_a": "t-nn64.dw", "eval_b": "t-nn64.dw", "minutes": "335", "runner": "arm", "extra": "",
            "kind": "match"}


def fail(msg):
    print(f"::error::{msg}", file=sys.stderr)
    sys.exit(1)


def load(path):
    try:
        m = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        fail("манифест круга не разбирается (JSON)")
    if not isinstance(m, dict) or m.get("v") != 1:
        fail("манифест круга: нужна версия v=1")
    return m


def open_(path, rnd, export_id):
    m = load(path)
    if m.get("round") != rnd:
        fail("манифест круга: поле round не совпадает с запуском")
    if m.get("export_id") != export_id:
        fail(f"версия кода не та (манифест {str(m.get('export_id'))[:40]}, в репозитории {export_id})")
    params, inputs = m.get("params"), m.get("inputs", {})
    if not isinstance(params, dict) or not isinstance(inputs, dict):
        fail("манифест круга: params и inputs — объекты")
    for n, h in inputs.items():
        # имя — HMAC ключа аккаунта (не sha256): сверка — по sha256 из манифеста, не по имени
        if not re.fullmatch(r"[wlc]-[0-9a-f]{32}\.(dw|txt|tar)", str(n)) or not re.fullmatch(r"[0-9a-f]{64}", str(h)):
            fail("манифест круга: вход inputs с недопустимым именем или sha256")
    out = []
    for k, d in DEFAULTS.items():
        v = params.get(k, d)
        if v is None:
            fail(f"манифест круга: нет параметра {k}")
        if isinstance(v, bool) or not isinstance(v, (str, int)):
            fail(f"манифест круга: параметр {k} — строка или целое")
        out.append(f"export IN_{k.upper()}={shlex.quote(str(v))}")
    extra = sorted(set(params) - set(DEFAULTS))
    if extra:
        fail(f"манифест круга: неизвестные параметры {', '.join(re.sub(r'[^a-z0-9_]', '?', str(k))[:20] for k in extra)}")
    print("\n".join(out))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["open"] and len(a) == 4:
        open_(a[1], a[2], a[3])
    elif a[:1] == ["sha"] and len(a) == 3:
        h = load(a[1]).get("inputs", {}).get(a[2])
        if not h:
            sys.exit(1)
        print(h)
    else:
        sys.exit(__doc__)
