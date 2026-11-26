#!/usr/bin/env python3
"""Счёт матчей по журналам пар runner match --state (без сторонних пакетов).

  merge.py canon FILE                   sha256 канонического журнала (заголовок + пары по номеру, поля p, s, pl)
  merge.py plan                         разбор параметров круга (IN_* из окружения; ci/round.sh) -> GITHUB_OUTPUT и PLAN_JSON
  merge.py shard DIR                    DIR/pairs.jsonl -> DIR/shard.json (непрерывный префикс пар 0..m-1)
  merge.py work DIR                     осколок selfplay/relabel: DIR/chunks -> DIR/shard.json (кусков, байт)
  merge.py merge ROOT EXPECTED [KIND]   ROOT/*/shard.json -> result.json, summary.md (в текущем каталоге)
  merge.py total FILE...                сумма result.json раундов -> JSON с итогом и решением long_verdict
Пентаномиал: счёт A за пару 0, 0.5, 1, 1.5, 2 (индекс = s1 + s2 в полуочках). Формулы — как в движке (matchplay.rs, learn/cycle.rs).
"""
import glob
import hashlib
import json
import math
import os
import re
import sys

ACCEPT_Z, RESOLVE_HI = 2.8, 3.0


def elo_of(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / p - 1)


def moments(penta):
    n = sum(penta)
    if not n:
        return 0, 0.5, 0.0
    mean = sum(c * i * 0.25 for i, c in enumerate(penta)) / n
    var = sum(c * (i * 0.25 - mean) ** 2 for i, c in enumerate(penta)) / n
    return n, mean, var


def interval(penta):
    n, mean, var = moments(penta)
    se = math.sqrt(var / n) if n else 0.0
    return elo_of(mean), elo_of(mean - 1.96 * se), elo_of(mean + 1.96 * se)


def nelo(penta):
    n, mean, var = moments(penta)
    sigma = math.sqrt(var) * math.sqrt(2)
    return 347.43 * (mean - 0.5) / sigma if n and sigma > 0 else 0.0


def accept_bound(lo, hi):
    s = lambda e: 1 / (1 + 10 ** (-e / 400))
    a, b = s(lo), s(hi)
    se = max(b - a, 0) / (2 * 1.96)
    return elo_of((a + b) / 2 - ACCEPT_Z * se)


def long_verdict(penta, min_pairs, max_pairs):
    """Решение длинного матча по пентаномиалу, как в движке: continue / accept / nogain / unresolved."""
    n = sum(penta)
    _, lo, hi = interval(penta)
    mn = max(min_pairs, 1)
    if n < mn:
        return "continue"
    if accept_bound(lo, hi) > 0:
        return "accept"
    if hi < RESOLVE_HI:
        return "nogain"
    if n >= max(max_pairs, mn):
        return "unresolved"
    return "continue"


def read_journal(path):
    """(заголовок, {номер: запись}); оборванные и повторные строки пропускаются."""
    head, recs = None, {}
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    except FileNotFoundError:
        return None, {}
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            v = json.loads(line)
        except ValueError:
            continue
        if i == 0:
            head = v
            continue
        if isinstance(v, dict) and "p" in v and "s" in v:
            recs.setdefault(int(v["p"]), v)
    return head, recs


def canon(path):
    head, recs = read_journal(path)
    out = json.dumps(head, sort_keys=True, separators=(",", ":")) + "\n"
    for p in sorted(recs):
        r = recs[p]
        out += json.dumps({"p": r["p"], "s": r["s"], "pl": r.get("pl")}, separators=(",", ":")) + "\n"
    return hashlib.sha256(out.encode()).hexdigest()


def score(recs, upto):
    penta, wdl, plies = [0] * 5, [0, 0, 0], 0
    for p in range(upto):
        r = recs[p]
        for s in r["s"]:
            wdl[0 if s == 2 else 2 if s == 0 else 1] += 1
        penta[r["s"][0] + r["s"][1]] += 1
        plies += sum(r.get("pl") or [0, 0])
    return penta, wdl, plies


def report(penta, wdl):
    n = sum(penta)
    e, lo, hi = interval(penta)
    games = 2 * n
    return {"pairs": n, "games": games, "wdl": wdl, "penta": penta, "score": round(moments(penta)[1], 5),
            "elo": round(e, 2), "elo_lo": round(lo, 2), "elo_hi": round(hi, 2), "nelo": round(nelo(penta), 2),
            "draws": round(wdl[1] / games, 4) if games else 0.0}


def env_int(name, lo, hi):
    v = os.environ.get(name, "").strip()
    if not re.fullmatch(r"\d{1,12}", v) or not lo <= int(v) <= hi:
        sys.exit(f"вход {name}: нужно целое {lo}..{hi}, получено '{v}'")
    return int(v)


def env_re(name, pat):
    v = os.environ.get(name, "").strip()
    if not re.fullmatch(pat, v):
        sys.exit(f"вход {name}: недопустимое значение '{v}'")
    return v


def plan():
    exp = env_re("IN_EXPERIMENT", r"[A-Za-z0-9._-]{1,64}")
    na, nb = env_int("IN_NODES_A", 0, 10**10), env_int("IN_NODES_B", 0, 10**10)
    games, shards = env_int("IN_GAMES", 0, 10**7), env_int("IN_SHARDS", 1, 40)
    seed_base, minutes = env_int("IN_SEED_BASE", 0, 10**15), env_int("IN_MINUTES", 1, 340)
    variant = env_re("IN_VARIANT", r"[a-z]{2,20}")
    runner = env_re("IN_RUNNER", r"x86|arm")
    extra = env_re("IN_EXTRA", r"[A-Za-z0-9 ._,=/+-]{0,200}")
    kind = os.environ.get("IN_KIND", "").strip() or "match"
    if kind not in ("match", "selfplay", "relabel"):
        sys.exit(f"вход kind: недопустимое значение '{kind}'")
    # самоигра: ещё флаги рецепта (дебют, случайный ход в середине партии) в этом порядке; шаблон применяется к " " + extra
    if kind == "selfplay" and extra and not re.fullmatch(r"( --free)?( --book-min \d{1,3} --book-max \d{1,3})?( --rnd-prob (0|1|0?\.\d{1,3}|1\.0{1,3}) --rnd-min \d{1,3} --rnd-max \d{1,3})?", " " + extra):
        sys.exit("вход extra: для selfplay допустимы только --free и флаги рецепта (--book-min/--book-max, --rnd-prob/--rnd-min/--rnd-max)")
    if kind == "relabel" and extra not in ("", "--free"):
        sys.exit("вход extra: для relabel допустимо только --free")
    if re.search(r"--(state|concurrency|games|seed|inproc|eval-[ab]|nodes-[ab]|variant)\b", extra):
        sys.exit("вход extra: эти флаги задаёт сценарий")
    # матч: таблица --hash N (МБ, 1..1024) и время на ход --time T (мс, 1..60000; узлы сторон тогда 0 — ci/shard.sh играет без
    # --nodes-a/--nodes-b) — каждый не больше одного раза; по узлам — узлы сторон ≥ 1
    words = extra.split()
    timed = False
    for flag, hi in (("--hash", 1024), ("--time", 60000)):
        at = [i for i, w in enumerate(words) if w == flag]
        if len(at) > 1:
            sys.exit(f"вход extra: {flag} задан больше одного раза")
        if at:
            v = words[at[0] + 1] if at[0] + 1 < len(words) else ""
            if kind != "match" or not re.fullmatch(r"\d{1,6}", v) or not 1 <= int(v) <= hi:
                sys.exit(f"вход extra: {flag} — целое 1..{hi}, только у матча (получено '{v}')")
            timed = timed or flag == "--time"
    if timed and (na or nb or re.search(r"--(nodes|time-nodes)\b", extra)):
        sys.exit("вход extra: матч по времени (--time) — узлы сторон 0, без --nodes/--time-nodes")
    if not timed and not (na and nb):
        sys.exit("вход nodes_a/nodes_b: нужно целое ≥ 1 (0 — только у матча по времени, --time во входе extra)")
    for k in ("IN_EVAL_A", "IN_EVAL_B"):
        v = env_re(k, r"[A-Za-z0-9._-]{1,64}")
        if v.lower() != "simple" and not (os.path.isfile(f"weights/{v}") or os.path.isfile(f"weights/{v}.dw")):
            sys.exit(f"вход {k}: нет файла weights/{v} (или simple)")
    if kind == "relabel" and not re.fullmatch(r"l-[0-9a-f]{32}\.txt", os.environ.get("IN_EVAL_B", "")):
        sys.exit("вход eval_b: для relabel нужен список кусков l-<хеш>.txt")
    pairs = (games + 1) // 2
    inc = []
    for k in range(shards):
        if kind == "selfplay":
            g = games          # потолок партий на осколок (0 — по времени)
        elif kind == "relabel":
            g = 0
        else:
            g = 0 if games == 0 else 2 * (pairs // shards + (1 if k < pairs % shards else 0))
            if games and not g:
                continue
        inc.append({"shard": k, "seed": seed_base + k, "games": g})
    label, arch = ("ubuntu-24.04-arm", "aarch64") if runner == "arm" else ("ubuntu-24.04", "x86_64")
    # в матрицу (видна в именах заданий) — только номер осколка; зерно и партии осколок берёт из PLAN_JSON (ci/round.sh)
    out = {"matrix": json.dumps({"include": [{"shard": x["shard"]} for x in inc]}), "count": str(len(inc)), "label": label, "arch": arch}
    if os.environ.get("PLAN_JSON"):
        json.dump({"include": inc, "count": len(inc), "label": label, "arch": arch}, open(os.environ["PLAN_JSON"], "w"))
    with open(os.environ.get("GITHUB_OUTPUT", "/dev/stdout"), "a") as f:
        for k, v in out.items():
            f.write(f"{k}={v}\n")
    ctl = f"по времени: {words[words.index('--time') + 1]} мс на ход" if timed else f"A {na} B {nb}"
    print(f"эксперимент {exp} ({kind}): осколков {len(inc)}, машина {label}, партий {games or 'по времени'}, {ctl}, минут {minutes}")


def shard(d):
    meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
    head, recs = read_journal(os.path.join(d, "pairs.jsonl"))
    m = 0
    while m in recs:
        m += 1
    penta, wdl, plies = score(recs, m)
    meta.update({"pairs_prefix": m, "pairs_all": len(recs), "plies": plies, "journal_hash": (head or {}).get("hash")})
    meta.update(report(penta, wdl))
    # матч по времени: средние узлы в секунду сторон (строка «speed: nps_a=… nps_b=…» runner match; по узлам её нет)
    log = os.path.join(d, "match.log")
    sp = re.findall(r"^speed: nps_a=(\d+) nps_b=(\d+)", open(log, errors="replace").read(), re.M) if os.path.exists(log) else []
    if sp:
        meta.update({"nps_a": int(sp[-1][0]), "nps_b": int(sp[-1][1])})
    json.dump(meta, open(os.path.join(d, "shard.json"), "w"), indent=1)
    print(json.dumps(meta))


def work(d):
    """Осколок selfplay/relabel: сколько кусков и байт выгружается (партии считает и проверяет движок на Mac)."""
    meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
    files = sorted(glob.glob(os.path.join(d, "chunks", "games-*.dgz")))
    meta.update({"chunks": len(files), "bytes": sum(os.path.getsize(f) for f in files)})
    json.dump(meta, open(os.path.join(d, "shard.json"), "w"), indent=1)
    print(json.dumps(meta))


def merge_work(root, expected, kind):
    rows, got, bad, chunks, size, secs = [], [], [], 0, 0, 0.0
    for f in sorted(glob.glob(os.path.join(root, "*", "shard.json"))):
        s = json.load(open(f))
        rows.append(s)
        if s.get("canary") != "ok":
            bad.append(s.get("shard"))
            continue
        got.append(s.get("shard"))
        chunks += s.get("chunks", 0)
        size += s.get("bytes", 0)
        secs += s.get("secs", 0)
    res = {"kind": kind, "chunks": chunks, "bytes": size, "shards_expected": expected, "shards_ok": len(got), "shards_rejected": bad,
           "shards_missing": expected - len(rows), "shard_hours": round(secs / 3600, 2),
           "params": {k: rows[0].get(k) for k in ("experiment", "eval_a", "eval_b", "nodes_a", "variant", "extra", "commit", "runner")} if rows else {}}
    json.dump({"result": res, "shards": rows}, open("result.json", "w"), indent=1)
    md = [f"## Итог ({kind}): {res['params'].get('experiment', '?')}", "",
          f"Осколков: ожидалось {expected}, получено {len(rows)}, годных {len(got)}, забраковано (тождество) {bad or 'нет'}, нет итога {res['shards_missing']}.",
          f"Кусков {chunks}, {size / 1048576:.1f} МБ; партии проверяет и забирает движок на Mac.", "",
          "| осколок | зерно | тождество | код | кусков | МБ | сек |", "|---|---|---|---|---|---|---|"]
    for s in rows:
        md.append(f"| {s.get('shard')} | {s.get('seed')} | {s.get('canary')} | {s.get('rc')} | {s.get('chunks')} | {s.get('bytes', 0) / 1048576:.1f} | {s.get('secs')} |")
    open("summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    if not got:
        sys.exit("ни одного годного осколка")


def merge(root, expected, kind="match"):
    if kind != "match":
        return merge_work(root, expected, kind)
    got, bad, penta, wdl, rows, secs = [], [], [0] * 5, [0, 0, 0], [], 0.0
    for f in sorted(glob.glob(os.path.join(root, "*", "shard.json"))):
        s = json.load(open(f))
        rows.append(s)
        if s.get("canary") != "ok":
            bad.append(s.get("shard"))
            continue
        got.append(s.get("shard"))
        penta = [a + b for a, b in zip(penta, s["penta"])]
        wdl = [a + b for a, b in zip(wdl, s["wdl"])]
        secs += s.get("secs", 0)
    res = report(penta, wdl)
    sp = [(s["nps_a"], s["nps_b"]) for s in rows if s.get("canary") == "ok" and s.get("nps_a") and s.get("nps_b")]
    if sp:
        res.update({"nps_a": round(sum(x[0] for x in sp) / len(sp)), "nps_b": round(sum(x[1] for x in sp) / len(sp))})
    res.update({"shards_expected": expected, "shards_ok": len(got), "shards_rejected": bad,
                "shards_missing": expected - len(rows), "shard_hours": round(secs / 3600, 2),
                "params": {k: rows[0].get(k) for k in ("experiment", "eval_a", "eval_b", "nodes_a", "nodes_b", "variant", "extra", "commit", "runner")} if rows else {}})
    json.dump({"result": res, "shards": rows}, open("result.json", "w"), indent=1)
    md = [f"## Итог: {res['params'].get('experiment', '?')}", "",
          f"Осколков: ожидалось {expected}, получено {len(rows)}, в сумме {len(got)}, забраковано (тождество) {bad or 'нет'}, нет итога {res['shards_missing']}.",
          "", f"Пар {res['pairs']} (партий {res['games']}), W/D/L {wdl[0]}/{wdl[1]}/{wdl[2]}, ничьих {res['draws']:.3f}, penta {penta}",
          f"Эло {res['elo']:+.1f} [{res['elo_lo']:+.1f}, {res['elo_hi']:+.1f}] (95 %), nElo {res['nelo']:+.1f}, счёт {res['score']:.4f}"]
    if sp:
        md.append(f"Узлов в секунду (среднее {len(sp)} осколков): A {res['nps_a']}, B {res['nps_b']}")
    md += ["",
          "| осколок | зерно | тождество | код | пар (префикс/всего) | сек | Эло |", "|---|---|---|---|---|---|---|"]
    for s in rows:
        md.append(f"| {s.get('shard')} | {s.get('seed')} | {s.get('canary')} | {s.get('rc')} | {s.get('pairs_prefix')}/{s.get('pairs_all')} | {s.get('secs')} | {s.get('elo', 0):+.1f} |")
    open("summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    if not got:
        sys.exit("ни одного годного осколка")


def total(files, min_pairs, max_pairs):
    penta, wdl, rounds, commits = [0] * 5, [0, 0, 0], 0, []
    for f in files:
        r = json.load(open(f))["result"]
        c = (r.get("params") or {}).get("commit")
        if c and c not in commits:
            commits.append(c)
        penta = [a + b for a, b in zip(penta, r["penta"])]
        wdl = [a + b for a, b in zip(wdl, r["wdl"])]
        rounds += 1
    res = report(penta, wdl)
    _, lo, hi = interval(penta)
    res.update({"rounds": rounds, "accept_bound": round(accept_bound(lo, hi), 2) if sum(penta) else None,
                "verdict": long_verdict(penta, min_pairs, max_pairs), "min_pairs": min_pairs, "max_pairs": max_pairs, "commits": commits})
    if len(commits) > 1:  # раунды сыграны разными сборками — складывать нельзя
        res["verdict"] = "mixed_builds"
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["canon"]:
        print(canon(a[1]))
    elif a[:1] == ["plan"]:
        plan()
    elif a[:1] == ["shard"]:
        shard(a[1])
    elif a[:1] == ["work"]:
        work(a[1])
    elif a[:1] == ["merge"]:
        merge(a[1], int(a[2]), a[3] if len(a) > 3 else "match")
    elif a[:1] == ["total"]:
        total(a[3:], int(a[1]), int(a[2]))
    elif a[:1] == ["selftest"]:
        lv = lambda p: long_verdict(p, 400, 4000)
        sc = lambda p, k: [x * k for x in p]
        assert lv([0, 10, 100, 60, 30]) == "continue"
        assert lv(sc([0, 10, 100, 60, 30], 2)) == "accept"
        assert lv([30, 90, 200, 70, 10]) == "nogain"
        assert lv([15, 45, 100, 35, 5]) == "continue"
        assert all(lv(sc([2, 8, 20, 8, 2], k)) == "continue" for k in (10, 20, 50, 90))
        assert lv(sc([2, 8, 20, 8, 2], 100)) == "unresolved"
        print("selftest ok")
    else:
        sys.exit(__doc__)
