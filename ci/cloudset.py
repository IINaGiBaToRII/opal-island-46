#!/usr/bin/env python3
"""Облачный набор обучения: куски набора поколения — из пачек-артефактов самоигры этого
репозитория по списку l-….txt (его собирает Mac по описи набора поколения). Вызывается из ci/train.sh; пачку по id качает train.sh.

  cloudset.py check LIST              список годен (строки, репозиторий = GITHUB_REPOSITORY); печатает число кусков
  cloudset.py packs LIST              id пачек списка, по строке
  cloudset.py take LIST ID PACK OUT   куски пачки ID: вырезать по смещению, расшифровать (ROUNDS_KEY, как ci/lib.sh dec),
                                      сверить sha256 открытого и заголовок (вариант, правила, учитель, узлы) — в OUT/<имя>
  cloudset.py seal LIST OUT           все куски на месте, лишних нет; sha256 набора (строки «имя sha256») = заголовку списка;
                                      печатает его
Код 5 — порча или подмена (обучение не начинается), 2 — вызов.
"""
import concurrent.futures, hashlib, json, os, re, struct, subprocess, sys, tempfile


def die(msg, code=5):
    print(f"::error::{msg}", file=sys.stderr)
    sys.exit(code)


def load(path):
    head, rows = {}, []
    for ln in open(path, encoding="utf-8"):
        w = ln.split()
        if not w or w[0].startswith("#"):
            continue
        if w[0] == "chunk":
            if len(w) != 7 or not re.fullmatch(r"games-[0-9]{6,7}\.dgz", w[1]) or not re.fullmatch(r"[0-9a-f]{64}", w[2]) or not all(x.isdigit() for x in w[3:]):
                die("список облачного набора: строка куска не годится")
            rows.append({"name": w[1], "sha": w[2], "pack": w[3], "off": int(w[4]), "len": int(w[5]), "bytes": int(w[6])})
        elif len(w) == 2 and w[0] in ("set", "repo", "variant", "forced", "nodes", "teachers"):
            head[w[0]] = w[1]
        else:
            die("список облачного набора: неизвестная строка")
    for k in ("set", "repo", "variant", "forced", "nodes", "teachers"):
        if k not in head:
            die(f"список облачного набора: нет поля {k}")
    if not re.fullmatch(r"[0-9a-f]{64}", head["set"]) or not rows or any(r["len"] == 0 for r in rows):
        die("список облачного набора: пуст или без sha набора")
    if len({r["name"] for r in rows}) != len(rows) or len({r["sha"] for r in rows}) != len(rows):
        die("список облачного набора: имена или sha256 повторяются")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if repo and repo != head["repo"]:
        die("список облачного набора: пачки в другом репозитории")
    return head, rows


def header(b):
    if len(b) < 8 or b[:4] != b"DGM1":
        return None
    n = struct.unpack("<I", b[4:8])[0]
    try:
        return json.loads(b[8:8 + n])
    except ValueError:
        return None


def dec_one(enc, dec):
    return subprocess.run(["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-md", "sha256", "-in", enc, "-out", dec,
                           "-pass", "env:ROUNDS_KEY"], stderr=subprocess.DEVNULL).returncode == 0


def take(lst, pid, pack, out):
    head, rows = load(lst)
    teachers = set(head["teachers"].split(","))
    os.makedirs(out, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(out)) or ".")
    mine = [r for r in rows if r["pack"] == pid]
    # PBKDF2 на кусок — параллельно по числу ядер (как ci/store.sh), окнами: на диске — не больше окна вырезанных кусков
    win = max(2, os.cpu_count() or 2) * 8
    n = 0
    with open(pack, "rb") as f, concurrent.futures.ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 2)) as ex:
        size = os.fstat(f.fileno()).st_size
        for at in range(0, len(mine), win):
            part = mine[at:at + win]
            jobs = []
            for i, r in enumerate(part):
                if r["off"] + r["len"] > size:
                    die(f"кусок {r['name']}: за концом пачки {pid}")
                f.seek(r["off"])
                enc, dec = os.path.join(tmp, f"{i}.enc"), os.path.join(tmp, f"{i}.dgz")
                open(enc, "wb").write(f.read(r["len"]))
                jobs.append((r, enc, dec))
            oks = list(ex.map(lambda j: dec_one(j[1], j[2]), jobs))
            for (r, enc, dec), ok in zip(jobs, oks):
                if not ok:
                    die(f"кусок {r['name']} пачки {pid} не расшифровался — подмена или ключ не тот; обучение остановлено")
                b = open(dec, "rb").read()
                if hashlib.sha256(b).hexdigest() != r["sha"] or len(b) != r["bytes"]:
                    die(f"кусок {r['name']} пачки {pid}: sha256 не тот, что в описи набора — подмена; обучение остановлено")
                h = header(b)
                if h is None or h.get("variant") != head["variant"] or bool(h.get("forced")) != (head["forced"] == "1") \
                        or str(h.get("nodes")) != head["nodes"] or "%08x" % int(h.get("eval_id", 0)) not in teachers:
                    die(f"кусок {r['name']} пачки {pid}: заголовок не тот, что в заказе (вариант, правила, узлы или учитель); обучение остановлено")
                dst = os.path.join(out, r["name"])
                if os.path.exists(dst):
                    die(f"кусок {r['name']} уже есть")
                os.replace(dec, dst)
                os.remove(enc)
                n += 1
    os.rmdir(tmp)
    print(f"пачка {pid}: {n} кусков, sha256 и заголовки сошлись")


def seal(lst, out):
    head, rows = load(lst)
    have = sorted(os.listdir(out))
    if have != sorted(r["name"] for r in rows):
        die(f"набор: кусков {len(have)}, в списке {len(rows)} (недостаёт или лишние)")
    h = hashlib.sha256()
    for r in rows:
        x = hashlib.sha256(open(os.path.join(out, r["name"]), "rb").read()).hexdigest()
        if x != r["sha"]:
            die(f"кусок {r['name']}: sha256 изменился")
        h.update(f"{r['name']} {x}\n".encode())
    if h.hexdigest() != head["set"]:
        die("sha256 набора не равен заказанному — обучение остановлено")
    print(h.hexdigest())


def main():
    a = sys.argv[1:]
    if a[:1] == ["check"] and len(a) == 2:
        print(len(load(a[1])[1]))
    elif a[:1] == ["packs"] and len(a) == 2:
        seen = []
        for r in load(a[1])[1]:
            if r["pack"] not in seen:
                seen.append(r["pack"])
        print("\n".join(seen))
    elif a[:1] == ["take"] and len(a) == 5:
        take(*a[1:])
    elif a[:1] == ["seal"] and len(a) == 3:
        seal(*a[1:])
    else:
        die(__doc__, 2)


if __name__ == "__main__":
    main()
