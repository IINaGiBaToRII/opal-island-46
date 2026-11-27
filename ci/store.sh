#!/usr/bin/env bash
# Облачная пачка круга самоигры: вызывается из ci/round.sh merge, если в манифесте cloud=1.
#   ci/store.sh RESULT_DIR PACK_DIR SHARD_DIR...
# Каждый кусок SHARD_DIR/chunks/games-*.dgz шифруется отдельно (как ci/lib.sh enc, ключ ROUNDS_KEY; параллельно по числу
# ядер — PBKDF2 на кусок) в <sha256 открытого>.enc; пачка PACK_DIR/pack — tar без сжатия (ustar) из этих файлов по имени
# (артефакт без архива, archive: false, 90 дней): Mac берёт куски по смещению (Range) и сверяет sha256 после
# расшифровки (AES-CBC целостность не проверяет).
# Опись RESULT_DIR/store.json (в итоге result) и её зашифрованная копия PACK_DIR/store.<kfp>.enc (артефакт рядом с пачкой —
# восстановление описи на Mac): круг, версия кода, параметры (вариант, учитель, узлы, extra), осколки (зерно, тождество),
# куски (sha256, байты, смещение и длина в пачке), sha256 пачки; mac = HMAC-SHA256(ROUNDS_KEY, каноническая опись без mac).
set -euo pipefail
cd "$(dirname "$0")/.."
. ci/lib.sh
need_key
[ $# -ge 3 ] || die "ci/store.sh RESULT_DIR PACK_DIR SHARD_DIR..."
RES=$1; PK=$2; shift 2
KFP=$(kfp)
mkdir -p "$RES" "$PK"
export ROUNDS_KEY
python3 - "$PK" "$RES/store.json" "${ROUND:-}" "$(tr -d '[:space:]' < ci/EXPORT_ID 2>/dev/null || true)" "$KFP" "$@" <<'PY'
import concurrent.futures, hashlib, hmac, json, os, subprocess, sys, tarfile, tempfile, time
pk, out, rnd, export_id, kfp = sys.argv[1:6]
dirs = sys.argv[6:]
t0 = time.time()
shards, rows = {}, []
for d in dirs:
    mp = os.path.join(d, "meta.json")
    if not os.path.isfile(mp):
        continue
    meta = json.load(open(mp, encoding="utf-8"))
    k = int(meta["shard"])
    cp = os.path.join(d, "canary.txt")
    shards[k] = {"shard": k, "seed": int(meta.get("seed", 0)), "secs": int(round(float(meta.get("secs", 0)))),
                 "canary": open(cp, encoding="utf-8").read() if os.path.exists(cp) else ""}
    cd = os.path.join(d, "chunks")
    for n in sorted(os.listdir(cd)) if os.path.isdir(cd) else []:
        if n.startswith("games-") and n.endswith(".dgz"):
            p = os.path.join(cd, n)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            rows.append((k, n, p, h, os.path.getsize(p)))
if not rows:
    print("облачная пачка: кусков нет")
    sys.exit(0)
tmp = tempfile.mkdtemp()
enc = {}
for k, n, p, h, b in rows:
    enc.setdefault(h, p)
def one(item):
    h, p = item
    dst = os.path.join(tmp, h + ".enc")
    subprocess.run(["openssl", "enc", "-e", "-aes-256-cbc", "-pbkdf2", "-iter", "200000", "-md", "sha256", "-salt",
                    "-in", p, "-out", dst, "-pass", "env:ROUNDS_KEY"], check=True, stderr=subprocess.DEVNULL)
    return h
with concurrent.futures.ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 2)) as ex:
    list(ex.map(one, sorted(enc.items())))
t1 = time.time()
part = os.path.join(pk, "pack.part")
with tarfile.open(part, "w", format=tarfile.USTAR_FORMAT) as t:
    for h in sorted(enc):
        ti = t.gettarinfo(os.path.join(tmp, h + ".enc"), arcname=h + ".enc")
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mtime = 0
        ti.mode = 0o644
        with open(os.path.join(tmp, h + ".enc"), "rb") as f:
            t.addfile(ti, f)
        os.remove(os.path.join(tmp, h + ".enc"))
os.rmdir(tmp)
pack = os.path.join(pk, "pack")
os.replace(part, pack)
off = {}
with tarfile.open(pack) as t:
    for m in t.getmembers():
        off[m.name] = (m.offset_data, m.size)
chunks = [{"shard": k, "name": n, "sha256": h, "bytes": b, "off": off[h + ".enc"][0], "len": off[h + ".enc"][1]} for k, n, p, h, b in rows]
chunks.sort(key=lambda c: (c["shard"], c["name"]))
hp = hashlib.sha256()
with open(pack, "rb") as f:
    for blk in iter(lambda: f.read(1 << 20), b""):
        hp.update(blk)
extra = os.environ.get("IN_EXTRA", "")
params = {"variant": os.environ.get("IN_VARIANT", ""), "eval_a": os.environ.get("IN_EVAL_A", ""), "nodes_a": os.environ.get("IN_NODES_A", ""),
          "extra": extra, "free": "1" if " --free " in f" {extra} " else "0"}
s = {"v": 1, "round": rnd, "export_id": export_id, "kfp": kfp, "params": params, "shards": [shards[k] for k in sorted(shards)], "chunks": chunks,
     "pack": {"name": "pack", "sha256": hp.hexdigest(), "bytes": os.path.getsize(pack)}}
canon = json.dumps(s, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
s["mac"] = hmac.new(os.environ["ROUNDS_KEY"].encode("utf-8"), canon, hashlib.sha256).hexdigest()
json.dump(s, open(out, "w", encoding="utf-8"), ensure_ascii=False, sort_keys=True)
print(f"облачная пачка: {len(chunks)} кусков, {os.path.getsize(pack)} байт; шифрование {t1 - t0:.0f} с, всего {time.time() - t0:.0f} с")
PY
[ -f "$RES/store.json" ] || exit 0
enc "$RES/store.json" "$PK/store.$KFP.enc.part" || die "шифрование описи не удалось"
mv -f "$PK/store.$KFP.enc.part" "$PK/store.$KFP.enc"
