"""Controllo di sola lettura dell'identita' degli OOF dei fit dello scaling 30M/300M (review di Claude del 06/11/2026).

Genera uno script che usa solo la libreria standard e che si esegue su Leonardo senza scrivere nulla:

    python -I <questo file> <checkout del ramo di GPT> > remoto.py
    ssh leonardo 'python3 -' < remoto.py

Lo script remoto non legge mai le predizioni (`*_pred`, solo l'intestazione) e non stampa punteggi. Per ogni fit concluso
ricalcola gli hash di `shard.json`, `shard.oof.npz` e `audit.jsonl` e li confronta con quelli scritti nel report; per ogni
cella e dataset controlla righe uniche e crescenti nel fold, fold riportati, utenti attesi dal piano, un solo fold per utente,
e ricostruisce l'impronta `identity_sha256` (righe, etichette, utenti, fold nell'ordine dei fold) che l'aggregato dovra'
riportare; infine confronta le tre celle fold per fold. La ricostruzione dell'impronta e' convalidata contro `aggregate` del
codice congelato nel file `..._checks.py` accanto a questo.
"""
import json
import pathlib
import sys

REMOTE = r'''
import ast, hashlib, json, os, struct, zipfile
P = json.loads(__PAYLOAD__)

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def member(z, name, payload=True):
    with z.open(name + ".npy") as f:
        magic = f.read(6); major = f.read(1)[0]; f.read(1)
        n = struct.unpack("<H", f.read(2))[0] if major == 1 else struct.unpack("<I", f.read(4))[0]
        h = ast.literal_eval(f.read(n).decode("latin1"))
        assert magic == b"\x93NUMPY" and not h["fortran_order"] and len(h["shape"]) == 1, name
        raw = f.read() if payload else None
    descr, count = h["descr"], h["shape"][0]
    size = 8 if descr == "<i8" else 4 * int(descr[2:]) if descr.startswith("<U") else None
    assert size and (raw is None or len(raw) == size * count), (name, descr, count)
    return descr, size, count, raw

def text(raw):
    return raw.decode("utf-32-le").rstrip("\x00")

cells, problems = {}, []
print("== file dei fit conclusi: sha256 di shard.json | shard.oof.npz | audit.jsonl")
for jid, cell, task, dss, folds, out in P["shards"]:
    sj = out + "/shard.json"
    if not os.path.exists(sj):
        print(jid, "NON CONCLUSO", sorted(os.listdir(out)) if os.path.isdir(out) else "cartella assente")
        continue
    s = json.load(open(sj))
    hs = dict((n, sha_file(out + "/" + n)) for n in ("shard.json", "shard.oof.npz", "audit.jsonl"))
    ok = (hs["shard.oof.npz"] == s["oof_sha256"] and hs["audit.jsonl"] == s["audit_sha256"] and
          s["oof"] == out + "/shard.oof.npz" and s["audit"] == out + "/audit.jsonl" and s["cell"] == cell and s["task"] == task)
    if not ok: problems.append((jid, "hash o percorsi interni"))
    print(jid[8:20], hs["shard.json"], hs["shard.oof.npz"], hs["audit.jsonl"], "coerenti" if ok else "INCOERENTI")
    with zipfile.ZipFile(out + "/shard.oof.npz") as z:
        for ds in dss:
            d_row, _, n, raw_row = member(z, ds + "_row")
            d_fold, _, n2, raw_fold = member(z, ds + "_fold")
            d_y, size_y, n3, raw_y = member(z, ds + "_y")
            d_s, size_s, n4, raw_s = member(z, ds + "_s")
            d_p, _, n5, _ = member(z, ds + "_pred", payload=False)
            assert n == n2 == n3 == n4 == n5 and d_row == d_fold == "<i8" and d_p == d_y, (jid, ds, "tipi o lunghezze")
            rows, fo = struct.unpack("<%dq" % n, raw_row), struct.unpack("<%dq" % n, raw_fold)
            reported = dict((fr["fold"], fr["n_held"]) for fr in s["datasets"][ds]["folds"])
            if sorted(reported) != folds or set(fo) != set(folds): problems.append((jid, ds, "fold riportati"))
            for f in folds:
                ix = [i for i in range(n) if fo[i] == f]
                if len(ix) != reported.get(f): problems.append((jid, ds, f, "n_held"))
                if any(rows[ix[k]] >= rows[ix[k+1]] for k in range(len(ix) - 1)): problems.append((jid, ds, f, "righe non crescenti"))
                slot = cells.setdefault((cell, ds), {})
                if f in slot: problems.append((jid, ds, f, "fold doppio"))
                slot[f] = {"rows": [rows[i] for i in ix], "ydescr": d_y, "ysize": size_y, "y": [raw_y[i*size_y:(i+1)*size_y] for i in ix],
                           "sdescr": d_s, "ssize": size_s, "s": [raw_s[i*size_s:(i+1)*size_s] for i in ix]}

def widen(items, size, target):
    return items if size == target else [x + b"\x00" * (target - size) for x in items]

print("== identita' per cella e dataset (righe, etichette, utenti, fold; predizioni non lette)")
per_fold, identity = {}, {}
for (cell, ds), slot in sorted(cells.items()):
    present = sorted(slot)
    all_rows = [r for f in present for r in slot[f]["rows"]]
    who = {}
    for f in present:
        for u in set(text(x) for x in slot[f]["s"]):
            who.setdefault(u, set()).add(f)
    exclusive = all(len(v) == 1 for v in who.values())
    ys = max(slot[f]["ysize"] for f in present); ss = max(slot[f]["ssize"] for f in present)
    ydescr = slot[present[0]]["ydescr"] if ys == 8 and slot[present[0]]["ydescr"] == "<i8" else "<U%d" % (ys // 4)
    sdescr = "<U%d" % (ss // 4)
    for f in present:
        d = slot[f]
        h = hashlib.sha256()
        for part in (struct.pack("<%dq" % len(d["rows"]), *d["rows"]), ydescr.encode(), b"".join(widen(d["y"], d["ysize"], ys)),
                     sdescr.encode(), b"".join(widen(d["s"], d["ssize"], ss))):
            h.update(part); h.update(b"|")
        per_fold[(cell, ds, f)] = h.hexdigest()
    labels = set()
    for f in present:
        labels |= set(slot[f]["y"])
    shown = sorted(text(x) for x in labels) if ydescr.startswith("<U") else "%d classi intere" % len(labels)
    line = [cell, ds, "fold", present, "righe", len(all_rows), "uniche", len(set(all_rows)) == len(all_rows),
            "utenti", len(who), "attesi", len(P["subjects"][ds]), "insieme atteso", sorted(who) == P["subjects"][ds] if present == [0, 1, 2, 3] else "parziale",
            "un fold per utente", exclusive, "y", ydescr, "s", sdescr, "etichette", shown]
    if present == [0, 1, 2, 3]:
        n = len(all_rows)
        h = hashlib.sha256()
        for descr, raw in (("<i8", struct.pack("<%dq" % n, *all_rows)),
                           (ydescr, b"".join(b"".join(widen(slot[f]["y"], slot[f]["ysize"], ys)) for f in present)),
                           (sdescr, b"".join(b"".join(widen(slot[f]["s"], slot[f]["ssize"], ss)) for f in present)),
                           ("<i8", struct.pack("<%dq" % n, *[f for f in present for _ in slot[f]["rows"]]))):
            h.update(json.dumps([descr, [n]]).encode()); h.update(raw); h.update(b"\0")
        identity[(cell, ds)] = h.hexdigest()
        line += ["identity_sha256 atteso", identity[(cell, ds)]]
    print(*line)
print("== appaiamento fra celle (stesse righe, etichette e utenti per fold)")
for ds in sorted(P["subjects"]):
    names = sorted(c for c, d in cells if d == ds)
    for f in range(4):
        got = dict((c, per_fold.get((c, ds, f))) for c in names)
        have = dict((c, v) for c, v in got.items() if v)
        print(ds, "fold", f, "celle con il fold:", len(have), "| identiche:", len(set(have.values())) == 1 if have else None,
              "| mancanti:", sorted(c for c, v in got.items() if not v) + sorted(set(["T100-G-30M-natural", "T100-G-300M-natural", "T100-G-300M-matched"]) - set(names)))
    ids = set(identity.get((c, ds)) for c in names if (c, ds) in identity)
    print(ds, "identity_sha256 uguale fra le celle complete:", len(ids) == 1, "celle complete:", sum((c, ds) in identity for c in names))
print("== problemi:", problems if problems else "nessuno")
'''


def payload(checkout):
    results = pathlib.Path(checkout) / "results/infra/scaling_300m_20261104"
    jobs = json.loads((results / "cpu_fit_release_20261105/submission_plan.json").read_text())["jobs"][1:]
    plan = json.loads((results / "cpu_readers_20261105/probe_plan.json").read_text())
    return {"shards": [[j["id"], j["cell"], j["task"], j["datasets"], j["folds"], j["out"]] for j in jobs],
            "subjects": {ds: v["subjects"] for ds, v in plan["datasets"].items()}}


def script(value):
    return REMOTE.replace("__PAYLOAD__", repr(json.dumps(value)))


if __name__ == "__main__":
    print(script(payload(sys.argv[1])))
