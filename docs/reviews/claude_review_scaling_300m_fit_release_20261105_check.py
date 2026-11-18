"""Review 9: controlli indipendenti sul rilascio dei fit (solo lettura, nessun import dal repo)."""
import hashlib, itertools, json, math, pathlib, re, sys

co = pathlib.Path(sys.argv[1])            # checkout della punta di GPT
remote_log = pathlib.Path(sys.argv[2])    # lettura di Leonardo con gli sha256
R = co / "results/infra/scaling_300m_20261104"
F = R / "cpu_fit_release_20261105"
CELLS = ("T100-G-30M-natural", "T100-G-300M-natural", "T100-G-300M-matched")
TASKS = {"p1": ("ninapro_db2", "ninapro_db3", "ninapro_db6"), "p2": ("epn612", "uci_emg")}
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

# 1. sha256 remoti letti da me
remote = {}
for line in remote_log.read_text().splitlines():
    m = re.fullmatch(r"([0-9a-f]{64})\s+(/leonardo_work/\S+)", line.strip())
    if m:
        remote[m.group(2)] = m.group(1)
res = json.loads((F / "probe_fit_resources.json").read_text())
ok = 0
for c in CELLS:
    for ds in sum(TASKS.values(), ()):
        p = res["benchmarks"][c][ds]
        ok += remote.get(p["path"]) == p["sha256"]
print(f"1. pin dei benchmark nel piano risorse uguali agli sha256 remoti letti da me: {ok}/15")

# 2. copie locali dei 15 report (pubblicate da GPT) con lo stesso sha256 dei pin
acc = json.loads((R / "cpu_recovery_20261105/accepted_benchmarks.json").read_text())
print("2. manifest: matrix_complete", acc["matrix_complete"], "report", len(acc["reports"]), "sha manifest", sha(R / "cpu_recovery_20261105/accepted_benchmarks.json")[:12])
reports = {}
for e in acc["reports"]:
    lp = co / e["local_report"]["path"]
    assert sha(lp) == e["report"]["sha256"] == res["benchmarks"][e["cell"]][e["dataset"]]["sha256"], (e["cell"], e["dataset"])
    assert e["report"]["remote_path"] == res["benchmarks"][e["cell"]][e["dataset"]]["path"]
    reports[e["cell"], e["dataset"]] = json.loads(lp.read_bytes())
print("   copie locali = pin = percorsi remoti:", len(reports), "/ 15")
for e in acc["reports"]:
    if e["cell"] == CELLS[1] and e["dataset"] in TASKS["p2"]:
        print("   nuovo:", e["dataset"], {k: e.get(k) for k in ("parent_job", "parent_state", "parent_exit_code", "parent_success_required")},
              "chiavi extra:", sorted(set(e) - {"cell", "dataset", "report", "local_report", "parent_job", "parent_state", "parent_exit_code",
                                                 "parent_accounting", "raw_sacct", "submitter_receipt", "complete_log_evidence",
                                                 "reuse_accepted_by_claude_review7", "parent_success_required",
                                                 "timeout_parent_not_promoted_to_success", "required_numeric_sha256"}))

# 3. qualita' dei 90 fit di benchmark
nfit = nconv = 0
for (c, ds), r in sorted(reports.items()):
    for f in r["fits"]:
        nfit += 1
        nconv += bool(f.get("converged_by_criterion")) and not f.get("reached_limit") and f.get("exception") is None
    assert r["code_sha"].startswith("aef665ba") and r["numeric_sha256"].startswith("1ac8481e") and r["complete"] is True
print(f"3. fit di benchmark: {nfit}, convergenti senza limite ne' eccezioni: {nconv}; codice e contesto numerico uguali in tutti")
for ds in TASKS["p2"]:
    r = reports[CELLS[1], ds]
    print(f"   {ds} 300M naturale:", [(f["C"], round(f["elapsed_s"], 1), f.get("n_iter")) for f in r["fits"]],
          "| somma", round(sum(f["elapsed_s"] for f in r["fits"]), 1), "max", round(max(f["elapsed_s"] for f in r["fits"]), 1),
          "| picco RSS GiB", round(r["peak_rss_bytes"] / 2**30, 2))

# 4. ricalcolo della formula congelata per ogni shard pubblicato
def needed(cell, dss, folds):
    n = 120.0
    for ds in dss:
        r = reports[cell, ds]
        t = [f["elapsed_s"] for f in r["fits"]]
        cnt = r["fold_train_counts"]
        n += 1.5 * (sum(t) + max(t)) * sum(max(1.0, cnt[f] / cnt[0]) for f in folds)
    return n

tot = 0
cover = []
worst = None
for s in res["shards"]:
    n = needed(s["cell"], s["datasets"], s["folds"])
    w = math.ceil(n / 60) * 60
    assert abs(n - s["estimated_seconds_with_margin"]) < 1e-6 and w == s["walltime_seconds"] <= 10800, s["shard_name"]
    assert s["cpus"] == 8 and s["memory_gib"] == 30
    rss = max(reports[s["cell"], d]["peak_rss_bytes"] for d in s["datasets"])
    assert rss * 1.3 <= 30 * 2**30
    tot += w
    cover += [(s["cell"], d, f) for d in s["datasets"] for f in s["folds"]]
    if worst is None or w - n < worst[0]:
        worst = (w - n, s["shard_name"], n, w)
exp = set(itertools.product(CELLS, sum(TASKS.values(), ()), range(4)))
print(f"4. shard {len(res['shards'])}: formula ricalcolata uguale, walltime = arrotondamento al minuto, tutti <= 10800 s;")
print(f"   copertura esatta {len(cover)}/60: {len(cover) == len(set(cover)) and set(cover) == exp}; totale {tot} s = {tot*8/3600:.6f} core-ore"
      f" = {tot/3600:.4f} h seriali; con margine {(tot + 180*len(res['shards']))*8/3600:.6f}")
print(f"   shard piu' stretto: {worst[1]} formula {worst[2]:.1f} s, allocati {worst[3]} s (scarto {worst[0]:.1f} s)")

# 5. numero minimo di shard per gruppo (forza bruta sulle partizioni in rettangoli)
def min_cover(cell, task):
    dss = TASKS[task]
    rects = []
    for k in range(1, len(dss) + 1):
        for sel in itertools.combinations(dss, k):
            for j in range(1, 5):
                for fo in itertools.combinations(range(4), j):
                    w = math.ceil(needed(cell, sel, fo) / 60) * 60
                    if w <= 10800:
                        mask = sum(1 << (dss.index(d) * 4 + f) for d in sel for f in fo)
                        rects.append((mask, w))
    full = (1 << (len(dss) * 4)) - 1
    best = {0: (0, 0)}
    frontier = {0}
    while frontier:
        nxt = set()
        for u in frontier:
            low = (~u & full) & -(~u & full)
            for mask, w in rects:
                if mask & low and not mask & u:
                    v = u | mask
                    cand = (best[u][0] + 1, best[u][1] + w)
                    if v not in best or cand < best[v]:
                        best[v] = cand
                        nxt.add(v)
        frontier = nxt
    return best[full]

mins = {(c, t): min_cover(c, t) for c in CELLS for t in TASKS}
print("5. minimo per gruppo (shard, secondi):", {f"{c[7:]}/{t}": v for (c, t), v in mins.items()},
      "| totale", sum(v[0] for v in mins.values()), "shard,", sum(v[1] for v in mins.values()), "s")

# 6. ricevute degli invii
print("6. ricevute:")
for p in sorted((F / "submissions").glob("*.json")) + sorted((R / "cpu_recovery_20261105/submissions").glob("*.json")):
    d = json.loads(p.read_text())
    sb = d["sbatch"]
    dep = [x for x in sb if x.startswith("--dependency")]
    t = [x for x in sb if x.startswith("--time=")]
    print(f"   {d['id']:40s} {d['stato']:10s} job {d.get('job_slurm')} stdin_sha {str(d.get('script_stdin_sha256'))[:8]} "
          f"script {pathlib.PurePosixPath(d.get('script_stdin') or '-').name} /dev/stdin={'/dev/stdin' in sb} {t} {dep}")

# 7. comandi del piano uguali alle righe sbatch delle ricevute
plan = json.loads((F / "submission_plan.json").read_text())
same = 0
for j in plan["jobs"]:
    rp = F / "submissions" / f"{j['id']}.json"
    if not rp.exists():
        continue
    d = json.loads(rp.read_text())
    a = j["argv"]
    tail = a[a.index("--") + 1:]
    got = [x for x in d["sbatch"][4:] if not x.startswith("--dependency")]
    same += got == tail
print(f"7. righe sbatch delle ricevute uguali agli argv pubblicati: {same}/{len(list((F / 'submissions').glob('SCALING-p*.json')))}")
