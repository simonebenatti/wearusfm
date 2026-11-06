"""Controlli sui metadati reali gia' raccolti da GPT (report della ridge e shard) contro il codice congelato dello scaling.

Review di Claude del 06/11/2026. Non stampa score: ne' bacc, ne' mse, ne' analysis o paired. Stampa gli hash `sources` che il
manifest del confronto dovra' dichiarare, il runtime registrato dalla ridge, il contesto numerico delle sonde, l'esito di
`runtime_coherence` e di `validate_pose_selection` sui report reali e i campi che `read_completed` richiede agli shard.

Uso: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<checkout>/src \
  ~/.venvs/wearusfm-tok/bin/python <questo file> <checkout del ramo di GPT>
"""
import json, pathlib, sys
co = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(co / "scripts"))
import scaling_compare as M
C, P = M.C, M.P
R = co / "results/infra/scaling_300m_20261104"
SUB = R / "cpu_fit_release_20261105/submissions/collected_jobs"
CODE = "aef665ba7ea66fc80291846487907bb3ef9107df"
plan = json.loads((R / "cpu_readers_20261105/probe_plan.json").read_text())
pose_plan = json.loads((R / "cpu_readers_20261105/pose_plan.json").read_text())
print("piani locali: sonde", C.sha(R / "cpu_readers_20261105/probe_plan.json")[:12], "posa", C.sha(R / "cpu_readers_20261105/pose_plan.json")[:12])

# 1. sorgenti che il manifest del confronto dovra' dichiarare
print("1. sources attesi nel manifest del confronto (file identici ad aef665b):")
for k, v in sorted(M.source_hashes().items()):
    print("   ", k, v)

# 2. ridge: runtime registrato
ridge = json.loads((SUB / "SCALING-pose-ridge-20261105/report.json").read_text())
rt = ridge["runtime"]
print("2. runtime della ridge: code", rt["code_sha"][:10], "versions", rt["versions"], "threads", rt["threads"], "torch_threads", rt["torch_threads"])
for t in rt["threadpools"]:
    print("    pool posa:", t)
print("   sources della posa uguali a P.source_hashes():", rt["sources"] == P.source_hashes())

# 3. shard reali: contesto numerico e coerenza col runtime della posa
shards = {}
for d in sorted(SUB.glob("SCALING-probe-fit-*")):
    shards[d.name] = json.loads((d / "shard.json").read_text())
first = next(iter(shards.values()))
nc = first["numeric_context"]
print("3. contesto numerico delle sonde:", {k: nc[k] for k in ("python", "numpy", "scipy", "sklearn")})
for t in nc["threadpools"]:
    print("    pool sonde:", t)
print("   numeric_sha256 distinti fra gli shard raccolti:", len({s["numeric_sha256"] for s in shards.values()}), "su", len(shards))
try:
    out = M.runtime_coherence(list(shards.values()), rt, CODE)
    print("   runtime_coherence(shard reali, runtime ridge): PASSA; probe_numeric_sha256", out["probe_numeric_sha256"][:12])
except Exception as e:
    print("   runtime_coherence: RIFIUTA ->", e)

# 4. campi che read_completed richiede agli shard, verificabili senza cache
ok = 0
for name, s in shards.items():
    recs = [json.loads(l) for l in (SUB / name / "audit.jsonl").read_text().splitlines()]
    good = (s["complete"] is True and s["kind"] == "shard" and s["protocol"] == C.PROTOCOL and s["plan_content_sha256"] == C.jhash(plan)
            and s["recipe"] == C.recipe() and s["step"] == C.STEP and (s["arm"], s["readout"]) == C.CELLS[s["cell"]]
            and s["projection_sha256"] == plan["projection"]["sha256"] and C.jhash(s["numeric_context"]) == s["numeric_sha256"]
            and C.audit_summary(recs) == s["audit_summary"] and C.sha(SUB / name / "audit.jsonl") == s["audit_sha256"]
            and s["code_sha"] == CODE and s["plan_sha256"] == C.sha(R / "cpu_readers_20261105/probe_plan.json"))
    ok += good
    nds = {ds: [(f["fold"], f["n_train"], f["n_val"], f["n_held"]) for f in v["folds"]] for ds, v in s["datasets"].items()}
    classes = {ds: len(next(r for r in recs if r["dataset"] == ds)["classes"]) for ds in s["datasets"]}
    ctype = {ds: type(next(r for r in recs if r["dataset"] == ds)["classes"][0]).__name__ for ds in s["datasets"]}
    el = sum(r["elapsed_s"] for r in recs)
    print(f"   {name}: {s['cell']} {s['task']} campi ok={good} fit={len(recs)} convergenti={s['audit_summary']['converged_by_criterion']} "
          f"somma tempi fit={el:.0f}s classi={classes} tipo etichette={ctype}")
    for ds, v in nds.items():
        print(f"      {ds}: (fold, n_train, n_val, n_held) {v} -> OOF {sum(x[3] for x in v)}")
print("4. shard raccolti con i campi di read_completed corretti:", ok, "su", len(shards))

# 5. soggetti per dataset nel piano
print("5. soggetti per dataset nel piano:", {ds: len(v["subjects"]) for ds, v in plan["datasets"].items()})

# 6. selezione dell'alpha della posa: il comparatore la ricalcola dalla griglia registrata (nessun valore stampato)
for name, cell in ridge["cells"].items():
    try:
        M.validate_pose_selection(cell["fit"]["selection"], pose_plan["fold_users"])
        print("6.", name, "validate_pose_selection: PASSA; punti di griglia", len(cell["fit"]["selection"]["grid"]),
              "boundary_unresolved", cell["fit"]["selection"]["boundary_unresolved"])
    except Exception as e:
        print("6.", name, "validate_pose_selection: RIFIUTA in locale ->", type(e).__name__, str(e)[:80])
