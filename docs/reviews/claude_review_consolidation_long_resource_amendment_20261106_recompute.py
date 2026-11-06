#!/usr/bin/env python3
"""Reviewer's recomputation of GPT's resource amendment proposal (long evaluation, line 2).

Fold costs come from the frozen runtime gate (`CR.resource_gate`, benchmark-phase code) applied to the 15 collected
receipts; the amended split, walltimes and budget are rebuilt here from the table of the proposal document, not from
GPT's JSON, and only then compared with it. No cluster access, no fit, no score.
Run from a checkout of the execution branch at 2a20316 (frozen source 1ca7aead...) with the project venv, one numeric
thread and PYTHONPATH=src:scripts:
    <this script> <GPT options.json> <ledger.tsv> <output json>
GPT's JSON is results/infra/consolidation_long_20261105/resource_amendment_proposal/options.json at commit 4bb39d2
(`git show 4bb39d2:<path> > file`); the ledger is ~/wearusfm_local/jobs/ledger.tsv. Output of the run of 06/11/2026:
claude_review_consolidation_long_resource_amendment_20261106_recompute.json (empty `problems`)."""
import csv, hashlib, io, json, math, subprocess, sys
from pathlib import Path
import consolidation_long_common as C
import consolidation_long_classification_runtime as CR
import consolidation_long_pose_runtime as PR

options_path, ledger_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
root = Path("results/infra/consolidation_long_20261105")
plan = json.loads((root / "plan.json").read_text())
launch = json.loads((root / "launch_plan.json").read_text())
gpt = json.loads(options_path.read_text())
problems, notes = [], []


def need(cond, text):
    if not cond:
        problems.append(text)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 1. Frozen gate on my receipts.
reports = {}
for row in launch["jobs"]:
    if row["stage"] == "benchmark-classification":
        r = json.loads((root / "execution/receipts" / (row["id"] + ".json")).read_text())
        reports.setdefault(r["dataset"], {})[r["view"]] = r
gate = CR.resource_gate(reports)
need(gate["all_passed"] is False and len(gate["shards"]) == 48 and len(gate["cells"]) == 40, "frozen gate is not the negative 48/40 one")
failed = [k for k, s in gate["shards"].items() if not s["passed"]]
failed_cells = [k for k, c in gate["cells"].items() if not c["passed"]]
need(len(failed) == 15 and len(failed_cells) == 10, "frozen gate: failed shards/cells differ from 15/10")

# 2. Amended split, from the table of the proposal (section 2).
DATASETS, VIEWS, DOSES = CR.M.DATASETS, CR.M.VIEWS, CR.M.DOSES
SINGLES, PAIRS, WHOLE = [[0], [1], [2], [3]], [[0, 1], [2, 3]], [[0, 1, 2, 3]]


def shapes(ds, view, db3="singles"):
    if ds == "ninapro_db2":
        return SINGLES if view in ("pre", "pool") else PAIRS
    if ds == "ninapro_db3" and view == "pre":
        return SINGLES if db3 == "singles" else PAIRS
    if ds == "ninapro_db6" and view == "pool":
        return PAIRS
    return WHOLE


policy = CR.BENCHMARK_POLICY
limit_memory = policy["memory_gib"] * 1024**3 * policy["memory_fraction"]


def matrix(db3):
    jobs = []
    for step in DOSES:
        for view in VIEWS:
            if step == 19618 and view == "pool":
                continue
            for ds in DATASETS:
                cell = gate["cells"][f"{step}/{view}/{ds}"]
                bench = reports[ds][view]["benchmark"]
                overhead = policy["fixed_seconds_per_job"] + 2 * (bench["load_s"] + bench["scale_s"])
                for folds in shapes(ds, view, db3):
                    p = sum(cell["projected_fit_seconds_by_fold"][f] for f in folds) + overhead
                    w = 900 * math.ceil((p + 600) / 900)
                    tag = "".join(map(str, folds))
                    jobs.append({"id": f"LONG-FIT-G{step}-{view}-{ds}-F{tag}-20261105-A1", "step": step, "view": view,
                                 "dataset": ds, "folds": folds, "P": p, "W": w, "margin": w - 600 - p,
                                 "memory": cell["projected_peak_rss_bytes"],
                                 "out": f"{plan['output_root']}/fits_a1/classification/{step}/{view}/{ds}/F{tag}"})
    return jobs


mine = matrix("singles")
need(len(mine) == 69, f"amended matrix has {len(mine)} jobs, not 69")
need(len({j["id"] for j in mine}) == 69 and len({j["out"] for j in mine}) == 69, "duplicate amended IDs or outputs")
cells = {}
for j in mine:
    cells.setdefault((j["step"], j["view"], j["dataset"]), []).extend(j["folds"])
need(len(cells) == 40 and all(sorted(v) == [0, 1, 2, 3] for v in cells.values()), "coverage is not 40 cells x 4 folds exactly once")
need(set(cells) == {(c["step"], c["view"], c["dataset"]) for c in gate["cells"].values()}, "amended cells differ from the frozen 40")
need(all(j["P"] <= j["W"] - 600 and j["W"] <= 14400 and j["W"] % 900 == 0 for j in mine), "a shard fails P <= W-600 or W <= 14400")
need(all(j["memory"] <= limit_memory for j in mine), "a shard fails the memory limit")
need(all(j["P"] <= 13800 for j in mine), "a shard would fail even the original 13800 s limit")
# An unchanged shard must keep the frozen projection.
for j in mine:
    key = CR.shard_key(j["step"], j["view"], j["dataset"], j["folds"])
    if key in gate["shards"]:
        need(abs(gate["shards"][key]["projected_seconds"] - j["P"]) < 1e-9, key + ": unchanged shard projection differs")
unchanged = sum(CR.shard_key(j["step"], j["view"], j["dataset"], j["folds"]) in gate["shards"] for j in mine)
by_case = {}
for j in mine:
    by_case.setdefault((j["dataset"], j["view"]), []).append(j)
counts = {f"{ds}/{view}": len(v) for (ds, view), v in sorted(by_case.items())}

# 3. Comparison with GPT's JSON (recommended option).
theirs = {j["id"]: j for j in gpt["options"]["db3_singles"]["jobs"]}
need(set(theirs) == {j["id"] for j in mine}, "amended IDs differ from GPT's 69")
worst = 0.0
for j in mine:
    t = theirs.get(j["id"])
    if t is None:
        continue
    worst = max(worst, abs(t["projection_seconds"] - j["P"]))
    need(t["folds"] == j["folds"] and t["step"] == j["step"] and t["view"] == j["view"] and t["dataset"] == j["dataset"]
         and t["out"] == j["out"] and t["measured_walltime_seconds"] == j["W"]
         and (t["cpus"], t["memory_gib"], t["gpus"]) == (8, 30, 0), j["id"] + ": differs from GPT's row")
need(worst < 1e-6, f"projection differs from GPT's by up to {worst} s")
need(gpt["recommended_option"] == "db3_singles/rounded15min" and gpt["git_commit"] == "2a20316a0bf77c93e354d4ce3f5acee94db459df",
     "GPT's recommended option or source commit differs")

# 4. Old and new identifiers, ledger, launch plan.
old_ids = {j["id"] for j in launch["jobs"] if j["stage"] == "fit-classification"}
need(len(old_ids) == 48 and not old_ids & set(theirs), "old fit IDs collide with the amended ones")
need(all(j["slurm_job_id"] is None for j in launch["jobs"] if j["phase"] in ("fit", "aggregate")), "a fit/aggregate row already has a job")
mapping = {m["old_unsubmitted_id"]: m["proposed_successor_ids"] for m in gpt["options"]["db3_singles"]["old_to_new_fit_ids"]}
need(set(mapping) == old_ids and sorted(s for v in mapping.values() for s in v) == sorted(theirs), "old-to-new mapping is not a partition of the 69")
ledger_text = ledger_path.read_text()
ledger_ids = {line.split("\t")[0] for line in ledger_text.splitlines()[1:] if line.strip()}
hits = sorted(i for i in ledger_ids if "LONG-FIT" in i or "LONG-checkout-fit" in i or "LONG-COMPARE" in i or "LONG-SUMMARY" in i
              or "LONG-checkout-aggregate" in i)
need(not hits, "ledger already holds fit/aggregate identifiers: " + ", ".join(hits))
need(not any(i in ledger_text for i in theirs), "an amended ID already appears in the ledger")

# 5. Budget.
core = lambda cpus, seconds: cpus * seconds / 3600
fit_alloc = sum(core(8, j["W"]) for j in mine)
fit_projection = sum(core(8, j["P"]) for j in mine)
later = [j for j in launch["jobs"] if j["stage"] in ("fit-pose", "compare-classification", "compare-pose", "summary")]
checkout = launch["caps"]["checkout_each"]
other = sum(core(j["allocation"]["cpus"], j["allocation"]["seconds"]) for j in later) + 2 * core(checkout["cpus_accounting_cap"], checkout["seconds"])
spent_core = spent_gpu = 0.0
rows_total = 0
for phase in ("extract", "benchmark"):
    rows = list(csv.DictReader(io.StringIO((root / f"execution/accounting/sacct_{phase}.tsv").read_text()), delimiter="\t"))
    for r in rows:
        need(r["State"] == "COMPLETED" and r["ExitCode"] == "0:0", f"{phase}: a job is not COMPLETED 0:0")
        tres = dict(v.split("=", 1) for v in r["AllocTRES"].split(","))
        spent_core += int(r["ElapsedRaw"]) * int(r["AllocCPUS"]) / 3600
        spent_gpu += int(r["ElapsedRaw"]) * int(tres.get("gres/gpu", "0")) / 3600
    rows_total += len(rows)
done_caps = sum(core(j["allocation"]["cpus"], j["allocation"]["seconds"]) for j in launch["jobs"] if j["phase"] in ("extract", "benchmark")) \
    + 2 * core(checkout["cpus_accounting_cap"], checkout["seconds"])
cap = launch["caps"]["allocated_core_hours"]
original_fit_cap = sum(core(j["allocation"]["cpus"], j["allocation"]["seconds"]) for j in launch["jobs"] if j["stage"] == "fit-classification")
budget = {"fit_projection_core_hours": fit_projection, "fit_allocation_core_hours": fit_alloc, "other_future_core_hours": other,
          "future_max_core_hours": fit_alloc + other, "spent_core_hours": spent_core, "spent_gpu_hours": spent_gpu,
          "spent_plus_future_core_hours": spent_core + fit_alloc + other, "concluded_phase_caps_core_hours": done_caps,
          "conservative_total_core_hours": done_caps + fit_alloc + other, "cap_core_hours": cap,
          "margin_conservative_core_hours": cap - done_caps - fit_alloc - other,
          "original_fit_cap_core_hours": original_fit_cap, "original_total_core_hours": done_caps + original_fit_cap + other,
          "uniform_4h_total_core_hours": done_caps + 69 * 32 + other, "completed_allocations": rows_total,
          "compute_jobs": len(launch["jobs"]) - 48 + 69, "allocations_with_checkouts": len(launch["jobs"]) - 48 + 69 + 4,
          "allocations_left": 69 + len(later) + 2}
b = gpt["options"]["db3_singles"]["budget"]["rounded15min"]
need(abs(b["classification_reserved_core_hours"] - fit_alloc) < 1e-9 and abs(b["future_total_reserved_core_hours"] - fit_alloc - other) < 1e-9
     and abs(b["past_actual_plus_future_cap_core_hours"] - budget["spent_plus_future_core_hours"]) < 1e-9
     and abs(b["whole_campaign_using_past_original_reservations_core_hours"] - budget["conservative_total_core_hours"]) < 1e-9
     and abs(gpt["options"]["db3_singles"]["projected_classification_core_hours"] - fit_projection) < 1e-9, "budget differs from GPT's")
need(budget["conservative_total_core_hours"] <= cap and abs(budget["original_total_core_hours"] - cap) < 1e-9, "cap arithmetic differs")
need(rows_total == 30 and abs(spent_gpu - 3.373611) < 1e-6 and abs(spent_core - 42.218889) < 1e-6, "consumption differs from the archived accounting")

# 6. Alternative kept in the JSON: DB3 pre by pairs.
pairs = matrix("pairs")
db3_pairs = sorted({round(13800 - j["P"], 1) for j in pairs if j["dataset"] == "ninapro_db3" and j["view"] == "pre"})
budget["db3_pairs_jobs"] = len(pairs)
budget["db3_pairs_min_margin_to_13800"] = db3_pairs[0]
budget["db3_pairs_fit_allocation_core_hours"] = sum(core(8, j["W"]) for j in pairs)

# 7. Pose admission from the collected receipt.
pose = json.loads((root / "execution/receipts/LONG-BENCH-pose-20261105.json").read_text())
projection = PR.project_benchmark(pose["timings_seconds"], pose["peak_rss_bytes"])
need(projection == pose["projection"] and projection["admitted"] is True, "pose projection differs or is not admitted")

# 8. The 28 input pins against my audits and the files in this worktree.
pins = gpt["proposed_original_source_bridge_28_input_pins"]
audit = {}
for phase in ("extract", "benchmark"):
    audit.update(json.loads((root / f"execution/audit/receipts_{phase}.json").read_text())["receipts"])
need(len(pins) == 28 and set(pins) == set(audit), "the 28 pinned inputs differ from my two audits")
sources, recipes, codes, plans, kinds = set(), set(), {}, {}, {}
for ident, pin in pins.items():
    path = Path(pin["git_path"])
    need(path == root / "execution/receipts" / (ident + ".json") and path.is_file(), ident + ": path differs")
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == pin["sha256"] == audit[ident]["sha256"] and len(raw) == pin["bytes"] == audit[ident]["bytes"],
         ident + ": pin differs from the file or from my audit")
    r = json.loads(raw)
    need(r["complete"] is True and r["protocol"] == C.PROTOCOL, ident + ": incomplete or other protocol")
    sources.add(r["source_sha256"]); recipes.add(r["recipe_sha256"])
    codes.setdefault(r["code_sha"], []).append(ident); plans.setdefault(r["plan_sha256"], []).append(ident)
    kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
need(sources == {plan["source_sha256"]} and recipes == {plan["recipe_sha256"]}, "the 28 receipts carry more than one source/recipe")
manifest = json.loads(Path(plan["cache_manifest"]["path"]).read_text())
need(sha_file(plan["cache_manifest"]["path"]) == plan["cache_manifest"]["sha256"] and manifest["source_sha256"] == plan["source_sha256"]
     and manifest["recipe_sha256"] == plan["recipe_sha256"], "cache manifest pin or identities differ")
payloads = [e["cache"]["sha256"] for d in manifest["classification"].values() for e in d.values()] \
    + [e["cache"]["sha256"] for d in manifest["pose"].values() for e in d.values()]
receipts_in_manifest = {e["receipt"]["sha256"] for d in manifest["classification"].values() for e in d.values()} \
    | {e["receipt"]["sha256"] for d in manifest["pose"].values() for e in d.values()}
need(len(payloads) == 21 and len(set(payloads)) == 21 and len(receipts_in_manifest) == 12, "cache manifest is not 21 payloads from 12 receipts")
need(C.source_identity() == plan["source_sha256"] and C.recipe_identity(plan) == plan["recipe_sha256"], "this worktree is not the frozen source")
recipe_keys = ("protocol", "doses", "views", "teacher", "classification_recipe", "pose_recipe", "reading_rule",
               "classification_baseline_policy", "extraction_batch")
notes.append("recipe identity covers only: " + ", ".join(recipe_keys) + "; resources and benchmark policy are outside it")

result = {"problems": problems, "notes": notes, "frozen_gate": {"failed_shards": len(failed), "failed_cells": len(failed_cells),
          "projected_core_hours": gate["projected_core_hours"]},
          "amended": {"jobs": len(mine), "unchanged_shards": unchanged, "jobs_by_case": counts,
                      "walltime_minutes_min_max": [min(j["W"] for j in mine) / 60, max(j["W"] for j in mine) / 60],
                      "min_margin_to_gate_seconds": min(j["margin"] for j in mine), "max_projection_seconds": max(j["P"] for j in mine),
                      "max_projected_memory_gib": max(j["memory"] for j in mine) / 1024**3,
                      "max_abs_difference_from_gpt_seconds": worst,
                      "walltime_histogram_minutes": {str(int(w / 60)): sum(j["W"] == w for j in mine) for w in sorted({j["W"] for j in mine})}},
          "budget": budget, "pose": {"seconds_per_dose": projection["seconds_per_dose"],
                                     "projected_gib": projection["projected_peak_rss_bytes"] / 1024**3, "admitted": projection["admitted"]},
          "inputs": {"receipts": len(pins), "kinds": kinds, "sources": sorted(sources), "recipes": sorted(recipes),
                     "code_shas": {k: len(v) for k, v in codes.items()}, "plan_sha256": {k: len(v) for k, v in plans.items()},
                     "cache_payloads": len(payloads)},
          "table": [{k: (round(j[k], 1) if k in ("P", "margin") else j[k]) for k in ("id", "folds", "P", "W", "margin")} for j in mine]}
out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "table"}, indent=1, sort_keys=True))
