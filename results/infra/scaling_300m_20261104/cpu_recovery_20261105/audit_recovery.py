"""Offline acceptance of collected recovery benchmarks; does not submit jobs."""
import csv, hashlib, importlib.util, io, json, math, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
OLD = BASE.parent / "cpu_readers_20261105"
CODE = "aef665ba7ea66fc80291846487907bb3ef9107df"
NUMERIC = "1ac8481eb8b53fbdd423404a927cca02e908197f16ed6494268b56e5ddc24b1c"
def read(p): return json.loads(p.read_text())
def sha(b): return hashlib.sha256(b).hexdigest()
def jhash(x): return sha(json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
def pin(p): return dict(path=str(p.relative_to(ROOT)), sha256=sha(p.read_bytes()), bytes=p.stat().st_size)
def main(dataset):
    assert dataset in ("epn612", "uci_emg")
    folder = BASE / (dataset + "_completed")
    collection = read(folder / "collection.json")
    assert collection["failed_json_exists"] is False
    for f in collection["files"]:
        b = (folder / Path(f["remote_path"]).name).read_bytes()
        assert sha(b) == f["sha256"] and len(b) == f["bytes"]
    plan = read(OLD / "probe_plan.json")
    release = read(BASE / "submission_plan.json")
    job = next(x for x in release["jobs"] if x["dataset"] == dataset)
    receipt_path = BASE / "submissions" / (job["id"] + ".json")
    receipt = read(receipt_path)
    assert receipt["stato"] == "sottomesso" and receipt["uscita"] == 0
    assert receipt["codice_sha"] == CODE
    assert receipt["script_stdin_sha256"] == release["dispatcher"]["sha256"]
    for key in ("dispatcher", "submitter"):
        assert pin(ROOT / release[key]["path"]) == release[key]
    assert receipt["sbatch"][4:] == job["argv"][job["argv"].index("--")+1:]
    r = read(folder / "benchmark.json")
    assert r["protocol"] == plan["protocol"] and r["kind"] == "benchmark" and r["complete"] is True
    assert (r["cell"], r["dataset"], r["arm"]) == ("T100-G-300M-natural", dataset, "T100-G-300M")
    assert r["code_sha"] == CODE and r["plan_sha256"] == sha((OLD/"probe_plan.json").read_bytes())
    assert r["plan_content_sha256"] == jhash(plan) and r["recipe"] == plan["recipe"]
    assert r["plan_path"] == release["probe_plan"]["path"]
    assert r["step"] == plan["step"] == 5000 and r["n_features"] == 1792
    assert r["train_only"] is True and r["selection_performed"] is False and r["fold"] == 0
    assert r["numeric_sha256"] == jhash(r["numeric_context"]) == NUMERIC
    assert all(x["num_threads"] == 8 for x in r["numeric_context"]["threadpools"])
    audit = [json.loads(line) for line in (folder/"audit_progress.jsonl").read_text().splitlines()]
    assert audit == r["fits"] and [x["C"] for x in audit] == plan["recipe"]["C_grid"]
    assert len(audit) == 6 and r["n_train"] == r["fold_train_counts"][0]
    spec=importlib.util.spec_from_file_location("old_audit", OLD/"probe_benchmark/audit_collected.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for f in audit:
        module.check_fit(f, r["cell"], dataset, r["arm"], r["n_features"])
        assert f["n_train"] == r["n_train"] and f["roles_sha256"] == r["roles_sha256"]
    assert len({x["scaler_sha256"] for x in audit}) == 1
    assert r["audit_summary"] == dict(fits=6, convergence_warnings=0, reached_limit=0, converged_by_criterion=True, unresolved_fits=[])
    for cell in ("T100-G-30M-natural", "T100-G-300M-matched"):
        previous = read(OLD / "probe_benchmark" / cell / dataset / "benchmark.json")
        for key in ("n_train", "fold_train_counts", "roles_sha256", "numeric_context"):
            assert r[key] == previous[key], key
        for key in ("classes", "class_counts"):
            assert audit[0][key] == previous["fits"][0][key], key
    if dataset == "epn612":
        partial = [json.loads(x) for x in (OLD/"probe_benchmark/T100-G-300M-natural/epn612/audit_progress.jsonl").read_text().splitlines()]
        for a,b in zip(partial,audit):
            assert {k:v for k,v in a.items() if k != "elapsed_s"} == {k:v for k,v in b.items() if k != "elapsed_s"}
    raw_path=folder/"sacct.txt"
    rows=list(csv.DictReader(io.StringIO(raw_path.read_text()),delimiter="|"))
    parent=next(x for x in rows if x["JobIDRaw"] == str(receipt["job_slurm"]))
    assert all(x["State"] == "COMPLETED" and x["ExitCode"] == "0:0" for x in rows)
    assert int(parent["AllocCPUS"]) == 8 and parent["Partition"] == "lrd_all_serial"
    assert parent["NodeList"] == "login08" and "mem=30G" in parent["AllocTRES"]
    assert 0 < int(parent["ElapsedRaw"]) <= job["walltime_seconds"]
    rss=max(int(x["MaxRSS"][:-1])*1024 for x in rows if x["MaxRSS"].endswith("K"))
    assert rss <= 30*2**30 and r["peak_rss_bytes"] <= 30*2**30
    assert sum(f["elapsed_s"] for f in audit) <= r["elapsed_seconds"] <= int(parent["ElapsedRaw"])
    log_path=folder/("scaling_readers_recovery_"+str(receipt["job_slurm"])+".out")
    complete_line=json.dumps(dict(complete=True,kind="benchmark",out=job["output_parent"]+"/"+dataset))
    evidence=[(i,l) for i,l in enumerate(log_path.read_text().splitlines(),1) if l == complete_line]
    assert len(evidence) == 1
    accounting=dict(parent, ElapsedRaw=int(parent["ElapsedRaw"]), AllocCPUS=8,
                    actual_core_hours=int(parent["ElapsedRaw"])*8/3600, MaxRSS_bytes=rss,raw_sacct=pin(raw_path))
    (folder/"accounting.json").write_text(json.dumps(accounting,indent=2)+"\n")
    result=dict(accepted=True,dataset=dataset,job_id=receipt["job_slurm"],
                report=pin(folder/"benchmark.json"),audit=pin(folder/"audit_progress.jsonl"),
                accounting=pin(folder/"accounting.json"),receipt=pin(receipt_path),
                numeric_sha256=NUMERIC,code_sha=CODE,train_only=True,scientific_scores_computed=False,
                all_six_C_converged=True,n_iterations=[f["n_iter"] for f in audit],
                complete_log_evidence=dict(log=pin(log_path),line=evidence[0][0],text=evidence[0][1]),
                elapsed_seconds=int(parent["ElapsedRaw"]),actual_core_hours=accounting["actual_core_hours"],
                max_rss_gib=rss/2**30,local_remote_bytes_and_sha_verified=True,
                limitations=["Does not rehash raw caches or reconstruct folds; those retain the accepted extraction/validation evidence."])
    (folder/"acceptance.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__ == "__main__": main(sys.argv[1])
