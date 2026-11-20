"""r6 (d): constants coded in readers/dispatcher vs values in the two plans (read-only)."""
import json, inspect
import contrastive_benchmark_readout as B
import contrastive_target_resume as TR
import contrastive_target_parity as TP
import contrastive_numeric_profile as NP
import contrastive_job as J
import contrastive_release as REL
import contrastive_prepare as P

plan = json.loads(P.PLAN.read_text()); pre = json.loads(REL.PREFLIGHT.read_text())
na = plan["numerical_amendment"]; rr = plan["calibration"]["benchmark_ratio_rule"]
rows = []
def chk(name, code, planv): rows.append((name, code, planv, code == planv))
chk("ratio median_min", B.RATIO_RULE["median_min"], rr["median_min"])
chk("ratio median_max", B.RATIO_RULE["median_max"], rr["median_max"])
chk("ratio p95_max", B.RATIO_RULE["p95_max"], rr["p95_max"])
chk("ratio between", [B.RATIO_RULE["between_min"], B.RATIO_RULE["between_max"]], rr["active_ratio_between_CN_CR"])
src = inspect.getsource(B.component_ratios)
chk("ratio panel 513..576 in component_ratios", "513 <= r['step'] <= 576" in src, rr["panel"] == [513, 576])
src = inspect.getsource(B.trajectory_drift)
chk("drift rel/abs tol", ("rel_tol=1e-4" in src and "abs_tol=1e-6" in src),
    (na["neutrality"]["trajectory_1_256"]["relative_tolerance"], na["neutrality"]["trajectory_1_256"]["absolute_tolerance"]) == (1e-4, 1e-6))
chk("drift updates 1..256", "range(1, 257)" in src, na["neutrality"]["updates_before"][-1] == 256)
chk("fork/end", [TR.FORK, TR.END], [na["resume"]["fork_update"], na["resume"]["end_update"]])
chk("resume 1% bound", TR.RELATIVE_LIMIT, na["resume"]["student_teacher_bound"])
chk("Adam/queue per-tensor bound", TR.RELATIVE_LIMIT, na["resume"]["Adam_and_queue_float_bound_per_tensor"])
chk("parity updates", list(TP.UPDATES), na["neutrality"]["updates_before"])
chk("parity arms x backwards", sorted({c[:2] for c in TP.CASES}), sorted(na["neutrality"]["arms"]))
chk("parity backwards per arm", len(TP.CASES) // 3, na["neutrality"]["backwards_per_arm"])
chk("profile name", NP.PROFILE, plan["common"]["numeric_profile"]["name"])
chk("required parity profile", TP.PROFILE, na["neutrality"]["required_profile"])
cfgs = P.build_review_configs(P._read(P.BASELINE))
chk("benchmark keep_checkpoints (calibrated_configs 576 adds 544)", sorted(set(cfgs["CN"]["schedule"]["keep_checkpoints"]) | {544}), plan["common"]["benchmark_keep_checkpoints"])
for cell in pre["cells"]:
    chk("seconds " + cell["id"], J.ALLOCATED_SECONDS[cell["mode"]], cell["seconds"])
gate = pre["audit_sacct_gate"]; src = inspect.getsource(REL.validate_audit_accounting)
chk("audit gate in code", all(s in src for s in ('"COMPLETED"', '"0:0"', "7200", "cpus != 8", "minutes != 120")),
    (gate["State"], gate["ExitCode"], gate["ElapsedRaw_max"], gate["AllocCPUS"], gate["TimelimitRaw"]) == ("COMPLETED", "0:0", 7200, 8, 120))
src = inspect.getsource(B._run_evidence)
c = plan["common"]
chk("run evidence 576/4/6/8/1", all(s in src for s in ('("steps", 576)', '("world_size", 4)', '("workers_per_rank", 6)', '("batch_per_process", 8)', '("accum_steps", 1)')),
    (c["world_size"], c["num_workers_per_rank"], c["batch_global"] // c["world_size"], c["accum_steps"]) == (4, 6, 8, 1))
for r in rows: print(("OK " if r[3] else "DIFF"), r[0], "| code:", r[1], "| plan:", r[2])
print("all equal:", all(r[3] for r in rows), len(rows))
