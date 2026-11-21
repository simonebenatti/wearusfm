import subprocess, sys, os, shutil
from pathlib import Path
CLONE = Path("/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_probe/clone")
W = Path(sys.argv[1])
src = (CLONE / "scripts/diagnostics/contrastive_cn_restart_compare.py").read_text()
src = src.replace("ROOT = Path(__file__).resolve().parents[2]", f"ROOT = Path({str(CLONE)!r})")
test = (CLONE / "tests/cpu/test_cn_restart_compare.py").read_text()
M = {
 "M01 classify ignores local": ('    if local_changed:\n        return "HYPOTHESIS_UNSUPPORTED", "step 545 local', '    if False:\n        return "HYPOTHESIS_UNSUPPORTED", "step 545 local'),
 "M02 classify ignores excessive": ("    if excessive:\n", "    if False:\n"),
 "M03 limit x10 in compare()": ('excessive |= stages["reduced"]["relative_l2"] > LIMIT', 'excessive |= stages["reduced"]["relative_l2"] > 10 * LIMIT'),
 "M04 any_reduced >=0": ('any_reduced |= stages["reduced"]["difference_l2"] > 0', 'any_reduced |= stages["reduced"]["difference_l2"] >= 0'),
 "M05 any_local by L2 (signed zero lost)": ('any_local |= bool(stages["local"]["different_elements"])', 'any_local |= stages["local"]["difference_l2"] > 0'),
 "M06 no ordinary fidelity gate": ('    require(endpoint_hashes["ordinary"] == ENDPOINT["ordinary"] and', '    require(True or endpoint_hashes["ordinary"] == ENDPOINT["ordinary"] and'),
 "M07 no warm metadata gate": ('    require(not endpoints["warm"]["metadata_or_schema_different"]', '    require(True or not endpoints["warm"]["metadata_or_schema_different"]'),
 "M08 no placement gate": ('    require(len({p["host"] for p in placements}) == 1', '    require(True or len({p["host"] for p in placements}) == 1'),
 "M09 no reduced cross-rank gate": ('require(reduced_by_leg[mode] == hashes,', 'require(True or reduced_by_leg[mode] == hashes,'),
 "M10 no post-Adam cross-rank gate": ('require(current == after_adam[mode],', 'require(True or current == after_adam[mode],'),
 "M11 .step treated as float": ('and not path.endswith(".step") else invalid', 'else invalid'),
 "M12 identical via torch.equal": ('            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),\n                            b.contiguous().reshape(-1).view(torch.uint8)))', '            and torch.equal(a, b))'),
 "M13 no cold!=hot": ('    require(cold != hot, "intervention did not change reducer layout")\n', ''),
 "M14 no cold membership": ('    require(sorted(ids_ for _, ids_ in cold) ==', '    require(True or sorted(ids_ for _, ids_ in cold) =='),
 "M15 first_exact always": ('first_exact = not any(', 'first_exact = True or not any('),
 "M16 no before==after_warmup": ('require(before == b["after_warmup"] and before["grad_none"] is True', 'require(before["grad_none"] is True'),
 "M17 no grad_none": ('and before["grad_none"] is True, "warmup', ', "warmup'),
 "M18 no TRACE_EQUAL": ('require(all(x[k] == y[k] for k in (*TRACE_EQUAL, *SUPPORT_FIELDS))', 'require(True or all(x[k] == y[k] for k in (*TRACE_EQUAL, *SUPPORT_FIELDS))'),
 "M19 metrics without grad_norm": ('names = ("grad_norm", "grad_clip_factor",', 'names = ("grad_clip_factor",'),
 "M20 no history status check": ('    require(r["status"] == "FAILED_REQUIRES_REVIEW"', '    require(True or r["status"] == "FAILED_REQUIRES_REVIEW"'),
 "M21 denominator min": ('nd / max(nx, ny, DENOM_FLOOR)', 'nd / max(min(nx, ny), DENOM_FLOOR)'),
 "M22 no post-update ordering": ('require(row["id"] == pid and row["name"] == name,', 'require(True or row["id"] == pid and row["name"] == name,'),
 "M23 training_approved True": ('"training_approved": False,\n              "historical_status"', '"training_approved": True,\n              "historical_status"'),
 "M24 no boundary rng/input join": ('        require(before[k] == trace[t], "batch/torch RNG', '        require(True or before[k] == trace[t], "batch/torch RNG'),
 "M25 no nonfinite float check": ('elif a.is_floating_point() and (not torch.isfinite(a).all() or not torch.isfinite(b).all()):', 'elif False:'),
 "M26 boundary vs trace join removed": ('require(before[k] == trace["boundary"][k],', 'require(True or before[k] == trace["boundary"][k],'),
 "M27 same-leg distinct dirs": ('== 4, "four distinct directories required")', '>= 1, "four distinct directories required")'),
 "M28 metrics origin check": ('require(all(a.get("origin") and a["origin"] == b["origin"] for a, b in zip(x, y)),', 'require(True,'),
 "M29 history endpoint SHA": ('require(file_sha(folder / "checkpoint.pt") == ENDPOINT[mode]', 'require(True or file_sha(folder / "checkpoint.pt") == ENDPOINT[mode]'),
 "M30 historical trace sha": ('require(file_sha(reference / name) == file_sha(archive / name)', 'require(True or file_sha(reference / name) == file_sha(archive / name)'),
}
res = {}
for label, (old, new) in M.items():
    assert src.count(old) == 1, (label, src.count(old))
    d = W / "mut" / label.split()[0]; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    (d / "contrastive_cn_restart_compare.py").write_text(src.replace(old, new))
    (d / "test_mut.py").write_text(test.replace('sys.path.insert(0, str(ROOT / "scripts/diagnostics"))', f'sys.path.insert(0, {str(d)!r})'))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": f"{CLONE}/src:{CLONE}/scripts:{CLONE}/scripts/diagnostics"}
    p = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--basetemp={d}/bt", str(d / "test_mut.py")],
                       cwd=CLONE, env=env, capture_output=True, text=True)
    last = p.stdout.strip().splitlines()[-1]
    res[label] = ("KILLED" if p.returncode else "SURVIVED", last)
    print(f"{res[label][0]:9s} {label:42s} {last}", flush=True)
