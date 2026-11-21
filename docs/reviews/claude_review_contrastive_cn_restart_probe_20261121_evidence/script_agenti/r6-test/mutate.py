"""Mutation harness: apply one mutation at a time to the copy-on-write clone, run the 4 new test files, restore."""
import json, os, subprocess, sys, time
from pathlib import Path

A = Path(__file__).resolve().parent
M = A / "mclone"
D = "scripts/diagnostics/"
O, C, L, G = D + "contrastive_cn_restart_observe.py", D + "contrastive_cn_restart_compare.py", D + "contrastive_cn_restart_leg.py", D + "contrastive_cn_restart.py"
MUT = [
 ("O1", O, "if len(self.parameters) != expected_parameters:", "if False:"),
 ("O2", O, "or view.untyped_storage().data_ptr() != buffer.untyped_storage().data_ptr()):", "or False):"),
 ("O3", O, "if metadata[pid] is not None or param.grad is None:", "if False:"),
 ("O4", O, 'or _tensor_hash(grad.cpu()) != row["reduced_sha256"]):', "or False):"),
 ("O5", O, 'if any(row is None for row in metadata) or len({b["index"] for b in layout}) != len(layout):', "if False:"),
 ("O6", O, '"local": local.cpu(),', '"local": reduced.to(device="cpu", copy=True),'),
 ("O7", O, "    if state.capturing:\n", "    if True:\n"),
 ("O8", O, "        if self.started:\n            raise", "        if False:\n            raise"),
 ("O9", O, 'or pin["sha256"] != _file_hash(tensor_path)):', "or False):"),
 ("O10", O, "if (len(rows) != expected_parameters or len(", "if (False or len("),
 ("O11", O, 'or sorted(pid for b in buckets for pid in b["parameter_ids"]) != list(range(len(rows)))):', "or False):"),
 ("O12", O, 'or str(tensor.dtype) != row["dtype"] or _tensor_hash(tensor) != row[stage + "_sha256"]', 'or str(tensor.dtype) != row["dtype"] or False'),
 ("O13", O, 'or tensor.storage_offset() != row["bucket_element_offset"]', "or False"),
 ("O14", O, 'if offset < end or offset + length > bucket["nbytes"]:', "if False:"),
 ("O15", O, 'or row["dtype"] != bucket["dtype"]):', "or False):"),
 ("O16", O, "rel_tol=1e-12", "rel_tol=10.0"),
 ("O17", O, "if not self.capturing or rank != self.rank:", "if False:"),
 ("C1", C, "LIMIT = 1e-6", "LIMIT = 1e-3"),
 ("C2", C, '"relative_l2": nd / max(nx, ny, DENOM_FLOOR)}', '"relative_l2": nd / max(nx, ny, 1.0)}'),
 ("C3", C, "    if local_changed:\n", "    if False:\n"),
 ("C4", C, "    if excessive:\n", "    if False:\n"),
 ("C5", C, "    if not reduced_changed:\n", "    if False:\n"),
 ("C6", C, "    if not warm_first_update_exact:\n", "    if False:\n"),
 ("C7", C, "    if not endpoint_exact:\n", "    if False:\n"),
 ("C8", C, 'any_reduced |= stages["reduced"]["difference_l2"] > 0', "any_reduced |= True"),
 ("C9", C, 'any_local |= bool(stages["local"]["different_elements"])', "any_local |= False"),
 ("C10", C, "require(reduced_by_leg[mode] == hashes,", "require(True,"),
 ("C11", C, "require(current == after_adam[mode],", "require(True,"),
 ("C12", C, 'and not metrics["ordinary"]["differences"], "instrumented', 'and True, "instrumented'),
 ("C13", C, "require(cold != hot,", "require(True,"),
 ("C14", C, "require(hot == list(zip(sizes, ids)),", "require(True,"),
 ("C15", C, "and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),", "and torch.equal(a, b) or torch.equal(a.contiguous().reshape(-1).view(torch.uint8),"),
 ("C16", C, 'kind = changed if a.is_floating_point() and not path.endswith(".step") else invalid', "kind = changed"),
 ("C17", C, 'first_exact = not any(d["step"] == FORK + 1 for d in metrics["warm"]["differences"])', "first_exact = True"),
 ("C18", C, 'require(before == b["after_warmup"] and before["grad_none"] is True,', 'require(before["grad_none"] is True,'),
 ("C19", C, 'and b["null_warmups"] == int(mode == "warm"), "intervention', 'and True, "intervention'),
 ("C20", C, 'and len({p["gpu_uuid"] for p in placements}) == RANKS,', "and True,"),
 ("C21", C, "require(all(x[k] == y[k] for k in (*TRACE_EQUAL, *SUPPORT_FIELDS)),", "require(True,"),
 ("C22", C, "for a, b in zip(x, y) for k in names if a[k] != b[k]]", "for a, b in zip(x, y) for k in names if False]"),
 ("C23", C, 'require(file_sha(continuous / "checkpoint_step0000544.pt") == FORK_SHA,', "require(True,"),
 ("C24", C, 'report = {"schema": "cn-restart-intervention-v1", "training_approved": False,', 'report = {"schema": "cn-restart-intervention-v1", "training_approved": True,'),
 ("C25", C, "count = int(raw.reshape(a.numel(), a.element_size()).any(1).sum())", "count = int((a != b).sum())"),
 ("C26", C, 'if k not in ("python_rng", "numpy_rng"):', "if False:"),
 ("C27", C, 'require(before[k] == trace["boundary"][k],', "require(True,"),
 ("C28", C, 'require(before[k] == trace[t],', "require(True,"),
 ("C29", C, 'require(not endpoints["warm"]["metadata_or_schema_different"],', "require(True,"),
 ("C30", C, 'require(left[k] == right[k], "diagnostic legs', 'require(True, "diagnostic legs'),
 ("L1", L, "warm_runtime = copy.deepcopy(runtime)", "warm_runtime = runtime"),
 ("L2", L, "        restore_rng(device, rng)\n", "        pass\n"),
 ("L3", L, "        opt.zero_grad(set_to_none=True)\n        with torch.no_grad():", "        pass\n        with torch.no_grad():"),
 ("L4", L, "                    dest.copy_(original)", "                    pass"),
 ("L5", L, "if before != after:", "if False:"),
 ("L6", L, 'if not before["grad_none"]:', "if False:"),
 ("L7", L, 'if variant == "warm":', 'if variant == "never":'),
 ("L8", L, "            if not count:\n                raise", "            if False:\n                raise"),
 ("L9", L, "if count or len(optimizers) != 1 or len(micros) != 1:", "if False:"),
 ("L10", L, 'if any(getattr(args, name.replace("-", "_")).resolve() != path for name, path in expected.items()):', "if False:"),
 ("L11", L, "if args.out_dir != out or out.is_symlink() or out.exists():", "if False:"),
 ("L12", L, 'if [p.resolve() for p in args.root] != [Path(p).resolve() for p in job["roots"]]:', "if False:"),
 ("L13", L, "if len(named) != expected_count or [id(p) for _, p in named] != [id(p) for p in all_params]:", "if False:"),
 ("L14", L, '"optimizer": opt.state_dict(),', '"optimizer": None,'),
 ("L15", L, '"prepared_runtime_sha256": transient,', '"prepared_runtime_sha256": None,'),
 ("L16", L, "    if count != 1:\n", "    if False:\n"),
 ("L17", L, 'if os.environ.get("WORLD_SIZE") != "4":', "if False:"),
 ("L18", L, "        observe.start()\n", "        pass\n"),
 ("L19", L, "        step_hooks.extend(record_optimizer_step(opt, student, folder, ctx.rank))\n", "        pass\n"),
 ("L20", L, '"student_buffers_sha256": dict(student.named_buffers()),', '"student_buffers_sha256": None,'),
 ("L21", L, 'gpu_uuid = getattr(props, "uuid", None)', 'gpu_uuid = "fixed"'),
 ("G1", G, "if len(blocks) != 1:", "if len(blocks) < 1:"),
 ("G2", G, "    if data != dict(schema=", "    if False and data != dict(schema="),
 ("G3", G, 'if git("show", f"{commit}:{path.relative_to(ROOT)}") != path.read_bytes():', "if False:"),
 ("G4", G, '"--is-ancestor", commit, REVIEW_REF], check=True)', '"--is-ancestor", commit, REVIEW_REF], check=False)'),
 ("G5", G, 'or hashlib.sha256(body.encode()).hexdigest() != bound["review_sha256"]):', "or False):"),
 ("G6", G, "    review_fields(body, pins)\n    return plan", "    return plan"),
 ("G7", G, 'plan["status"] != "REVIEW_REQUIRED"', "False"),
 ("G8", G, 'or source_digest() != plan["production_source_sha256"]):', "or False):"),
 ("G9", G, '*[str(base / path) for path in evidence_paths]], check=True)', '*[str(base / path) for path in evidence_paths]], check=False)'),
 ("G10", G, 'if job["arm"] != "CN" or job["repeat"] != "a" or job["fork_checkpoint"] != plan["fork"]:', "if False:"),
 ("G11", G, "deadline = min(start + 875, time.monotonic() + end_epoch - time.time() - 25)", "deadline = start + 10**6"),
 ("G12", G, '    validate_allocation("gpu")\n', "    pass\n"),
 ("G13", G, "or root.exists() or root.is_symlink()):", "or False):"),
 ("G14", G, "COMPARE, LAUNCHER, PROTOCOL)", "COMPARE, PROTOCOL)"),
 ("G15", G, "validate_review_binding(job[\"package_review\"])", "pass"),
 ("G16", G, 'if (plan["schema"] != "cn-restart-probe-v1"', 'if (False'),
]
TESTS = ["tests/cpu/test_cn_restart_package.py", "tests/cpu/test_cn_restart_leg.py",
         "tests/cpu/test_cn_restart_observe.py", "tests/cpu/test_cn_restart_compare.py"]
env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": "src:scripts:scripts/diagnostics"}
PY = os.path.expanduser("~/.venvs/wearusfm-tok/bin/python")
only = set(sys.argv[1:])
results = []
out = A / "mutation_results.jsonl"
for mid, rel, old, new in MUT:
    if only and mid not in only:
        continue
    path = M / rel
    src = path.read_text()
    n = src.count(old)
    if n != 1:
        rec = {"id": mid, "status": f"NOT_APPLIED(count={n})"}
    else:
        path.write_text(src.replace(old, new))
        t = time.time()
        try:
            p = subprocess.run([PY, "-B", "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                                "--basetemp", str(A / "bt_mut" / mid), *TESTS],
                               cwd=M, env=env, capture_output=True, text=True, timeout=240)
            rc, tail = p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1:]
        except subprocess.TimeoutExpired:
            rc, tail = "timeout", []
        finally:
            path.write_text(src)
        rec = {"id": mid, "file": rel.split("/")[-1], "status": "SURVIVED" if rc == 0 else "KILLED",
               "rc": rc, "tail": tail, "s": round(time.time() - t, 1)}
    print(json.dumps(rec), flush=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
