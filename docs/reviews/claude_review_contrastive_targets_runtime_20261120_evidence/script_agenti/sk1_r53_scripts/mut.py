"""Independent mutation runner (scettico 1, R5-3). Usage: mut.py CLONE OUT.jsonl TESTSET ID [ID...]

Each ID is applied alone (or ALL=apply all listed together), the given test set is run,
then the file is restored with git checkout. Only files inside CLONE are touched.
"""
import json, os, subprocess, sys, time
from pathlib import Path

RES = "scripts/contrastive_target_resume.py"
MUTS = {
    # id: (file, old, new)
    "boundary_state": (RES, '                if trace[0].get("boundary", {}).get(field) != value:\n',
                            '                if False:\n'),
    "boundary_rng": (RES, '                if trace[0].get(f"rng_{domain}_before_sha256") != digest:\n',
                          '                if False:\n'),
    "start_json": (RES, '            if (start.get("protocol") != PROTOCOL or start.get("phase") != phase\n',
                        '            if False and (start.get("protocol") != PROTOCOL or start.get("phase") != phase\n'),
    "endpoint": (RES, '        if (summary.get("steps") != end_step or summary.get("stopped") != "max_steps"\n',
                      '        if False and (summary.get("steps") != end_step or summary.get("stopped") != "max_steps"\n'),
    "summary_counters": (RES, '        if (c.get("optimizer_updates") != end_step or c.get("queue_last_step") != end_step - 1\n',
                              '        if False and (c.get("optimizer_updates") != end_step or c.get("queue_last_step") != end_step - 1\n'),
    "row_invariants": (RES, '                if (row.get("rank") != rank or row.get("phase") != phase\n',
                            '                if False and (row.get("rank") != rank or row.get("phase") != phase\n'),
    "weights": (RES, '                if weights != expected_weights or not any(v > 0 for v in weights):\n',
                     '                if False:\n'),
    "trace_metrics": (RES, '                    if not _finite_number(row[name]) or row[name] != metric.get(name):\n',
                           '                    if False:\n'),
    "rank_rng": (RES, '    if ranks > 1:\n        rank_rng = state.get("rank_rng")\n',
                      '    if False:\n        rank_rng = state.get("rank_rng")\n'),
    "copy_identity": (RES, '    if file_sha(target) != expected_sha or file_sha(source_checkpoint) != expected_sha:\n',
                           '    if False:\n'),
    # weaker variants: boundary compared on the wrong row, rng boundary compared on wrong row
    "boundary_state_lastrow": (RES, '                if trace[0].get("boundary", {}).get(field) != value:\n',
                                    '                if trace[-1].get("boundary", {}).get(field, value) != value:\n'),
    # extra, not in the finding: origin mapping
    "origin_mapping": (RES, '            if origin_sha != file_sha(paths["continuous"] / "origin_sessions.json"):\n',
                            '            if False:\n'),
}


def apply(clone, ids):
    for i in ids:
        f, old, new = MUTS[i]
        p = Path(clone) / f
        text = p.read_text()
        if text.count(old) != 1:
            raise SystemExit(f"{i}: pattern count {text.count(old)}")
        p.write_text(text.replace(old, new))


def restore(clone):
    subprocess.run(["git", "-C", clone, "checkout", "-q", "--", "."], check=True)
    st = subprocess.run(["git", "-C", clone, "status", "--porcelain"], capture_output=True, text=True).stdout
    if st.strip():
        raise SystemExit("clone not clean after restore:\n" + st)


def main():
    clone, out, testset, *ids = sys.argv[1:]
    tests = Path(testset).read_text().split()
    tmp = Path("/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=f"{clone}/src:{clone}/tests/cpu:{clone}/scripts")
    groups = [[i] for i in ids] if ids[0] != "ALL" else [ids[1:]]
    for group in groups:
        label = "+".join(group)
        restore(clone)
        apply(clone, group)
        diff = subprocess.run(["git", "-C", clone, "diff", "--stat"], capture_output=True, text=True).stdout.strip()
        bt = tmp / f"sk1_r53_bt_{label[:40]}"
        t0 = time.time()
        proc = subprocess.run(["/Users/simonebenatti/.venvs/wearusfm-tok/bin/python", "-B", "-m", "pytest", "-q",
                               "-p", "no:cacheprovider", "-o", "tmp_path_retention_policy=none",
                               f"--basetemp={bt}", "--tb=line", *tests],
                              cwd=clone, env=env, capture_output=True, text=True)
        tail = [l for l in proc.stdout.splitlines() if l.strip()][-6:]
        restore(clone)
        subprocess.run(["rm", "-rf", str(bt)])
        rec = {"mutation": label, "rc": proc.returncode, "elapsed_s": round(time.time() - t0, 1),
               "diffstat": diff, "tail": tail}
        with open(out, "a") as stream:
            stream.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
