"""Mutazioni indipendenti di contrastive_target_parity.py (scettico 1, R5-4).

Uso: mut.py <clone> <out.jsonl> <ids separati da virgola | ALL:<id+id+...>> <tests...>
Ogni mutazione: sostituzione testuale esatta (deve comparire una sola volta), pytest, ripristino via git.
"""
import json, os, subprocess, sys, time, shutil
from pathlib import Path

TARGET = "scripts/contrastive_target_parity.py"
M = {
    # 215: identita' replicate fra rank
    "cross_rank": ('    if replicated_inputs and len(set(replicated_inputs)) != 1:\n',
                   '    if False:\n'),
    # 294: live_state_restored calcolato -> forzato True
    "restored_calc": ('            row["live_state_restored"] = before == after and state_sha256(saved_grads) == state_sha256(_gradient_state(student))\n',
                      '            row["live_state_restored"] = True\n'),
    # 300-301: arresto se non ripristinato
    "restored_raise": ('        if not all(r["live_state_restored"] for r in point["ranks"]):\n',
                       '        if False:\n'),
    # 262-263: batch.index
    "batch_index": ('        if batch.index != runtime.step:\n', '        if False:\n'),
    # 255-256: cfg di produzione
    "prod_cfg": ('        if cfg.batch_size != 32 or cfg.accum_steps != 1 or cfg.amp_bf16 is not True or ctx.world_size != 4:\n',
                 '        if False:\n'),
    # 252-253: cfg cambiato fra i punti
    "cfg_changed": ('        if self.config_sha256 not in (None, cfg_digest):\n', '        if False:\n'),
    # 257-258: punto duplicato
    "dup_point": ('        if update in [p["update"] for p in self.points]:\n', '        if False:\n'),
    # 227-228: schema e profilo del rapporto
    "schema_profile": ('    if report.get("schema") != 1 or report.get("profile") != PROFILE:\n', '    if False:\n'),
    # varianti parziali (regressioni plausibili)
    "profile_only": ('    if report.get("schema") != 1 or report.get("profile") != PROFILE:\n',
                     '    if report.get("schema") != 1:\n'),
    "ws_only": ('        if cfg.batch_size != 32 or cfg.accum_steps != 1 or cfg.amp_bf16 is not True or ctx.world_size != 4:\n',
                '        if cfg.batch_size != 32 or cfg.accum_steps != 1 or cfg.amp_bf16 is not True:\n'),
    "global_ids_only": ('                                 (row.get("global_ids_sha256"),))\n',
                        '                                 (None,))\n'),
    # controllo: mutazione che DEVE essere uccisa (positivo del ripristino)
    "CTRL_restored_false": ('            row["live_state_restored"] = before == after and state_sha256(saved_grads) == state_sha256(_gradient_state(student))\n',
                            '            row["live_state_restored"] = False\n'),
    "CTRL_byte_identity": ('    if hashes and len(set(hashes)) != 1 and not missing:\n', '    if False:\n'),
}


def apply(clone, ids):
    path = Path(clone) / TARGET
    text = path.read_text()
    for i in ids:
        old, new = M[i]
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{i}: atteso 1 occorrenza, trovate {n}")
        text = text.replace(old, new)
    path.write_text(text)


def restore(clone):
    subprocess.run(["git", "-C", clone, "checkout", "-q", "--", TARGET], check=True)
    st = subprocess.run(["git", "-C", clone, "status", "--porcelain", "--untracked-files=all"],
                        capture_output=True, text=True, check=True).stdout
    return st.strip()


def run(clone, tests, tag, basetemp_root):
    bt = Path(basetemp_root) / ("sk1_r54_bt_" + tag[:60].replace("+", "_"))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": f"{clone}/src:{clone}/tests/cpu:{clone}/scripts"}
    t = time.time()
    p = subprocess.run(["/Users/simonebenatti/.venvs/wearusfm-tok/bin/python", "-B", "-m", "pytest", "-q",
                        "-p", "no:cacheprovider", "-o", "tmp_path_retention_policy=none",
                        "--basetemp", str(bt), "-rf", "--tb=line", *tests],
                       cwd=clone, env=env, capture_output=True, text=True)
    shutil.rmtree(bt, ignore_errors=True)
    lines = [l for l in p.stdout.splitlines() if l.strip()]
    summary = lines[-1] if lines else p.stderr[-500:]
    failed = [l for l in lines if l.startswith("FAILED")]
    return {"rc": p.returncode, "summary": summary, "failed": failed[:20], "seconds": round(time.time() - t, 1)}


def main():
    clone, out, spec, *tests = sys.argv[1:]
    basetemp_root = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp"
    groups = [s.split("+") for s in spec.split(",")]
    for ids in groups:
        tag = "+".join(ids)
        if ids != ["BASE"]:
            apply(clone, ids)
        try:
            res = run(clone, tests, tag, basetemp_root)
        finally:
            dirty = restore(clone)
        res.update(mutation=tag, tree_after_restore=dirty or "clean", n_test_files=len(tests))
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
