"""Simula orchestrate() con worker finti: quale profilo gira, quali restano, quale status."""
import importlib.util, json, subprocess, sys, types
from pathlib import Path
import torch

CLONE = Path(sys.argv[1]); OUT = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("ccs", CLONE / "scripts/diagnostics/contrastive_common_state_cuda.py")
ccs = importlib.util.module_from_spec(spec); spec.loader.exec_module(ccs)

def profile_json(p):
    case = {k: "x" for k in ccs.MATCH_FIELDS}
    case.update(after_rng="a", nce_eligible=1, rnc_windows=1.0)
    return {"profile": p,
            "ranks": [{"rank": r, "global_batch_sha256": "g", "cases": {c: dict(case) for c in ccs.CASES}, "flags": {}} for r in range(4)],
            "pairs": {},
            "post_backward_rng_matches_C0a_by_rank": {c: [True] * 4 for c in ccs.CASES},
            "reduced_gradient_agreement_by_case": {c: True for c in ccs.CASES}}

def scenario(name, fail, unsupported=False):
    attempted = []
    def fake_run(cmd, stdout=None, stderr=None, env=None):
        p = cmd[cmd.index("--worker") + 1]; out = Path(cmd[cmd.index("--out-dir") + 1]); scratch = Path(cmd[cmd.index("--scratch") + 1])
        attempted.append(p)
        if p == fail:
            for r in range(4):
                (out / f"{p}.error.rank{r}.json").write_text(json.dumps({"unsupported": unsupported, "type": "X", "message": "m"}))
            return types.SimpleNamespace(returncode=1)
        (out / f"{p}.json").write_text(json.dumps(profile_json(p)))
        torch.save({"backbone.blocks.0.w": torch.ones(3)}, scratch / f"{p}.pt")
        return types.SimpleNamespace(returncode=0)
    old = subprocess.run; subprocess.run = fake_run
    try:
        rc = ccs.orchestrate(OUT / name)
    finally:
        subprocess.run = old
    rep = json.loads((OUT / name / "report.json").read_text())
    print(f"{name}: attempted={attempted} completed={rep['completed_profiles']} "
          f"failures={{{', '.join(k + ':' + v['category'] for k, v in rep['failures'].items())}}} "
          f"cross={sorted(rep.get('cross_profile', {}))} status={rep['status']} rc={rc}")

scenario("S1_bf16_math_failed", "bf16_math")
scenario("S2_fp32_math_failed", "fp32_math")
scenario("S3_det_failed_oom", "bf16_math_deterministic")
scenario("S4_det_unsupported", "bf16_math_deterministic", unsupported=True)
scenario("S5_all_ok", None)
