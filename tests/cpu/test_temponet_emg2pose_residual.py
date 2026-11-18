"""TEMPONet residua su emg2pose (18/11/2026, `docs/temponet_emg2pose_residual_plan_20261118.md`): training e valutazione con `--arch residual`
e ricarica del checkpoint, stepper, controlli di ARCH e TRAIN_TIME nello sbatch. File separato da `test_temponet_emg2pose.py` per non toccarlo."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import temponet_emg2pose as TE  # noqa: E402
from test_temponet_emg2pose import _dataset  # noqa: E402

SBATCH = REPO / "scripts" / "slurm" / "temponet_emg2pose.sbatch"


def test_residual_train_then_eval_reloads_the_same_architecture(tmp_path):
    # `--arch residual` nel training, scritta in run.json; la valutazione la rilegge e ricarica il checkpoint con la stessa rete
    import torch

    csv, proc, pose, manifest = _dataset(tmp_path)
    common = ["--csv", str(csv), "--processed-root", str(proc), "--pose-root", str(pose), "--targets-manifest", str(manifest), "--device", "cpu"]
    run = tmp_path / "run"
    assert TE.main(["train", *common, "--out", str(run), "--epochs", "1", "--batch", "16", "--chunk-steps", "3", "--max-steps-per-epoch", "5",
                    "--arch", "residual"]) == 0
    r = json.loads((run / "run.json").read_text())
    assert r["status"] == "ok" and r["arch"] == "residual" and r["args"]["arch"] == "residual" and r["parameters"] == 74_452
    sd = torch.load(run / "model_final.pt")
    ref = TE.T.denormalizer(TE.build(0, torch.device("cpu"), 1e-4, residual=True)[0], torch.zeros(20), torch.ones(20)).state_dict()
    assert any(k.startswith("net.skips.") for k in sd) and list(sd) == list(ref)
    ev = tmp_path / "eval_val"
    assert TE.main(["eval", *common, "--out", str(ev), "--model-dir", str(run), "--val-only"]) == 0
    e = json.loads((ev / "eval.json").read_text())
    assert e["arch"] == "residual" and e["val"]["windows"] == 2 and np.isfinite(e["val"]["ema"]["mean_over_users_deg"])
    assert TE.main(["eval", *common, "--out", str(tmp_path / "eval_same"), "--model-dir", str(run), "--val-only", "--arch", "residual"]) == 0
    with pytest.raises(SystemExit):
        TE.main(["eval", *common, "--out", str(tmp_path / "eval_wrong"), "--model-dir", str(run), "--val-only", "--arch", "plain"])
    plain = tmp_path / "run_plain"  # senza --arch: la rete di [Z21], come prima
    assert TE.main(["train", *common, "--out", str(plain), "--epochs", "1", "--batch", "16", "--chunk-steps", "3", "--max-steps-per-epoch", "5"]) == 0
    p = json.loads((plain / "run.json").read_text())
    assert p["arch"] == "plain" and p["parameters"] == 71_412
    assert not any(k.startswith("net.skips.") for k in torch.load(plain / "model_final.pt"))
    old = json.loads((plain / "run.json").read_text())  # run.json di prima del 18/11, senza "arch": si legge come plain
    del old["arch"]
    (plain / "run.json").write_text(json.dumps(old))
    assert TE.main(["eval", *common, "--out", str(tmp_path / "eval_old"), "--model-dir", str(plain), "--val-only"]) == 0
    assert json.loads((tmp_path / "eval_old" / "eval.json").read_text())["arch"] == "plain"
    with pytest.raises(SystemExit):
        TE.main(["eval", *common, "--out", str(tmp_path / "eval_old_wrong"), "--model-dir", str(plain), "--val-only", "--arch", "residual"])


def test_stepper_with_the_residual_network_matches_a_plain_loop():
    import torch

    rng = np.random.default_rng(4)
    x = torch.from_numpy(rng.integers(-3000, 3000, size=(3, 4, 256, 16)).astype(np.int16))
    f = torch.from_numpy(rng.uniform(0.01, 0.02, size=(3, 4)).astype(np.float32))
    y = torch.from_numpy(rng.normal(0.7, 0.4, size=(3, 4, 20)).astype(np.float16))
    mu, sd = np.full(20, 0.7), np.full(20, 0.4)
    cpu = torch.device("cpu")
    m1, o1, l1 = TE.build(5, cpu, 1e-3, dropout=0.0, residual=True)
    st = TE.Stepper(m1, o1, l1, mu, sd, 4, False, cpu)
    for i in range(3):
        st.step(x[i], f[i], y[i])
    m2, o2, l2 = TE.build(5, cpu, 1e-3, dropout=0.0, residual=True)
    for i in range(3):
        xi = x[i].permute(0, 2, 1).float() * f[i][:, None, None]
        yi = (y[i].float() - torch.tensor(mu, dtype=torch.float32)) / torch.tensor(sd, dtype=torch.float32)
        o2.zero_grad(set_to_none=True)
        l2(m2(xi), yi).backward()
        o2.step()
    assert m1.residual and all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))


# --- sbatch: ARCH e TRAIN_TIME (eseguito fuori da SLURM: si ferma al checkout inesistente, dopo i controlli) -----------------------------

def _sbatch(tmp_path, mode="train", **env):
    base = {"PATH": "/usr/bin:/bin", "WORK": str(tmp_path / "work"), "SCRATCH": str(tmp_path / "scratch"), "CODE_DIR": str(tmp_path / "nocode")}
    args = [mode, "/a/manifest.json", "/b/out"] + (["/c/train"] if mode == "eval" else [])
    r = subprocess.run(["bash", "-x", str(SBATCH), *args], env={**base, **env}, capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.parametrize("env, budget, limit", [({}, "12000", "03:30:00"), ({"TRAIN_TIME": "03:30:00"}, "12000", "03:30:00"),
                                                ({"ARCH": "residual", "TRAIN_TIME": "04:00:00"}, "13500", "04:00:00")])
def test_sbatch_train_limit_budget_and_arch(tmp_path, env, budget, limit):
    code, out = _sbatch(tmp_path, **env)
    assert code == 2 and "lib_code_dir.sh" in out  # controlli superati, fermo al checkout inesistente
    assert f"+ BUDGET={budget}" in out and f"+ EXPECTED_TIME={limit}" in out and f"+ ARCH={env.get('ARCH', 'plain')}" in out


@pytest.mark.parametrize("env, message", [({"ARCH": "resnet"}, "ARCH: plain o residual"), ({"TRAIN_TIME": "05:00:00"}, "TRAIN_TIME non ammesso"),
                                          ({"TRAIN_TIME": "4:00:00"}, "TRAIN_TIME non ammesso")])
def test_sbatch_rejects_values_outside_the_tables(tmp_path, env, message):
    code, out = _sbatch(tmp_path, **env)
    assert code == 2 and message in out and "lib_code_dir.sh" not in out


def test_sbatch_eval_keeps_its_limit_and_every_python_call_gets_the_arch(tmp_path):
    code, out = _sbatch(tmp_path, mode="eval", ARCH="residual", TRAIN_TIME="04:00:00")
    assert code == 2 and "+ EXPECTED_TIME=00:45:00" in out and "lib_code_dir.sh" in out
    text = SBATCH.read_text()
    assert text.count("python3 scripts/temponet_emg2pose.py") == text.count('--arch "$ARCH"') == 4  # collaudo (2), training, valutazione
    assert '--time-budget-s "$BUDGET"' in text and "--time-budget-s 12000" not in text
