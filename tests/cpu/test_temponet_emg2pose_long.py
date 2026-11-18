"""TEMPONet su emg2pose, training lungo (18/11/2026, `docs/temponet_emg2pose_long_training_proposal_20261118.md`): EPOCHS, limiti 13:00 e 15:00,
valutazione di un checkpoint dichiarato (`--checkpoint`, CHECKPOINT nello sbatch)."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import temponet_emg2pose as TE  # noqa: E402
from test_temponet_emg2pose import _dataset  # noqa: E402
from test_temponet_emg2pose_residual import SBATCH, _sbatch  # noqa: E402


def test_eval_of_a_declared_checkpoint(tmp_path):
    csv, proc, pose, manifest = _dataset(tmp_path)
    common = ["--csv", str(csv), "--processed-root", str(proc), "--pose-root", str(pose), "--targets-manifest", str(manifest), "--device", "cpu"]
    run = tmp_path / "run"
    assert TE.main(["train", *common, "--out", str(run), "--epochs", "2", "--batch", "16", "--chunk-steps", "3", "--max-steps-per-epoch", "5"]) == 0
    out = {}
    for name, extra in (("final", []), ("e01", ["--checkpoint", "epoch_01.pt"]), ("e02", ["--checkpoint", "epoch_02.pt"])):
        assert TE.main(["eval", *common, "--out", str(tmp_path / name), "--model-dir", str(run), "--val-only", *extra]) == 0
        out[name] = json.loads((tmp_path / name / "eval.json").read_text())
    assert out["final"]["checkpoint"].endswith("model_final.pt") and out["e01"]["checkpoint"].endswith("checkpoints/epoch_01.pt")
    assert out["e02"]["val"] == out["final"]["val"] and out["e02"]["alpha"] == out["final"]["alpha"]  # epoca 2 = modello finale
    assert out["e01"]["val"] != out["final"]["val"]
    for bad in ("epoch_05.pt", "../model_final.pt", "epoch_1.pt", "model_final.pt"):
        with pytest.raises(SystemExit):
            TE.main(["eval", *common, "--out", str(tmp_path / f"bad_{len(bad)}"), "--model-dir", str(run), "--val-only", "--checkpoint", bad])


@pytest.mark.parametrize("env, budget", [({"TRAIN_TIME": "13:00:00", "EPOCHS": "100"}, "45900"),
                                         ({"TRAIN_TIME": "15:00:00", "EPOCHS": "100", "ARCH": "residual"}, "53100")])
def test_sbatch_long_limits(tmp_path, env, budget):
    code, out = _sbatch(tmp_path, **env)
    assert code == 2 and "lib_code_dir.sh" in out  # controlli superati, fermo al checkout inesistente
    assert f"+ BUDGET={budget}" in out and f"+ EXPECTED_TIME={env['TRAIN_TIME']}" in out and "+ EPOCHS=100" in out


def test_sbatch_defaults_are_unchanged(tmp_path):
    code, out = _sbatch(tmp_path)
    assert code == 2 and "+ EPOCHS=19" in out and "+ CHECKPOINT=" in out and "+ BUDGET=12000" in out
    text = SBATCH.read_text()
    assert '--arch "$ARCH" --epochs "$EPOCHS"' in text and text.count('--epochs "$EPOCHS"') == 1  # solo il training (il collaudo ha --epochs 2)
    assert text.count('${CHECKPOINT:+--checkpoint "$CHECKPOINT"}') == 1


@pytest.mark.parametrize("env, message", [({"EPOCHS": "abc"}, "EPOCHS"), ({"EPOCHS": "0"}, "EPOCHS"), ({"EPOCHS": "019"}, "EPOCHS"),
                                          ({"EPOCHS": ""}, None), ({"CHECKPOINT": "../x.pt"}, "CHECKPOINT"),
                                          ({"CHECKPOINT": "epoch_1.pt"}, "CHECKPOINT"), ({"TRAIN_TIME": "14:00:00"}, "TRAIN_TIME non ammesso")])
def test_sbatch_rejects_bad_epochs_checkpoint_and_limits(tmp_path, env, message):
    code, out = _sbatch(tmp_path, mode="eval", **env)
    if message is None:  # EPOCHS vuota vale il default 19
        assert code == 2 and "+ EPOCHS=19" in out and "lib_code_dir.sh" in out
    else:
        assert code == 2 and message in out and "lib_code_dir.sh" not in out


@pytest.mark.parametrize("name", ["epoch_19.pt", "epoch_50.pt", "epoch_100.pt"])
def test_sbatch_accepts_the_declared_checkpoints(tmp_path, name):
    code, out = _sbatch(tmp_path, mode="eval", CHECKPOINT=name, ARCH="residual")
    assert code == 2 and f"+ CHECKPOINT={name}" in out and "+ EXPECTED_TIME=00:45:00" in out and "lib_code_dir.sh" in out
