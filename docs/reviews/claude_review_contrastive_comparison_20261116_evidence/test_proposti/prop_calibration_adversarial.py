"""Review Claude, dimensione `calibrazione` (16/11/2026): prove avverse sul ramo `calibration_only` del trainer vero (modello minuscolo
dei test di GPT, albero sintetico dei test). Si lancia con pytest dal mio spazio; non scrive negli alberi condivisi."""
import copy, importlib.util, json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from test_contrastive_runtime import _config, _load
from test_pose_training import _pose_setup
from wearusfm.model.fm import WearUsFM
from wearusfm.training import run as R
from wearusfm.training.contrastive_objectives import ContrastiveConfig, ContrastiveRuntime

TREE = Path(R.__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("contrastive_prepare", TREE / "scripts/contrastive_prepare.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
spec2 = importlib.util.spec_from_file_location("contrastive_run", TREE / "scripts/contrastive_run.py")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cal")
    root, manifest, pose_root = _pose_setup(tmp)
    return tmp, root, manifest, pose_root


def _recs(folder):
    return [json.loads(line) for line in (Path(folder) / "metrics.jsonl").read_text().splitlines()]


def test_delivered_training_configs_cannot_build_a_runtime_and_calibration_config_is_inert():
    folder = TREE / "results/infra/contrastive_comparison_20261116/configs"
    for arm in ("C0", "CN", "CR"):
        raw = json.loads((folder / f"{arm}.config.json").read_text())
        assert raw["contrastive"]["nce_weight"] is None and raw["contrastive"]["rnc_weight"] is None
        cfg = P.config_from_dict(raw)  # il RunConfig si costruisce, ma il runtime contrastivo no: nessun training possibile
        with pytest.raises(ValueError, match="invalid contrastive"):
            ContrastiveRuntime(cfg.contrastive)
    cal = P.config_from_dict(json.loads((folder / "calibration.config.json").read_text()))
    cc = ContrastiveConfig(**cal.contrastive)
    assert cc.calibration_only and cc.arm == "C0" and all(cc.weights(s) == (0., 0.) for s in (0, 255, 256, 400, 511))
    # i quattro config consegnati sono esattamente quelli del costruttore, e i tre di training differiscono solo per `arm`
    built = P.build_review_configs(json.loads(P.BASELINE.read_text()))
    for name in ("C0", "CN", "CR", "calibration"):
        assert json.loads((folder / f"{name}.config.json").read_text()) == built[name]
    a, b, c = (copy.deepcopy(built[n]) for n in ("C0", "CN", "CR"))
    for x in (a, b, c):
        x["contrastive"].pop("arm")
    assert a == b == c


def test_missing_weight_keys_silently_take_the_technical_default():
    # rilievo: una chiave assente NON e' un peso nullo; prende il segnaposto 0,1 della dataclass
    cc = ContrastiveConfig(arm="CN")
    assert cc.nce_weight == 0.1 and cc.rnc_weight == 0.1


def test_calibration_measures_every_batch_after_fill_with_frozen_weights_and_advancing_queue(setup):
    tmp, root, manifest, pose_root = setup
    cfg = _config(pose_root, "C0", calibration=True)
    cfg = replace(cfg, contrastive={**cfg.contrastive, "fill_steps": 2, "grad_every": 0, "nce_weight": 1.0, "rnc_weight": 1.0})
    torch.manual_seed(cfg.seed)
    init = WearUsFM(cfg.model).state_dict()
    summary = R.train(cfg, manifest, [root], tmp / "cal", log=lambda _: None)
    recs = _recs(tmp / "cal")
    assert [r["step"] for r in recs] == [1, 2, 3, 4] and summary["stopped"] == "max_steps"
    # riempimento (update 1-2): nessuna norma globale; dopo: una per batch
    assert all("contrast_global_grad_base" not in r for r in recs[:2])
    assert all(r["contrast_global_grad_base"] > 0 for r in recs[2:])
    # pesi fermi: stessi tensori dell'inizializzazione in student e teacher, ottimizzatore mai usato, lr irrilevante
    ckpt = _load(tmp / "cal")
    for part in ("student", "teacher"):
        assert all(torch.equal(init[k], ckpt[part][k]) for k in init)
    assert not ckpt["optimizer"]["state"] and summary["contrastive"]["optimizer_updates"] == 0
    # la memoria avanza: ultimo passo 3 (base 0) e chiavi presenti
    q = ckpt["contrastive_state"]["queue"]
    assert q["last_step"] == 3 and len(q["entries"]) > 0
    # i pesi-segnaposto 1,0 non entrano nella perdita applicata (braccio C0): peso registrato zero a ogni passo
    assert all(r["contrast_nce_weight"] == 0 and r["contrast_rnc_weight"] == 0 for r in recs)
    # la testa di posa resta a zero per tutto il pannello: la base misurata non contiene mai la posa sull'encoder
    assert float(ckpt["student"]["pose_head.out.weight"].abs().max()) == 0.0


def test_training_cannot_resume_from_a_calibration_directory_and_vice_versa(setup):
    tmp, root, manifest, pose_root = setup
    cal = _config(pose_root, "C0", calibration=True)
    R.train(cal, manifest, [root], tmp / "cal2", log=lambda _: None)
    for arm in ("C0", "CN", "CR"):  # un training lanciato nella cartella della calibrazione non puo' ereditarne memoria/contatori
        with pytest.raises(ValueError):
            R.train(_config(pose_root, arm), manifest, [root], tmp / "cal2", log=lambda _: None)
    R.train(_config(pose_root, "CN"), manifest, [root], tmp / "trainCN", log=lambda _: None)
    with pytest.raises(ValueError, match="start fresh"):
        R.train(_config(pose_root, "CN", calibration=True), manifest, [root], tmp / "trainCN", log=lambda _: None)


def test_three_arms_start_from_the_calibration_initial_state(setup):
    tmp, root, manifest, pose_root = setup
    cal = R.train(_config(pose_root, "C0", calibration=True), manifest, [root], tmp / "cal3", log=lambda _: None)
    shas = {arm: R.train(replace(_config(pose_root, arm), max_steps=1), manifest, [root], tmp / f"one{arm}", log=lambda _: None)
            ["contrastive"]["initial_sha256"] for arm in ("C0", "CN", "CR")}
    assert set(shas.values()) == {cal["contrastive"]["initial_sha256"]}
