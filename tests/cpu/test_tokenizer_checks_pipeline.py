import json
from functools import partial

import numpy as np
import pytest

from wearusfm.tokenizer_checks.emg2pose import load_emg2pose_session
from wearusfm.tokenizer_checks.pipeline import Config, _public, run_all
from wearusfm.tokenizer_checks.sessions import discover_sessions, load_session


class FakeRunner:
    """Codici deterministici dall'RMS del token (il rumore li cambia poco) e ricostruzione
    = frazione fissa del segnale standardizzato: V2 = penalita'^2 ricostruito."""

    def __init__(self, single_recon=0.9, multi_recon=0.9):
        self.single_recon, self.multi_recon = single_recon, multi_recon

    def spatial_index(self, name):
        return int(name[1:]) - 1

    def run(self, x, spatial_idx, want_recon=True):
        b, n_ch, length = x.shape
        tokens = x.reshape(-1, 200).astype(np.float64)
        rms = np.sqrt(np.mean(tokens**2, axis=-1))
        base = np.floor(rms * 20).astype(np.int64)
        codes = (base[None, None, :] * (np.arange(4)[:, None, None] + 1) + np.arange(16)[None, :, None]) % 8192
        out = {"codes": codes.astype(np.int32)}
        if want_recon:
            sx = (tokens - tokens.mean(-1, keepdims=True)) / (tokens.std(-1, keepdims=True) + 1e-9)
            k = self.single_recon if n_ch == 1 else self.multi_recon
            out["std_x"], out["std_xrec"] = sx, k * sx
        return out


def _write_processed(root, subjects, fs=2000, layout="nested", seconds=10, n_ch=4, scale=None):
    rng = np.random.default_rng(0)
    for i, subj in enumerate(subjects):
        d = root / subj / "session1" if layout == "nested" else root / subj
        d.mkdir(parents=True)
        data = rng.normal(scale=5.0 * (i + 1), size=(int(seconds * fs), n_ch))
        if scale:
            data = np.round(data * scale).astype(np.int16)
        else:
            data = data.astype(np.int16)
        np.save(d / "data_int16.npy", data)
        chans = [{"qc_valid": True} for _ in range(n_ch)]
        meta = {"montage": {"groups": [{"channels": chans}]}, "native_fs_hz": fs}
        if scale:
            meta["int16_scale"] = scale
        (d / "metadata.json").write_text(json.dumps(meta))


def _write_emg2pose(root, users, files_per_user=2):
    import h5py

    root.mkdir(parents=True)
    csv_lines = ["session,user,filename"]
    rng = np.random.default_rng(1)
    dt = np.dtype([("time", "<f8"), ("joint_angles", "<f4", (20,)), ("emg", "<f4", (16,))])
    for u in users:
        for j in range(files_per_user):
            stem = f"rec-{u}-{j}"
            arr = np.zeros(20000, dtype=dt)
            arr["emg"] = rng.normal(scale=4.0, size=(20000, 16))
            with h5py.File(root / f"{stem}.hdf5", "w") as f:
                g = f.create_group("emg2pose")
                g.create_dataset("timeseries", data=arr)
                g.attrs["sample_rate"] = 2000.0
                g.attrs["num_channels"] = 16
            csv_lines.append(f"{stem},{u},{stem}.hdf5")
    (root / "meta.csv").write_text("\n".join(csv_lines) + "\n")


@pytest.fixture
def world(tmp_path):
    from wearusfm.tokenizer_checks.emg2pose import load_user_map

    _write_emg2pose(tmp_path / "em", ["u1", "u2", "u3"])
    _write_processed(tmp_path / "dsA", ["s1", "s2", "s3", "s4"], scale=10.0)
    _write_processed(tmp_path / "dsB", ["p1", "p2", "p3"], layout="flat")
    _write_processed(tmp_path / "low", ["l1", "l2", "l3"], fs=200)
    umap = load_user_map(tmp_path / "em" / "meta.csv")
    em_items = [(umap[p.stem], p.stem, p) for p in sorted((tmp_path / "em").glob("*.hdf5"))]
    em_loader = lambda subj, sess, path: load_emg2pose_session(path, umap)  # noqa: E731

    def ds(name, root):
        return (discover_sessions(root, name), partial(_ld, name))

    def _ld(name, subj, sess, path):
        return load_session(path, name, subj, sess)

    return (em_items, em_loader), {"dsA": ds("dsA", tmp_path / "dsA"), "dsB": ds("dsB", tmp_path / "dsB"),
                                   "low": ds("low", tmp_path / "low")}


CFG = Config(n_groups=9, v1_windows=4, seed=0, max_sessions_per_subject=1, batch=8, n_noise_seeds=2)


def test_full_pipeline_structure_and_exclusion(world):
    em, datasets = world
    rep = run_all(FakeRunner(), CFG, em, datasets)
    assert rep["scale_calibration"]["chosen"] in ("unit", "emg2pose_native")
    assert rep["v1"]["passes"] and rep["v1"]["ratio_single_over_multi"] == pytest.approx(1.0, rel=0.05)
    assert set(rep["v2"]) == {"emg2pose", "dsA", "dsB"}
    assert "low" in rep["excluded_datasets"] and "non e' eleggibile" in rep["excluded_datasets"]["low"]
    for name in ("dsA", "dsB"):
        assert rep["v2"][name]["v2_ratio_vs_emg2pose"] == pytest.approx(1.0, rel=1e-6)
        assert rep["v2"][name]["n_groups"] == 9 and rep["v2"][name]["n_windows"] == 9 * 16
    assert rep["v2_all_pass"]
    assert rep["v3"]["computable"] and rep["v3"]["datasets"] == ["dsA", "dsB", "emg2pose"]
    assert set(rep["v4"]["stable_overall"]) <= {True, False} and len(rep["v4"]["stable_overall"]) == 16
    json.dumps(rep)  # serializzabile


def test_v1_failure_stops_before_other_datasets(world):
    em, datasets = world
    rep = run_all(FakeRunner(single_recon=0.5, multi_recon=0.9), CFG, em, datasets)
    assert not rep["v1"]["passes"] and "nuova decisione" in rep["stopped"]
    assert "v2" not in rep


def test_v2_failure_is_flagged_by_analyze_dataset(world):
    from wearusfm.tokenizer_checks.pipeline import analyze_dataset, draw_groups

    _, datasets = world
    items, loader = datasets["dsA"]
    draw = draw_groups(items, loader, "dsA", CFG, np.random.default_rng(0))
    # riferimento con errore 0,01 (ricostruzione 0,9): il dataset a ricostruzione 0,5 ha errore 0,25
    res = analyze_dataset(FakeRunner(single_recon=0.5), "dsA", draw, 1.0, CFG, ref_median_nmse=0.01)
    assert res["v2_median_nmse"] == pytest.approx(0.25, rel=1e-6)
    assert res["v2_ratio_vs_emg2pose"] == pytest.approx(25.0, rel=1e-6) and res["v2_passes"] is False
    ok = analyze_dataset(FakeRunner(single_recon=0.9), "dsA", draw, 1.0, CFG, ref_median_nmse=0.01)
    assert ok["v2_passes"] is True


def test_skip_v3_and_public_helper(world):
    em, datasets = world
    cfg = Config(**{**CFG.__dict__, "skip_v3": True})
    rep = run_all(FakeRunner(), cfg, em, datasets)
    assert rep["v3"] == {"skipped": True}
    assert _public({"a": 1, "_b": 2}) == {"a": 1}


def test_v3_not_computable_with_fewer_than_three_subjects(tmp_path):
    from wearusfm.tokenizer_checks.emg2pose import load_user_map

    _write_emg2pose(tmp_path / "em", ["u1", "u2", "u3"])
    _write_processed(tmp_path / "dsA", ["s1", "s2"])  # solo 2 soggetti
    umap = load_user_map(tmp_path / "em" / "meta.csv")
    em_items = [(umap[p.stem], p.stem, p) for p in sorted((tmp_path / "em").glob("*.hdf5"))]
    ld = lambda subj, sess, path: load_session(path, "dsA", subj, sess)  # noqa: E731
    rep = run_all(
        FakeRunner(), CFG,
        (em_items, lambda s, ss, p: load_emg2pose_session(p, umap)),
        {"dsA": (discover_sessions(tmp_path / "dsA", "dsA"), ld)},
    )
    assert rep["v3"]["computable"] is False and "meno di 3 soggetti" in rep["v3"]["reason"]


def test_v4_controls_reported_and_dose_response(world):
    em, datasets = world
    rep = run_all(FakeRunner(), CFG, em, datasets)
    for name, d in rep["v4_per_dataset"].items():
        assert d["control_zero_noise_min_fraction"] == 1.0
        ctrl = np.array(d["control_tenth_floor_fraction_unchanged"])
        full = np.array(d["fraction_unchanged"])
        assert ctrl.shape == full.shape == (4, 16)
        assert ctrl.mean() >= full.mean()  # meno rumore, piu' codici invariati


def test_v4_zero_noise_control_stops_a_nondeterministic_runner(world):
    em, datasets = world

    class Jittery(FakeRunner):
        calls = 0

        def run(self, x, spatial_idx, want_recon=True):
            out = super().run(x, spatial_idx, want_recon)
            if not want_recon:
                Jittery.calls += 1
                if Jittery.calls % 2 == 1:  # la ripetizione a rumore zero cambia i codici
                    out["codes"] = (out["codes"] + 1) % 8192
            return out

    with pytest.raises(RuntimeError, match="controllo V4 fallito"):
        run_all(Jittery(), CFG, em, datasets)


def test_save_arrays_writes_one_npz_per_dataset(world, tmp_path):
    em, datasets = world
    out = tmp_path / "arrays"
    rep = run_all(FakeRunner(), CFG, em, datasets, save_arrays_dir=out)
    assert rep["arrays_dir"] == str(out)
    assert {p.stem for p in out.glob("*.npz")} == {"emg2pose", "dsA", "dsB"}
    z = np.load(out / "dsA.npz")
    n_tok = CFG.n_groups * 16 * 16
    assert z["codes"].shape == (4, 16, n_tok) and z["tokens"].shape == (n_tok, 200)
    assert len(z["group_subject"]) == CFG.n_groups and z["fraction_unchanged"].shape == (4, 16)


def test_progress_callback_reports_each_stage_and_is_json_serializable(world):
    em, datasets = world
    seen = []
    rep = run_all(FakeRunner(), CFG, em, datasets, on_progress=lambda r: seen.append(json.loads(json.dumps(r))))
    stages = [r["progress"] for r in seen]
    assert stages[0].startswith("calibrazione") and "V1 completata" in stages
    assert "emg2pose completato" in stages and "dsA completato" in stages and "dsB completato" in stages
    assert "emg2pose" in seen[-1]["datasets_done"] and "datasets_done" not in rep
    assert rep["progress"] == "completato"
