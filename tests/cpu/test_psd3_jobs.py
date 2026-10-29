"""Piano dei job e launcher di PSD-3 (`scripts/psd3_jobs.py`, `scripts/slurm/psd3_readout.sbatch`): un job per cella, blocchi sotto il limite
della coda seriale, comandi del submitter comune, piano scientifico e uscita fissati dal launcher, regola dei checkout di campagna."""

import importlib.util
import json
import shlex
import subprocess
from pathlib import Path

import pytest

from test_campaign_deploy import work  # noqa: F401 -- $WORK finto col repo canonico
from test_probe_launchers import _env as _launcher_env  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("psd3_jobs_script", REPO / "scripts" / "psd3_jobs.py")
PJ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PJ)
SLURM = REPO / "scripts" / "slurm"


def _plan(**kw):
    base = dict(shard_cpus=4, shard_mem="30G", shard_time="01:00:00", benchmark_cpus=4, benchmark_mem="30G", benchmark_time="00:30:00",
                inventory_time="00:30:00", compare_time="00:30:00")
    base.update(kw)
    return PJ.build_jobs("a" * 40, "pilot-psd3-20261029", "/W/psd3", "gpt", "pilot-psd3-20261029", **base)


def test_jobs_cover_the_grid_once_in_blocks_under_the_queue_limit(tmp_path):
    plan = _plan()
    jobs = plan["jobs"]
    assert len(jobs) == 26 and len({j["id"] for j in jobs}) == 26 and plan["gpu_hours_cap"] == 0
    runs = [j for j in jobs if j["stage"] == "run"]
    cells = []
    for j in runs:
        argv = shlex.split(j["command"])
        cells.append((argv[argv.index("--model") + 1], argv[argv.index("--dataset") + 1], argv[argv.index("--view") + 1]))
        assert j["cpus"] == 4 and j["mem"] == "30G" and j["time"] == "01:00:00" and j["block"] in (3, 4)
    assert sorted(cells) == sorted((m, d, v) for m in PJ.MODELS for d in PJ.DATASETS for v in PJ.VIEWS)
    bench = [j for j in jobs if j["stage"] == "benchmark"]
    assert len(bench) == 4 and all("--model B" in j["command"] for j in bench) and {j["block"] for j in bench} == {2}
    assert [j["stage"] for j in jobs if j["block"] == 1] == ["inventory"] and "--verify-sha" in jobs[0]["command"]
    compare = next(j for j in jobs if j["stage"] == "compare")
    argv = shlex.split(compare["command"])
    assert compare["block"] == 5 and argv[argv.index("--reports") + 1:] == [j["out"] for j in runs] and len(argv[argv.index("--reports") + 1:]) == 20
    for b in plan["blocks"]:
        assert b["n_jobs"] <= PJ.MAX_QUEUED
    for j in jobs:
        argv = shlex.split(j["command"])
        assert argv[:2] == ["python3", "scripts/submit_job.py"] and argv[argv.index("--gpu-ore") + 1] == "0" and "--dopo" not in argv
        tail = argv[argv.index("--") + 1:]
        assert tail[0] == f"--export=ALL,CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pilot-psd3-20261029,EXPECTED_SHA={'a' * 40}"
        assert tail[4].endswith("scripts/slurm/psd3_readout.sbatch") and tail[5:7] == [j["out"], j["stage"]] and j["out"].startswith("/W/psd3/")
        assert "--plan" not in argv and "--out" not in argv  # li fissa il launcher
    assert plan["core_hours_cap"]["total"] == pytest.approx(sum(j["core_hours_cap"] for j in jobs))
    assert plan["core_hours_cap"]["block_3"] + plan["core_hours_cap"]["block_4"] == pytest.approx(20 * 4 * 1.0)
    assert plan["core_hours_cap"]["total_with_checkout"] == pytest.approx(plan["core_hours_cap"]["total"] + 2 / 6)
    assert plan["budget"]["deviation_from_provisional"] == pytest.approx(plan["core_hours_cap"]["total"] - 86)
    assert plan["budget"]["probe_caps_with_checkout"] == pytest.approx(112.0)  # 50,83 + 61,0 + 0,17 dopo i tempi della fase A (6424fdc)
    assert plan["budget"]["residual_after_psd3_caps"] == pytest.approx(432.14 - 112.0 - plan["core_hours_cap"]["total_with_checkout"])
    assert plan["qos_limits"] == {"cpus_per_job": 8, "mem_mb_per_job": 30800, "walltime": "04:00:00", "max_queued_per_user": 10}
    # risorse degli shard fissate dopo il benchmark: si rigenera; mai oltre i limiti della QoS seriale (8 core, 30800M, 4 ore) per NESSUN job;
    # stessi core di benchmark e shard
    longer = _plan(shard_time="02:30:00", shard_mem="30800M")
    assert all(j["time"] == "02:30:00" and j["mem"] == "30800M" for j in longer["jobs"] if j["stage"] == "run")
    for bad in (dict(shard_time="04:30:00"), dict(inventory_time="04:00:01"), dict(compare_time="05:00:00"), dict(benchmark_time="04:00:30")):
        with pytest.raises(ValueError, match="seriale"):
            _plan(**bad)
    for bad in (dict(shard_mem="32G"), dict(shard_mem="30801M"), dict(benchmark_mem="48G"), dict(shard_mem="1T")):
        with pytest.raises(ValueError, match="30800M"):  # 32G = 32768M: sbatch rifiuterebbe (QoS, 27/10)
            _plan(**bad)
    with pytest.raises(ValueError, match="al massimo 8"):
        _plan(shard_cpus=9, benchmark_cpus=9)
    with pytest.raises(ValueError, match="memoria non valida"):
        _plan(shard_mem="30GB")
    with pytest.raises(ValueError, match="stessi core"):
        _plan(shard_cpus=8)
    # tempi degli shard per dataset dopo il benchmark: solo i job «run» di quel dataset cambiano; dataset ignoti rifiutati; limiti della QoS validi
    tuned = _plan(shard_times={"ninapro_db2": "02:00:00", "uci_emg": "00:15:00"}, shard_time_note="dal benchmark del 29/10")
    runs = {(shlex.split(j["command"])[shlex.split(j["command"]).index("--dataset") + 1], j["time"]) for j in tuned["jobs"] if j["stage"] == "run"}
    assert ("ninapro_db2", "02:00:00") in runs and ("uci_emg", "00:15:00") in runs and ("ninapro_db6", "01:00:00") in runs
    assert tuned["shard_resources"]["time_by_dataset"]["ninapro_db2"] == "02:00:00" and tuned["shard_resources"]["note"] == "dal benchmark del 29/10"
    with pytest.raises(ValueError, match="sconosciuto"):
        _plan(shard_times={"ninapro_db9": "01:00:00"})
    with pytest.raises(ValueError, match="seriale"):
        _plan(shard_times={"ninapro_db2": "04:30:00"})
    with pytest.raises(ValueError, match="40 caratteri"):
        PJ.build_jobs("abc", "c", "/W", "gpt", "c", shard_cpus=4, shard_mem="32G", shard_time="01:00:00", benchmark_cpus=4, benchmark_mem="32G",
                      benchmark_time="00:30:00", inventory_time="00:30:00", compare_time="00:30:00")
    out_dir = tmp_path / "piano"
    assert PJ.main(["--code-sha", "a" * 40, "--out-dir", str(out_dir)]) == 0
    lines = [l for l in (out_dir / "comandi.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert lines == [j["command"] for j in json.loads((out_dir / "piano_job.json").read_text())["jobs"]]


def test_launcher_fixes_plan_and_output_and_requires_a_frozen_checkout(work):  # noqa: F811
    root, canon, first, second = work
    env = _launcher_env(root)
    launcher = SLURM / "psd3_readout.sbatch"
    out = f"{root}/psd3/shards/B_ninapro_db6_dec.json"
    args = [out, "run", "--model", "B", "--dataset", "ninapro_db6", "--view", "dec"]
    frozen = {**env, "CODE_DIR": str(canon), "EXPECTED_SHA": second}
    r = subprocess.run(["bash", str(launcher), *args], env=frozen, capture_output=True, text=True)
    assert (f"VENV_PYTHON scripts/psd3_readout.py run --plan {canon}/results/infra/psd3_preparation_20261029/plan.json --out {out} "
            "--model B --dataset ninapro_db6 --view dec") in r.stdout, r.stdout + r.stderr
    r = subprocess.run(["bash", str(launcher), *args], env=env, capture_output=True, text=True)  # senza CODE_DIR
    assert r.returncode != 0 and "VENV_PYTHON" not in r.stdout
    r = subprocess.run(["bash", str(launcher), *args], env={**env, "CODE_DIR": str(canon), "EXPECTED_SHA": first}, capture_output=True, text=True)
    assert r.returncode == 2 and "atteso" in r.stderr and "VENV_PYTHON" not in r.stdout
    for bad, msg in (("rel/out.json", "relativa"), (f"{root}/wearusfm/out.json", "dentro il codice"), (f"{root}/wearusfm_campaigns/x/out.json", "dentro il codice")):
        r = subprocess.run(["bash", str(launcher), bad, "run"], env=frozen, capture_output=True, text=True)
        assert r.returncode == 2 and msg in r.stdout and "VENV_PYTHON" not in r.stdout
    r = subprocess.run(["bash", str(launcher), out, "plan"], env=frozen, capture_output=True, text=True)
    assert r.returncode == 2 and "sconosciuta" in r.stdout
    # piano e uscita li fissa il launcher: forme esatte, abbreviazioni di argparse e qualunque opzione fuori dalla lista sono rifiutate (critico 29/10)
    for extra in (["--plan", "/altro/plan.json"], ["--plan=/altro.json"], ["--out", "/altro.json"], ["--archive", "/x"], ["--pla", "/altro.json"],
                  ["--p=/altro.json"], ["--ou", "/altro.json"], ["--outp=/altro.json"], ["-o", "/altro.json"], ["--model", "B", "--view", "dec", "--ou", "/x"]):
        r = subprocess.run(["bash", str(launcher), out, "run", *extra], env=frozen, capture_output=True, text=True)
        assert r.returncode == 2 and "non ammesso" in r.stdout and "VENV_PYTHON" not in r.stdout, extra
    r = subprocess.run(["bash", str(launcher), f"{root}/psd3/inventory.json", "inventory", "--verify-sha"], env=frozen, capture_output=True, text=True)
    assert "VENV_PYTHON scripts/psd3_readout.py inventory --plan" in r.stdout and r.stdout.rstrip().endswith("--verify-sha")
    assert "--mem=30G" in launcher.read_text() and "--mem=32G" not in launcher.read_text()  # limite della QoS seriale (30800M)


def test_psd3_cli_rejects_abbreviated_options_and_keeps_its_real_source_identity(tmp_path, monkeypatch):
    """Il parser di `psd3_readout.py` non accetta abbreviazioni (`--pla`, `--ou`); l'identita' del sorgente vera (git, versioni, thread) e' stabile
    attraverso un mini `run_shard` (PCA, logistica, MLP reali) e finisce nel rapporto: nessuno stub, a differenza dei test del CLI di GPT."""
    import importlib.util as iu

    import numpy as np

    from wearusfm.harness.downstream_ab import fingerprint
    from wearusfm.harness.psd3_inputs import hash_file
    from wearusfm.harness.psd3_readers import FeatureConfig, LogisticConfig, MLPConfig

    spec = iu.spec_from_file_location("psd3_cli_for_jobs_test", REPO / "scripts" / "psd3_readout.py")
    cli = iu.module_from_spec(spec)
    spec.loader.exec_module(cli)
    import argparse

    parser = argparse.ArgumentParser(description="x", allow_abbrev=False)
    parser.add_argument("--plan"), parser.add_argument("--out")
    with pytest.raises(SystemExit):  # stesso comportamento che il CLI vero ora dichiara con allow_abbrev=False
        parser.parse_args(["--pla", "/x", "--out", "/y"])
    assert "allow_abbrev=False" in (REPO / "scripts" / "psd3_readout.py").read_text()
    rng = np.random.default_rng(3)
    s = np.repeat([f"s{i}" for i in range(8)], 6)
    y = np.tile(["rest", "move"], 24)
    x = rng.normal(size=(48, 4)).astype(np.float32)
    x[:, 0] += (y == "move") * 3
    path = tmp_path / "cache.npz"
    np.savez(path, x_pool=x, x_dec=x.astype(np.float16), y=y, s=s)
    roles = []
    for i in range(4):
        held = [f"s{2 * i}", f"s{2 * i + 1}"]
        rest = [f"s{j}" for j in range(8) if f"s{j}" not in held]
        roles.append({"train": rest[2:], "val": rest[:2], "held": held})
    entry = {"path": str(path), "pins": {"sha256": hash_file(path), "bytes": path.stat().st_size, "input_sha256": "a" * 64, "checkpoint_sha256": "b" * 64},
             "roles": roles, "subjects": sorted(set(s)), "roles_sha256": fingerprint(roles), "expected_oof_windows": 48,
             "expected_oof_rows_sha256": fingerprint(np.arange(48, dtype=np.int64)), "model": "B", "dataset": "uci_emg", "task": "p2"}
    original = cli.load_cache_view
    monkeypatch.setattr(cli, "load_cache_view", lambda p, pins, view: original(p, {k: pins[k] for k in ("sha256", "bytes")}, view))
    before = cli.source_identity()
    report = cli.run_shard(entry, "pool", feature_config=FeatureConfig(k=2), logistic_config=LogisticConfig(feature_dim=2, c_grid=(0.1, 1.0), max_iter=50),
                           mlp_config=MLPConfig(feature_dim=2, hidden_dim=3, epochs=2, lambdas=(0.0, 0.01)))
    assert report["status"] == "completed" and report["runtime"]["source_sha256"] == before["source_sha256"] == cli.source_identity()["source_sha256"]
    assert cli.numerical_identity(report["runtime"]) == cli.numerical_identity(before) and report["runtime"]["code_sha"]
    assert report["recipe_sha256"] == fingerprint(report["recipe"], cli.numerical_identity(report["runtime"]))
