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
    base = dict(shard_cpus=4, shard_mem="32G", shard_time="01:00:00", benchmark_cpus=4, benchmark_mem="32G", benchmark_time="00:30:00",
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
        assert j["cpus"] == 4 and j["mem"] == "32G" and j["time"] == "01:00:00" and j["block"] in (3, 4)
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
    # risorse degli shard fissate dopo il benchmark: si rigenera; mai oltre le 4 ore della seriale; stessi core di benchmark e shard
    longer = _plan(shard_time="02:30:00", shard_mem="48G")
    assert all(j["time"] == "02:30:00" and j["mem"] == "48G" for j in longer["jobs"] if j["stage"] == "run")
    with pytest.raises(ValueError, match="seriale"):
        _plan(shard_time="04:30:00")
    with pytest.raises(ValueError, match="stessi core"):
        _plan(shard_cpus=8)
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
    for extra in (["--plan", "/altro/plan.json"], ["--plan=/altro.json"], ["--out", "/altro.json"], ["--archive", "/x"]):
        r = subprocess.run(["bash", str(launcher), out, "run", *extra], env=frozen, capture_output=True, text=True)
        assert r.returncode == 2 and "non ammesso" in r.stdout and "VENV_PYTHON" not in r.stdout
