"""Catene WSD della finestra 1 (15/10/2026): il pianificatore `scripts/wsd_chain.py` (job, dipendenze, rami di decay, preventivo, piani
incoerenti rifiutati prima di generare comandi) e il launcher `scripts/slurm/train_ddp.sbatch` (run gia' completo, ramo dal checkpoint della
fase stabile), provato con bash e un torchrun finto."""

import copy
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from test_campaign_deploy import REPO, work  # noqa: F401 -- fixture del $WORK finto col repo canonico

spec = importlib.util.spec_from_file_location("wsd_chain", REPO / "scripts" / "wsd_chain.py")
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)

E = 23150  # passi per epoca di esempio
SHA = "a" * 40


def _plan(**over):
    plan = {"campaign": "esempio", "agent": "claude", "code_dir": "/C", "expected_sha": SHA, "manifest": "/M", "scales": "/S",
            "window_seconds": "/W", "run_root": "/R", "slurm": {"qos": "normal", "time": "24:00:00"}, "margin_min": 40, "startup_min": 15,
            "slack": 1.15, "spare_jobs": 1, "gpus": 4, "s_per_step": {"100m/32": 1.39}, "ckpt_gb": {"100m": 1.7},
            "common_args": "--window1-rules --datasets all --prefetch",
            "runs": [{"id": "P100-s0", "preset": "100m", "seed": 0, "batch_size": 32, "accum_steps": 1, "steps_at_d": 4 * E,
                      "total_steps": 8 * E, "keep_checkpoints": [round(0.8 * E), round(1.6 * E), round(3.2 * E)], "args": ""},
                     {"id": "P100-s0-E1", "branch_of": "P100-s0", "from_step": round(0.8 * E), "total_steps": E},
                     {"id": "P100-s0-E4", "branch_of": "P100-s0", "from_step": round(3.2 * E), "total_steps": 4 * E}]}
    plan.update(over)
    return plan


def _parsed(command: str) -> dict:
    """Un comando di submit_job.py: id, dipendenza, export (come dizionario), opzioni di sbatch e argomenti dello script."""
    tok = shlex.split(command)
    sep = tok.index("--")
    head, tail = tok[:sep], tok[sep + 1:]
    exports = dict(item.split("=", 1) for item in tail[0].removeprefix("--export=ALL,").split(","))
    return {"id": head[head.index("--id") + 1], "dopo": head[head.index("--dopo") + 1] if "--dopo" in head else None,
            "gpu_ore": float(head[head.index("--gpu-ore") + 1]), "export": exports, "sbatch": tail[1:3], "script_args": tail[3:]}


def test_pilot_trunk_and_branches_become_chained_jobs():
    result = W.plan_jobs(_plan())
    trunk, e1, e4 = result["runs"]
    per_job = (24 * 3600 - 55 * 60) / (1.39 * 1.15)  # pessimista: durata meno margine e avvio, s/passo x slack
    assert trunk["jobs_needed"] == 4 and trunk["jobs"] == 5 and trunk["time"] == "24:00:00"
    assert trunk["decay_start"] == round(6.4 * E) and trunk["warmup_steps"] == 1852  # D14 con D = 4 epoche (decisione del 09/10)
    cmds = [_parsed(c) for c in result["commands"]]
    assert [c["id"] for c in cmds[:5]] == [f"P100-s0-j0{k}" for k in range(1, 6)]
    assert [c["dopo"] for c in cmds[:5]] == [None, "P100-s0-j01", "P100-s0-j02", "P100-s0-j03", "P100-s0-j04"]
    first = cmds[0]
    assert first["export"]["TIME_LIMIT_S"] == str(24 * 3600 - 40 * 60) and "BRANCH_FROM" not in first["export"]
    extra = first["export"]["EXTRA_ARGS"].split()
    assert extra[extra.index("--keep-checkpoints") + 1] == "18520:37040:74080"  # niente virgole nell'export
    assert extra[extra.index("--total-steps") + 1] == str(8 * E) and extra[extra.index("--steps-at-d") + 1] == str(4 * E)
    assert "--prefetch" in extra and extra[extra.index("--schedule") + 1] == "wsd"
    assert first["sbatch"][:2] == ["--qos=normal", "--time=24:00:00"]
    assert first["script_args"] == ["/C/scripts/slurm/train_ddp.sbatch", "/M", "/S", "/R/P100-s0", str(8 * E)]
    # rami: dal checkpoint con nome del tronco, dopo il job del tronco che lo ha sicuramente scritto (stima pessimista), durata su misura
    b1 = next(c for c in cmds if c["id"] == "P100-s0-E1-j01")
    b4 = next(c for c in cmds if c["id"] == "P100-s0-E4-j01")
    assert b1["dopo"] == "P100-s0-j01" and round(0.8 * E) <= per_job
    assert b4["dopo"] == "P100-s0-j02" and per_job < round(3.2 * E) <= 2 * per_job
    assert b1["export"]["BRANCH_FROM"] == "/R/P100-s0/checkpoint_step0018520.pt"
    assert "--keep-checkpoints" not in b1["export"]["EXTRA_ARGS"]
    assert e1["jobs_needed"] == 1 and e1["time"] == "03:00:00" and b1["sbatch"][1] == "--time=03:00:00"
    assert b1["export"]["TIME_LIMIT_S"] == str(3 * 3600 - 40 * 60) and b1["gpu_ore"] == 12
    assert next(c for c in cmds if c["id"] == "P100-s0-E1-j02")["dopo"] == "P100-s0-E1-j01"  # scorta in coda al ramo
    assert e4["decay_start"] == round(3.2 * E) and e4["from_step"] == round(3.2 * E)
    t = result["totals"]
    assert t["jobs"] == 9 and t["gpu_hours_max"] == pytest.approx(sum(r["gpu_hours_max"] for r in result["runs"]))
    assert trunk["gpu_hours_expected"] == pytest.approx(4 * (8 * E * 1.39 + 4 * 15 * 60) / 3600, abs=0.1)
    assert "| **Totale** |" in W.table(result)


@pytest.mark.parametrize("change,message", [
    (lambda p: p["runs"][1].update(from_step=12345), "non e' fra i keep_checkpoints"),
    (lambda p: p["runs"][1].update(total_steps=round(0.9 * E)), "il decay del ramo comincia"),
    (lambda p: p["runs"][1].update(seed=1), "un ramo cambia solo"),
    (lambda p: p["runs"].insert(0, p["runs"].pop(1)), "elencato prima"),
    (lambda p: p["runs"].append(dict(p["runs"][0])), "ripetuto"),
    (lambda p: p.update(s_per_step={}), "manca s_per_step"),
    (lambda p: p["runs"][0].update(args="--datasets a,b"), "virgola"),
    (lambda p: p.update(expected_sha="abc"), "40 caratteri"),
    (lambda p: p.update(extra=1), "campi sconosciuti"),
    (lambda p: p["runs"][0].update(args="--decay-fraction 0.1"), "li imposta il pianificatore"),
    (lambda p: p.update(common_args="--window1-rules --batch-size=64"), "li imposta il pianificatore"),
    (lambda p: p["runs"][0].update(total_steps=50000), "fase stabile"),
    (lambda p: (p["runs"][0].update(total_steps=1000, keep_checkpoints=[]), p.update(runs=p["runs"][:1])), "prima della fine del warmup"),
])
def test_inconsistent_plans_produce_no_commands(change, message):
    plan = copy.deepcopy(_plan())
    change(plan)
    with pytest.raises(ValueError, match=message):
        W.plan_jobs(plan)


def test_continuation_to_2d_starts_from_the_predecay_checkpoint():
    """D14: la continuazione a 2D riparte dal pre-decay di un run a D (80% di D, non fra i keep_checkpoints) e decade sul 20% finale di
    2D; parte dopo il job del tronco che arriva al pre-decay."""
    plan = _plan(runs=[{"id": "R30", "preset": "100m", "seed": 0, "batch_size": 32, "accum_steps": 1, "steps_at_d": 4 * E,
                        "total_steps": 4 * E, "keep_checkpoints": [], "args": ""},
                       {"id": "R30-2D", "branch_of": "R30", "from_step": round(3.2 * E), "total_steps": 8 * E}])
    result = W.plan_jobs(plan)
    trunk, cont = result["runs"]
    assert trunk["decay_start"] == round(3.2 * E) and cont["decay_start"] == round(6.4 * E)
    first = _parsed(next(c for c in result["commands"] if "R30-2D-j01" in c))
    assert first["export"]["BRANCH_FROM"] == "/R/R30/checkpoint_predecay.pt" and first["dopo"] == "R30-j02"
    assert cont["warmup_steps"] == trunk["warmup_steps"] == 1852  # stesso warmup di D: identico a un run a 2D da zero


def test_branch_from_a_step_inside_the_trunk_decay_is_refused():
    plan = _plan()
    plan["runs"][0]["keep_checkpoints"].append(round(7 * E))  # nel decay del tronco (da 6,4 epoche): il trainer lo rifiuterebbe
    plan["runs"][1].update(from_step=round(7 * E), total_steps=round(9 * E))
    with pytest.raises(ValueError):
        W.plan_jobs(plan)


def test_commands_pass_the_real_submitter(tmp_path):
    """I comandi generati, uno per uno nell'ordine, passano i controlli di `submit_job.py` (nome e commento li impone lui, dipendenze da
    job gia' sottomessi) e arrivano a sbatch con QoS, durata, export e dipendenza afterok; login node finto (`test_submit_job.FAKE`)."""
    from test_submit_job import FAKE, S

    fake = tmp_path / "fake_ssh.py"
    fake.write_text(FAKE)
    calls = tmp_path / "calls.txt"
    result = W.plan_jobs(_plan())
    for command in result["commands"]:
        argv = shlex.split(command)
        assert argv[:2] == ["python3", "scripts/submit_job.py"]
        assert S.main(["--registro", str(tmp_path / "jobs"), "--ssh", f"{sys.executable} {fake} ok {calls}", *argv[2:]]) == 0
    remote = calls.read_text().splitlines()
    assert len(remote) == len(result["commands"]) == result["totals"]["jobs"]
    assert all("--qos=normal" in r and "--export=ALL,CODE_DIR=/C," in r and "train_ddp.sbatch" in r for r in remote)
    assert "--dependency" not in remote[0] and all("--dependency=afterok:12345" in r for r in remote[1:])
    assert "--job-name=P100-s0-E1-j01" in next(r for r in remote if "P100-s0-E1-j01" in r)


def test_cli_writes_commands_and_estimate(tmp_path, capsys):
    p = tmp_path / "piano.json"
    p.write_text(json.dumps(_plan()))
    assert W.main([str(p), "--commands", str(tmp_path / "c.txt"), "--json", str(tmp_path / "r.json")]) == 0
    lines = (tmp_path / "c.txt").read_text().splitlines()
    saved = json.loads((tmp_path / "r.json").read_text())
    assert lines == saved["commands"] and len(lines) == saved["totals"]["jobs"] and "P100-s0-E4" in capsys.readouterr().out


# --- launcher ------------------------------------------------------------------------------------------------------------------------

def _launch(tmp, run_dir, max_steps="100", **env_extra):
    """train_ddp.sbatch col $WORK finto (repo canonico pulito), nvidia-smi e torchrun finti: torchrun scrive i suoi argomenti."""
    bin_dir = tmp / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\nfor i in 1 2 3 4; do echo 'A100, 64GB'; done\n")
    (bin_dir / "torchrun").write_text(f"#!/bin/sh\necho \"$@\" > {shlex.quote(str(tmp / 'torchrun_args'))}\n")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("CODE_DIR", "EXPECTED_SHA", "BRANCH_FROM")}
    env.update(WORK=str(tmp), SCRATCH=str(tmp / "scratch"), PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}", **env_extra)
    return subprocess.run(["bash", str(REPO / "scripts" / "slurm" / "train_ddp.sbatch"), "m.json.gz", "s.json", str(run_dir), max_steps],
                          env=env, capture_output=True, text=True)


def test_launcher_skips_a_completed_run_and_resumes_an_unfinished_one(work):  # noqa: F811
    tmp = work[0]
    run_dir = tmp / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"stopped": "max_steps", "steps": 100}))
    done = _launch(tmp, run_dir)
    assert done.returncode == 0 and "gia' completo" in done.stdout and not (tmp / "torchrun_args").exists()
    (run_dir / "summary.json").write_text(json.dumps({"stopped": "limite di tempo", "steps": 60}))
    again = _launch(tmp, run_dir)
    assert again.returncode == 0, again.stdout + again.stderr
    args = (tmp / "torchrun_args").read_text().split()
    assert "--branch-from" not in args and args[args.index("--max-steps") + 1] == "100"
    (tmp / "torchrun_args").unlink()
    (run_dir / "summary.json").write_text(json.dumps({"stopped": "max_steps", "steps": 100}))
    longer = _launch(tmp, run_dir, max_steps="150")  # completo a 100, ma la catena chiede 150: si riprende
    assert longer.returncode == 0 and (tmp / "torchrun_args").exists()


def test_launcher_starts_a_branch_only_from_an_existing_checkpoint(work):  # noqa: F811
    tmp = work[0]
    run_dir = tmp / "runs" / "ramo"
    source = tmp / "runs" / "tronco" / "checkpoint_step0000500.pt"
    early = _launch(tmp, run_dir, BRANCH_FROM=str(source))
    assert early.returncode == 3 and "non esiste ancora" in early.stdout and not (tmp / "torchrun_args").exists()
    source.parent.mkdir(parents=True)
    source.write_bytes(b"x")
    ok = _launch(tmp, run_dir, BRANCH_FROM=str(source))
    assert ok.returncode == 0, ok.stdout + ok.stderr
    args = (tmp / "torchrun_args").read_text().split()
    assert args[args.index("--branch-from") + 1] == str(source)
    source.unlink()
    (run_dir / "checkpoint.pt").write_bytes(b"x")  # il ramo ha gia' il suo checkpoint: si riprende anche senza quello del tronco
    assert _launch(tmp, run_dir, BRANCH_FROM=str(source)).returncode == 0
