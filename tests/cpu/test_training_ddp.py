"""Training con piu' processi (DDP, decisione del 09/10/2026): 2 processi `gloo` sulla CPU, gruppo inizializzato da file, ciclo completo
di `train()`: batch globale diviso, pesi allineati, file e diagnostiche solo dal processo 0, loss col contratto delle loss distribuite
(anche con l'accumulo), arresti coordinati, checkpoint e ripresa coi generatori di ogni processo. L'equivalenza numerica col batch globale
su un processo e' in `test_ddp_equivalence.py`."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
mp = pytest.importorskip("torch.multiprocessing")

from wearusfm.training import run as R  # noqa: E402

from test_pretraining_loader import _tree  # noqa: E402


def _state_hash(module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _worker(rank, world, store, root, mpath, out, results, max_steps, nan_rank, time_limit_s, perturb_rank, accum, bad_count_rank):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        held = {}
        make_teacher = R.make_teacher

        def capture(student):  # per leggere alla fine i pesi di questo processo
            held["student"] = student
            held["teacher"] = make_teacher(student)
            if rank == perturb_rank:  # pesi diversi su un processo PRIMA della sincronizzazione: DDP e il broadcast li devono allineare
                with torch.no_grad():
                    for m in (student, held["teacher"]):
                        for p in m.parameters():
                            p.add_(0.01)
            return held["teacher"]

        R.make_teacher = capture
        first, terms_first = {}, R.jepa_loss_terms

        def recording(student, teacher, *a, **k):  # studente e teacher al primo forward di questo processo
            if not first:
                first.update(student=_state_hash(student), teacher=_state_hash(teacher))
            return terms_first(student, teacher, *a, **k)

        R.jepa_loss_terms = recording
        if nan_rank == rank:  # perdita non finita al terzo forward, solo su questo processo
            terms_fn, calls = R.jepa_loss_terms, {"n": 0}

            def poisoned(*a, **k):
                out_ = terms_fn(*a, **k)
                calls["n"] += 1
                if calls["n"] == 3:
                    out_.num["jepa"] = out_.num["jepa"] * float("nan")
                return out_

            R.jepa_loss_terms = poisoned
        if bad_count_rank == rank:  # conteggi precalcolati sbagliati dal secondo passo, solo su questo processo
            counts_fn, calls_c = R.loss_counts, {"n": 0}

            def wrong(*a, **k):
                out_ = counts_fn(*a, **k)
                calls_c["n"] += 1
                if calls_c["n"] > accum:
                    out_["jepa"] = out_["jepa"] + 1
                return out_

            R.loss_counts = wrong
        cfg = replace(R.small_config(datasets=None, max_steps=max_steps), batch_size=4, accum_steps=accum)
        summary = R.train(cfg, Path(mpath), [Path(root)], Path(out), log=lambda m: None, dist=R.DistContext(rank, world, rank),
                          time_limit_s=time_limit_s)
        Path(results, f"rank{rank}.json").write_text(json.dumps({
            "summary": summary, "student": _state_hash(held["student"]), "teacher": _state_hash(held["teacher"]), "first": first}))
    finally:
        dist.destroy_process_group()


def _data(tmp_path):
    if (tmp_path / "processed").exists():
        return tmp_path / "processed", next(tmp_path.glob("*.json.gz"))
    return _tree(tmp_path)


def _run(tmp_path, tag, out, *, world=2, max_steps=4, nan_rank=-1, time_limit_s=None, perturb_rank=-1, accum=1, bad_count_rank=-1):
    root, mpath = _data(tmp_path)
    results = tmp_path / f"res_{tag}"
    results.mkdir()
    mp.spawn(_worker, args=(world, str(tmp_path / f"store_{tag}"), str(root), str(mpath), str(out), str(results), max_steps, nan_rank,
                            time_limit_s, perturb_rank, accum, bad_count_rank), nprocs=world, join=True)
    return [json.loads((results / f"rank{r}.json").read_text()) for r in range(world)]


def _lines(p):
    return [json.loads(x) for x in (p / "metrics.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("accum", [1, 2])
def test_ddp_two_processes_train_resume_and_stay_in_sync(tmp_path, accum):
    out = tmp_path / "run"
    res = _run(tmp_path, "a", out, accum=accum)
    assert res[0]["student"] == res[1]["student"] and res[0]["teacher"] == res[1]["teacher"]  # pesi e teacher uguali su tutti
    s = json.loads((out / "summary.json").read_text())
    assert s["steps"] == 4 and s["world_size"] == 2 and s["batch_per_process"] == 2 and s["stopped"] == "max_steps"
    assert s["accum_steps"] == accum and s["micro_batch"] == 2 // accum and s["loss_reduction"].startswith("contratto")
    rec = _lines(out)
    assert [r["step"] for r in rec] == [1, 2, 3, 4]  # un solo scrittore
    assert all(r["windows"] == 4 and sum(r["classes"].values()) == 4 for r in rec)  # batch globale
    assert all(r["total"] > 0 and "jepa" in r and "log_rms" in r for r in rec)  # loss globali (somme / conteggi) nel record
    assert all(r["t_data_s"] >= r["t_data_s_mean"] for r in rec) and "student_erank" in rec[1]
    state = torch.load(out / "checkpoint.pt", weights_only=False)
    assert state["world_size"] == 2 and len(state["rank_rng"]) == 2 and state["step"] == 4
    assert not torch.equal(state["rank_rng"][0]["torch_rng"], state["rank_rng"][1]["torch_rng"])  # un generatore per processo
    res2 = _run(tmp_path, "b", out, max_steps=6, accum=accum)  # ripresa con lo stesso numero di processi
    assert res2[0]["student"] == res2[1]["student"]
    assert [r["step"] for r in _lines(out)] == [1, 2, 3, 4, 5, 6]
    root, mpath = _data(tmp_path)
    with pytest.raises(ValueError, match="processi"):  # un processo solo da un checkpoint di due: rifiutato
        R.train(replace(R.small_config(datasets=None, max_steps=8), batch_size=4), mpath, [root], out, log=lambda m: None)


def test_ddp_student_and_teacher_equal_from_the_first_forward(tmp_path):
    res = _run(tmp_path, "p", tmp_path / "run", max_steps=2, perturb_rank=1)
    assert res[0]["first"]["student"] == res[1]["first"]["student"]  # allineati da DDP (dal processo 0)
    assert res[0]["first"]["teacher"] == res[1]["first"]["teacher"]  # allineato dal broadcast esplicito
    assert res[0]["student"] == res[1]["student"] and res[0]["teacher"] == res[1]["teacher"]


def test_ddp_nonfinite_loss_on_one_process_stops_all(tmp_path):
    out = tmp_path / "run"
    res = _run(tmp_path, "n", out, nan_rank=1)
    assert {r["summary"]["stopped"] for r in res} == {"perdita non finita al passo 2"}  # stesso passo, stessa ragione, su tutti e due
    assert (out / "STOP").read_text().startswith("perdita non finita")
    assert torch.load(out / "checkpoint.pt", weights_only=False)["step"] == 2  # l'ultimo checkpoint valido resta
    assert [r["step"] for r in _lines(out)] == [1, 2]


def test_ddp_wrong_counts_on_one_process_stop_all(tmp_path):
    out = tmp_path / "run"
    res = _run(tmp_path, "c", out, bad_count_rank=1, accum=2)
    assert {r["summary"]["stopped"] for r in res} == {"conteggi del forward diversi da quelli precalcolati al passo 1"}
    assert (out / "STOP").read_text().startswith("conteggi del forward")
    assert [r["step"] for r in _lines(out)] == [1]


def test_ddp_time_limit_is_coordinated(tmp_path):
    out = tmp_path / "run"
    res = _run(tmp_path, "t", out, time_limit_s=0.0)
    assert {r["summary"]["stopped"] for r in res} == {"limite di tempo"} and {r["summary"]["steps"] for r in res} == {0}


def test_ddp_refusals(tmp_path):
    root, mpath = _tree(tmp_path)
    cfg = replace(R.small_config(datasets=None), batch_size=4)
    with pytest.raises(ValueError, match="inizializzato"):  # gruppo dei processi non inizializzato
        R.train(cfg, mpath, [root], tmp_path / "r", log=lambda m: None, dist=R.DistContext(0, 2, 0))
    with pytest.raises(ValueError, match="micro-batch"):  # 4 finestre per processo non divisibili in 3 micro-batch
        R.train(replace(cfg, accum_steps=3), mpath, [root], tmp_path / "r2", log=lambda m: None)
    assert R.dist_from_env() == R.DistContext()  # senza torchrun: un processo solo


def test_cli_under_torchrun_two_processes(tmp_path):
    import subprocess
    import sys
    root, mpath = _tree(tmp_path)
    out = tmp_path / "cli"
    script = Path(__file__).resolve().parents[2] / "scripts" / "train_jepa.py"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", str(script), "--preset", "small",
           "--manifest", str(mpath), "--root", str(root), "--out-dir", str(out), "--device", "cpu", "--batch-size", "4", "--max-steps", "2", "--datasets", "all",
           "--distributed", "--accum-steps", "2"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text())
    assert s["steps"] == 2 and s["world_size"] == 2 and s["accum_steps"] == 2 and s["loss_reduction"].startswith("contratto")
    assert r.stdout.count("preset small") == 1  # stampa solo il processo 0
