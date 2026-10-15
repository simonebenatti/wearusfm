"""Catena di valutazione (`scripts/evaluation_chain.py`, accordo del 15/10/2026) su piani sintetici col submitter vero e un login node
finto: validazione dei piani, guardie dei checkpoint, ordine e dipendenze multiple, dry-run, sottomissione rieseguibile senza doppioni,
attesa del training, blocco su stati ambigui, limiti di coda; guardia `scripts/check_checkpoint.py`."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from test_submit_job import FAKE

REPO = Path(__file__).resolve().parents[2]


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EC, CK = _mod("evaluation_chain"), _mod("check_checkpoint")
SHA = "a" * 40
OUT = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/eval"


def _node(i, kind, after=(), inputs=(), outputs=None, checkpoint=None, queue="normal"):
    n = {"id": i, "kind": kind, "after": list(after), "queue": queue, "gpu_hours": 0.5, "code_dir": "/C", "expected_sha": SHA,
         "launcher": f"scripts/slurm/{kind}.sbatch", "args": ["--x", "1"], "exports": {"FEATURES": "backbone"}, "sbatch": ["--time=01:00:00"],
         "inputs": list(inputs), "outputs": outputs if outputs is not None else [f"{OUT}/{i}.json"]}
    if checkpoint:
        n["checkpoint"] = checkpoint
    return n


def _plan():
    ck = {"run_dir": "/R/W1-A", "name": "checkpoint_postdecay.pt", "training_job": "W1-A-j04", "min_steps": 98094, "sha256": None}
    return {"schema": "eval-plan-v0", "campaign": "w1-eval", "agent": "claude",
            "guard": {"code_dir": "/C", "expected_sha": SHA, "report_root": f"{OUT}/guards"},
            "nodes": [_node("A-ext", "extract", checkpoint=ck),
                      _node("A-p1", "probe", ["A-ext"], [f"{OUT}/A-ext.json"]),
                      _node("A-p2", "probe", ["A-ext"], [f"{OUT}/A-ext.json"]),
                      _node("A-cmp", "compare", ["A-p1", "A-p2"], [f"{OUT}/A-p1.json", f"{OUT}/A-p2.json"])]}


class FakeRegistry:
    def __init__(self, states=None):
        self.states = states or {}

    def read(self, i):
        return {"stato": self.states[i]} if i in self.states else None


def test_guard_is_inserted_and_order_respects_dependencies():
    order = EC.validate(_plan())
    ids = [n["id"] for n in order]
    guard = next(n for n in order if n["kind"] == "guard")
    assert ids.index(guard["id"]) < ids.index("A-ext") < ids.index("A-p1") < ids.index("A-cmp")
    assert guard["after"] == ["W1-A-j04"] and next(n for n in order if n["id"] == "A-ext")["after"] == [guard["id"]]
    assert guard["args"][:2] == ["/R/W1-A", "checkpoint_postdecay.pt"] and "--min-steps" in guard["args"]
    two = _plan()
    two["nodes"].append(_node("A-ext-local", "extract", checkpoint=dict(two["nodes"][0]["checkpoint"])))
    assert sum(n["kind"] == "guard" for n in EC.validate(two)) == 1  # una guardia per checkpoint


@pytest.mark.parametrize("change,message", [
    (lambda p: p["nodes"].append(dict(p["nodes"][1])), "ripetuti"),
    (lambda p: p["nodes"][1].update(after=[]), "senza dipenderne"),
    (lambda p: p["nodes"][3].update(after=[]), "un confronto dipende"),
    (lambda p: p["nodes"][1].update(expected_sha="abc"), "SHA completo"),
    (lambda p: p["nodes"][1].update(exports={"X": "a,b"}), "export"),
    (lambda p: p["nodes"][1].update(sbatch=["--dependency=afterok:1"]), "dipendenze li impongono"),
    (lambda p: p["nodes"][1].update(outputs=["/leonardo_work/IscrB_WearUsFM/wearusfm/out.json"]), "dentro il codice"),
    (lambda p: p["nodes"][2].update(outputs=[f"{OUT}/A-p1.json"]), "gia' prodotta"),
    (lambda p: (p["nodes"][1].update(after=["A-ext", "A-cmp"])), "circolari"),
    (lambda p: p.update(schema="altro"), "schema"),
    (lambda p: p.pop("guard"), "guard"),
    (lambda p: p["nodes"][1].update(launcher="/abs/x.sbatch"), "launcher relativo"),
])
def test_malformed_plans_are_refused(change, message):
    p = copy.deepcopy(_plan())
    change(p)
    with pytest.raises(ValueError, match=message):
        EC.validate(p)


def test_nodes_wait_for_training_and_never_resubmit():
    plan = _plan()
    rows = EC.schedule(plan, FakeRegistry().read)
    assert {r["status"] for r in rows} == {"in attesa"}  # il training non e' ancora nel ledger
    rows = EC.schedule(plan, FakeRegistry({"W1-A-j04": "sottomesso"}).read)
    assert [r["status"] for r in rows] == ["sottomesso ora"] * 5  # la catena intera, in ordine
    cmp = next(r for r in rows if r["id"] == "A-cmp")["command"]
    assert cmp[cmp.index("--dopo") + 1] == "A-p1" and cmp.count("--dopo") == 2  # afterok su entrambe le sonde
    sep = cmp.index("--")
    assert cmp[sep + 1] == f"--export=ALL,CODE_DIR=/C,EXPECTED_SHA={SHA},FEATURES=backbone" and cmp[-3] == "/C/scripts/slurm/compare.sbatch"
    done = {r["id"]: "sottomesso" for r in rows} | {"W1-A-j04": "sottomesso"}
    assert {r["status"] for r in EC.schedule(plan, FakeRegistry(done).read)} == {"gia' sottomesso"}


def test_ambiguous_states_block_downstream_without_resubmission():
    plan = _plan()
    guard = next(n["id"] for n in EC.validate(plan) if n["kind"] == "guard")
    rows = {r["id"]: r for r in EC.schedule(plan, FakeRegistry({"W1-A-j04": "sottomesso", guard: "sottomesso", "A-ext": "ambiguo"}).read)}
    assert rows["A-ext"]["status"] == "bloccato" and "mai rilancio" in rows["A-ext"]["why"]
    assert all(rows[i]["status"] == "bloccato" for i in ("A-p1", "A-p2", "A-cmp"))
    rows = {r["id"]: r for r in EC.schedule(plan, FakeRegistry({"W1-A-j04": "rifiutato"}).read)}
    assert rows[guard]["status"] == "bloccato"


def test_queue_limits_hold_back_the_rest():
    plan = _plan()
    plan["limits"] = {"normal": 2}
    rows = EC.schedule(plan, FakeRegistry({"W1-A-j04": "sottomesso"}).read, {"normal": 0})
    st = {r["id"]: r["status"] for r in rows}
    assert st["A-ext"] == "sottomesso ora" and st["A-p1"] == "sottomesso ora" and st["A-p2"] == "in attesa"
    assert st["A-cmp"] == "in attesa"  # la sua sonda A-p2 non e' ancora sottomessa
    assert {r["status"] for r in EC.schedule(plan, FakeRegistry({"W1-A-j04": "sottomesso"}).read, {"normal": 2})} - {"sottomesso ora"} == {"in attesa"}


def test_submit_with_the_real_submitter_is_idempotent(tmp_path, capsys):
    fake = tmp_path / "fake_ssh.py"
    fake.write_text(FAKE)
    calls = tmp_path / "calls.txt"
    ssh = f"{sys.executable} {fake} ok {calls}"
    reg = tmp_path / "jobs"
    S = EC._submitter()
    assert S.main(["--registro", str(reg), "--ssh", ssh, "--id", "W1-A-j04", "--agente", "claude", "--campagna", "w1", "--gpu-ore", "96",
                   "--", "--export=ALL,CODE_DIR=/C,EXPECTED_SHA=" + SHA, "/C/scripts/slurm/train_ddp.sbatch"]) == 0
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(_plan()))
    assert EC.main([str(p), "--registro", str(reg)]) == 0 and len(calls.read_text().splitlines()) == 1  # dry-run: nessuna sottomissione
    assert EC.main([str(p), "--submit", "--registro", str(reg), "--ssh", ssh]) == 0
    remote = calls.read_text().splitlines()
    assert len(remote) == 6 and "--dependency=afterok:12345:12345" in remote[-1]  # training + guardia + 4 nodi
    assert EC.main([str(p), "--submit", "--registro", str(reg), "--ssh", ssh]) == 0
    assert len(calls.read_text().splitlines()) == 6  # rieseguita: nessun doppione
    assert "gia' sottomesso" in capsys.readouterr().out


# --- guardia del checkpoint ----------------------------------------------------------------------------------------------------------

def _run(tmp_path, stopped="max_steps", steps=100, stop=False, ckpt=b"pesi"):
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps({"stopped": stopped, "steps": steps}))
    if stop:
        (d / "STOP").write_text("allarme: collasso")
    if ckpt is not None:
        (d / "checkpoint_postdecay.pt").write_bytes(ckpt)
    return d


def test_checkpoint_guard(tmp_path):
    import hashlib

    good = CK.check(_run(tmp_path), "checkpoint_postdecay.pt", min_steps=100, sha256=hashlib.sha256(b"pesi").hexdigest())
    assert good["ok"] and good["size"] == 4
    assert not CK.check(_run(tmp_path / "a", stop=True), "checkpoint_postdecay.pt")["ok"]
    assert not CK.check(_run(tmp_path / "b", stopped="limite di tempo"), "checkpoint_postdecay.pt")["ok"]
    assert not CK.check(_run(tmp_path / "c", steps=50), "checkpoint_postdecay.pt", min_steps=100)["ok"]
    assert not CK.check(_run(tmp_path / "d", ckpt=None), "checkpoint_postdecay.pt")["ok"]
    assert not CK.check(_run(tmp_path / "e"), "checkpoint_postdecay.pt", sha256="0" * 64)["ok"]
    report = tmp_path / "rep" / "g.json"
    assert CK.main([str(_run(tmp_path / "f", stop=True)), "checkpoint_postdecay.pt", "--report", str(report)]) == 3
    assert json.loads(report.read_text())["problems"]
