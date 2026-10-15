"""Catena di valutazione (`scripts/evaluation_chain.py`, accordo del 15/10/2026) su piani sintetici col submitter vero e un login node
finto: validazione dei piani, guardie dei checkpoint, ordine e dipendenze multiple, dry-run, sottomissione rieseguibile senza doppioni,
attesa del training, blocco su stati ambigui, limiti di coda; guardia `scripts/check_checkpoint.py`."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from test_campaign_deploy import work  # noqa: F401 -- $WORK finto col repo canonico

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
    (lambda p: p["nodes"][1].update(sbatch=["--dependency=afterok:1"]), "li impongono catena e submitter"),
    # regressioni della review GPT di 5d7ced6: export che riscrive CODE_DIR/EXPECTED_SHA, abbreviazioni accettate da sbatch, opzioni fuori elenco
    (lambda p: p["nodes"][1].update(sbatch=["--export=ALL,CODE_DIR=/OTHER,EXPECTED_SHA=" + "b" * 40]), "li impongono catena e submitter"),
    (lambda p: p["nodes"][1].update(sbatch=["--exp=ALL,CODE_DIR=/OTHER"]), "li impongono catena e submitter"),
    (lambda p: p["nodes"][1].update(sbatch=["--dep=afterok:1"]), "li impongono catena e submitter"),
    (lambda p: p["nodes"][1].update(sbatch=["-dafterok:1"]), "li impongono catena e submitter"),
    (lambda p: p["nodes"][1].update(sbatch=["--array=0-7"]), "li impongono catena e submitter"),
    (lambda p: p["nodes"][1].update(sbatch=["--export-file=/tmp/e"]), "li impongono catena e submitter"),
    (lambda p: p["nodes"][1].update(sbatch=["--part=boost_usr_prod"]), "non ammessa"),
    (lambda p: p["nodes"][1].update(sbatch=["--chdir=/C"]), "non ammessa"),
    (lambda p: p["nodes"][1].update(sbatch=["--time=01:00:00", "-t", "02:00:00"]), "ripetuta"),
    (lambda p: p["nodes"][1].update(sbatch=["--time=01:00:00", "-p"]), "manca il valore"),
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
    plan["limits"] = {"normal": 3}
    rows = EC.schedule(plan, FakeRegistry({"W1-A-j04": "sottomesso"}).read, {"normal": 0})
    st = {r["id"]: r["status"] for r in rows}
    guard = next(r["id"] for r in rows if r["kind"] == "guard")
    # la guardia (lrd_all_serial, QoS di default normal) conta anche lei per "normal"
    assert st[guard] == st["A-ext"] == st["A-p1"] == "sottomesso ora" and st["A-p2"] == "in attesa"
    assert st["A-cmp"] == "in attesa"  # la sua sonda A-p2 non e' ancora sottomessa
    assert {r["status"] for r in EC.schedule(plan, FakeRegistry({"W1-A-j04": "sottomesso"}).read, {"normal": 3})} - {"sottomesso ora"} == {"in attesa"}


def test_queue_limits_count_partition_and_qos_of_each_node(tmp_path):
    """Regressione della review GPT di 5d7ced6: un nodo conta per `queue`, per la partizione e per la QoS effettive."""
    serial = _node("serial-probe", "probe", queue="lrd_all_serial")
    gpu = _node("gpu-extract", "extract", queue="normal")
    serial["sbatch"] = ["--partition=lrd_all_serial", "--qos=normal"]
    gpu["sbatch"] = ["--partition=boost_usr_prod", "--qos=normal"]
    plan = {"schema": "eval-plan-v0", "campaign": "repro", "agent": "claude", "limits": {"normal": 1}, "nodes": [serial, gpu]}
    ssh, state = _cluster(tmp_path)
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    assert EC.main([str(p), "--submit", "--registro", str(tmp_path / "jobs"), "--ssh", ssh]) == 0
    calls = _sbatch_calls(state)
    assert len(calls) == 1 and "--qos=normal" in calls[0]
    assert EC.node_labels(serial) == {"lrd_all_serial", "normal"} and EC.node_labels(gpu) == {"normal", "boost_usr_prod"}
    gpu["sbatch"] = ["-p", "boost_usr_prod", "-q", "boost_qos_dbg"]  # forma breve, valore nell'elemento successivo
    assert EC.node_labels(gpu) == {"normal", "boost_usr_prod", "boost_qos_dbg"}
    plan["limits"] = {"boost_usr_prod": 1}  # solo i nodi GPU: la sonda seriale non conta
    assert [r["status"] for r in EC.schedule(plan, FakeRegistry().read, {"boost_usr_prod": 0})] == ["sottomesso ora"] * 2


def test_queue_labels_come_from_the_launcher_header_at_the_expected_sha():
    import subprocess

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    n = _node("g", "probe", queue="mia_etichetta")
    n.update(expected_sha=head, launcher="scripts/slurm/check_checkpoint.sbatch", sbatch=[])
    assert EC.node_labels(n) == {"mia_etichetta", "lrd_all_serial", "normal"}  # -p dell'intestazione, QoS di default
    n["sbatch"] = ["--qos=boost_qos_dbg"]  # l'opzione del nodo vince sull'intestazione, come in sbatch
    assert EC.node_labels(n) == {"mia_etichetta", "lrd_all_serial", "boost_qos_dbg"}
    n.update(expected_sha="c" * 40, sbatch=[])  # SHA assente qui: solo le etichette dichiarate
    assert EC.node_labels(n) == {"mia_etichetta"}


CLUSTER = r"""
import json, pathlib, sys
state, remote = pathlib.Path(sys.argv[1]), sys.argv[2]
st = json.loads(state.read_text())
st["calls"].append(remote)
if remote.startswith("sbatch"):
    st["next"] += 1
    print(st["next"])
elif remote.startswith("sacct"):
    if st.get("sacct_fail"):
        print("slurmdbd non raggiungibile", file=sys.stderr); state.write_text(json.dumps(st)); sys.exit(1)
    ids = remote.split("-j ")[1].split()[0].split(",")
    for j in ids:
        if j in st["jobs"]:
            print(f"{j}|{st['jobs'][j]}")
elif remote.startswith("squeue"):
    if st.get("squeue_fail"):
        print("errore", file=sys.stderr); state.write_text(json.dumps(st)); sys.exit(1)
    for line in st["queue"]:
        print(line)
state.write_text(json.dumps(st))
"""


def _cluster(tmp_path, **st):
    fake = tmp_path / "cluster.py"
    fake.write_text(CLUSTER)
    state = tmp_path / "cluster.json"
    state.write_text(json.dumps({"calls": [], "next": 1000, "jobs": {}, "queue": [], **st}))
    return f"{sys.executable} {fake} {state}", state


def _sbatch_calls(state):
    return [c for c in json.loads(state.read_text())["calls"] if c.startswith("sbatch")]


def _train(S, reg, ssh):
    assert S.main(["--registro", str(reg), "--ssh", ssh, "--id", "W1-A-j04", "--agente", "claude", "--campagna", "w1", "--gpu-ore", "96",
                   "--", "--export=ALL,CODE_DIR=/C,EXPECTED_SHA=" + SHA, "/C/scripts/slurm/train_ddp.sbatch"]) == 0


def test_submit_with_the_real_submitter_is_idempotent(tmp_path, capsys):
    ssh, state = _cluster(tmp_path, jobs={"1001": "RUNNING|0:0"})
    reg = tmp_path / "jobs"
    S = EC._submitter()
    _train(S, reg, ssh)
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(_plan()))
    assert EC.main([str(p), "--registro", str(reg)]) == 0 and len(_sbatch_calls(state)) == 1  # dry-run: nessuna sottomissione
    assert EC.main([str(p), "--submit", "--registro", str(reg), "--ssh", ssh]) == 0
    calls = _sbatch_calls(state)
    assert len(calls) == 6 and "--dependency=afterok:1001" in calls[1]  # training in esecuzione: afterok
    assert "--dependency=afterok:1004:1005" in calls[-1]  # il confronto dopo le due sonde
    assert EC.main([str(p), "--submit", "--registro", str(reg), "--ssh", ssh]) == 0
    assert len(_sbatch_calls(state)) == 6  # rieseguita: nessun doppione
    assert "gia' sottomesso" in capsys.readouterr().out


def test_finished_predecessors_get_no_afterok_and_failed_ones_block(tmp_path):
    """Ultrareview del 15/10: SLURM dimentica i job finiti (MinJobAge) e rifiuterebbe afterok su un ID sconosciuto. Il predecessore finito
    bene (sacct) non entra nella dipendenza; uno finito male blocca; uno che sacct non conosce fa aspettare."""
    S = EC._submitter()
    for training, expect in (("COMPLETED|0:0", "senza"), ("FAILED|1:0", "bloccato"), ("COMPLETED|3:0", "bloccato"), (None, "in attesa")):
        d = tmp_path / str(len(list(tmp_path.iterdir())))
        d.mkdir()
        ssh, state = _cluster(d, jobs={} if training is None else {"1001": training})
        reg = d / "jobs"
        _train(S, reg, ssh)
        p = d / "plan.json"
        p.write_text(json.dumps(_plan()))
        assert EC.main([str(p), "--submit", "--registro", str(reg), "--ssh", ssh]) == 0
        calls = _sbatch_calls(state)
        if expect == "senza":
            assert len(calls) == 6 and "--dependency" not in calls[1]  # la guardia parte subito
        else:
            assert len(calls) == 1  # solo il training: nulla sottomesso
    ssh, state = _cluster(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, sacct_fail=True)
    reg = tmp_path / "x" / "jobs"
    _train(S, reg, ssh)
    p = tmp_path / "x" / "plan.json"
    p.write_text(json.dumps(_plan()))
    assert EC.main([str(p), "--submit", "--registro", str(reg), "--ssh", ssh]) == 1 and len(_sbatch_calls(state)) == 1


def test_queue_limits_use_live_counts_when_submitting(tmp_path):
    S = EC._submitter()
    plan = _plan()
    plan["limits"] = {"normal": 3}
    ssh, state = _cluster(tmp_path, jobs={"1001": "RUNNING|0:0"}, queue=["boost_usr_prod|normal", "boost_usr_prod|normal", "lrd_all_serial|normal"])
    reg = tmp_path / "jobs"
    _train(S, reg, ssh)
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    assert EC.main([str(p), "--submit", "--registro", str(reg), "--ssh", ssh]) == 0
    # 3 job dell'utente gia' in coda sulla QoS normal: nessun nodo sottomesso, nemmeno la guardia (lrd_all_serial, QoS di default normal);
    # l'unico sbatch e' quello del training
    assert len(_sbatch_calls(state)) == 1
    ssh2, state2 = _cluster(tmp_path / "y" if (tmp_path / "y").mkdir() is None else tmp_path, squeue_fail=True)
    reg2 = tmp_path / "y" / "jobs"
    _train(S, reg2, ssh2)
    assert EC.main([str(p), "--submit", "--registro", str(reg2), "--ssh", ssh2]) == 1 and len(_sbatch_calls(state2)) == 1


def test_any_other_ledger_state_or_a_stateless_reservation_blocks(tmp_path):
    plan = _plan()
    guard = next(n["id"] for n in EC.validate(plan) if n["kind"] == "guard")
    rows = {r["id"]: r for r in EC.schedule(plan, FakeRegistry({"W1-A-j04": "sottomesso", guard: "prenotato_di_nuovo"}).read)}
    assert rows[guard]["status"] == "bloccato" and rows["A-ext"]["status"] == "bloccato"
    S = EC._submitter()
    reg = S.Registry(tmp_path / "jobs")
    (tmp_path / "jobs" / guard).mkdir(parents=True)  # prenotazione rimasta senza state.json
    read = EC._ledger_reader(reg)
    assert read(guard) == {"stato": "prenotazione senza stato"} and read("mai_visto") is None


def test_a_refusal_by_the_submitter_stops_the_call_cleanly(tmp_path, monkeypatch, capsys):
    real = EC._submitter()

    class Refusing:
        Registry, DEFAULT_DIR, run_remote = real.Registry, real.DEFAULT_DIR, real.run_remote

        @staticmethod
        def main(argv):
            raise SystemExit("prova: gia' prenotato")

    monkeypatch.setattr(EC, "_submitter", lambda: Refusing)
    ssh, state = _cluster(tmp_path, jobs={"1001": "RUNNING|0:0"})
    reg = tmp_path / "jobs"
    _train(real, reg, ssh)
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(_plan()))
    assert EC.main([str(p), "--submit", "--registro", str(reg), "--ssh", ssh]) == 1
    assert "rifiutato dal submitter" in capsys.readouterr().out


def test_guards_differ_when_requirements_differ():
    plan = _plan()
    ck = dict(plan["nodes"][0]["checkpoint"], sha256="b" * 64)
    plan["nodes"].append(_node("A-ext-sha", "extract", checkpoint=ck))
    guards = [n for n in EC.validate(plan) if n["kind"] == "guard"]
    assert len(guards) == 2 and sum("--sha256" in g["args"] for g in guards) == 1


def test_guard_launcher_uses_the_venv_python(work, tmp_path):  # noqa: F811
    import os
    import subprocess

    root = work[0]
    venv = root / "venv_wearusfm" / "bin"
    venv.mkdir(parents=True)
    fakebin = root / "venvpy"
    fakebin.mkdir()
    (fakebin / "python3").write_text("#!/bin/sh\necho VENV_PYTHON \"$@\"\n")
    (fakebin / "python3").chmod(0o755)
    (venv / "activate").write_text(f"export PATH={fakebin}:$PATH\n")
    env = {k: v for k, v in os.environ.items() if k not in ("CODE_DIR", "EXPECTED_SHA")}
    r = subprocess.run(["bash", str(REPO / "scripts" / "slurm" / "check_checkpoint.sbatch"), "/R/run", "checkpoint.pt", f"{root}/out/g.json"],
                       env={**env, "WORK": str(root)}, capture_output=True, text=True)
    assert "VENV_PYTHON scripts/check_checkpoint.py /R/run checkpoint.pt" in r.stdout, r.stdout + r.stderr


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


def test_dummy_node_launcher_checks_inputs_and_writes_its_output(work, tmp_path):  # noqa: F811
    """Il nodo fittizio del collaudo w1_valutazione_v1: rifiuta uscite dentro il codice, esce con 3 se manca un ingresso, scrive l'uscita."""
    import os
    import subprocess

    root = work[0]
    env = {**{k: v for k, v in os.environ.items() if k not in ("CODE_DIR", "EXPECTED_SHA")}, "WORK": str(root)}
    launcher = str(REPO / "scripts" / "slurm" / "chain_dummy.sbatch")

    def run(*a):
        return subprocess.run(["bash", launcher, *a], env=env, capture_output=True, text=True)

    assert run(str(root / "wearusfm" / "x.json")).returncode == 2
    present = tmp_path / "in.json"
    present.write_text("{}")
    assert run(str(tmp_path / "out" / "a.json"), str(present), str(tmp_path / "manca")).returncode == 3
    ok = run(str(tmp_path / "out" / "b.json"), str(present))
    assert ok.returncode == 0 and json.loads((tmp_path / "out" / "b.json").read_text())["inputs"] == 1
