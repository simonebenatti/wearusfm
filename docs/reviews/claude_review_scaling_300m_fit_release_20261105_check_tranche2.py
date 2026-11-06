"""Review 9, aggiunta: ricevute dei fit inviati dal processo locale di GPT contro gli argv pubblicati (sola lettura)."""
import csv, json, pathlib, shlex, sys
co = pathlib.Path(sys.argv[1])          # checkout con piano e ricevute
registry = pathlib.Path(sys.argv[2])    # ~/wearusfm_local/jobs
F = co / "results/infra/scaling_300m_20261104/cpu_fit_release_20261105"
plan = json.loads((F / "submission_plan.json").read_text())
ledger = list(csv.DictReader((registry / "ledger.tsv").open(), delimiter="\t"))
ids = {}
for j in plan["jobs"]:
    st = registry / j["id"] / "state.json"
    if not st.exists():
        print(f"{j['id']}: non inviato"); continue
    s = json.loads(st.read_text())
    a = j["argv"]
    exp = ["sbatch", "--parsable", "--job-name=" + j["id"], "--comment=wearusfm:" + j["id"]]
    if j["depends_on"]:
        exp.append("--dependency=afterok:" + str(ids[j["depends_on"]]))
    exp += a[a.index("--") + 1:]
    rows = [r for r in ledger if r["id"] == j["id"]]
    local = F / "submissions" / (j["id"] + ".json")
    ok = (s["sbatch"] == exp and s["stato"] == "sottomesso" and s["script_stdin_sha256"] == plan["dispatcher"]["sha256"]
          and s["agente"] == "gpt" and rows[-1]["job_slurm"] == str(s["job_slurm"]) and shlex.split(rows[-1]["comando"]) == exp
          and local.exists() and json.loads(local.read_text()) == s)
    ids[j["id"]] = s["job_slurm"]
    t = [x for x in s["sbatch"] if x.startswith("--time=")][0]
    d = [x for x in s["sbatch"] if x.startswith("--dependency")]
    print(f"{j['id']}: job {s['job_slurm']} {t} {d} walltime piano {j['walltime_seconds']} s | ricevuta = argv pubblicato, ledger e copia locale: {ok}")
print("inviati:", len(ids), "di", len(plan["jobs"]))
