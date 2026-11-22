"""Prova (sola lettura sull'albero congelato): quali percorsi entrano in provenance.source_manifest.
Scrive solo sotto la cartella passata come argomento (radice finta)."""
import json, shutil, sys
from pathlib import Path
from pose_temponet import provenance as P

TREE = Path("/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244")
out = {}
m = P.source_manifest(TREE)
out["tree_files"] = len(m)
out["tree_source_sha256"] = P.digest(m)
plan = json.loads((TREE / "results/infra/pose_temponet_execution_20261121_A1/plan_controls.json").read_text())
out["plan_source_sha256"] = plan["reader"]["source_sha256"]
out["tree_manifest_equals_plan_manifest"] = (m == plan["reader"]["source_manifest"])

fake = Path(sys.argv[1]).resolve()
if fake.exists():
    shutil.rmtree(fake)
for rel in ("src/pkg/a.py", "scripts/top.py", "scripts/pose_temponet/x.py", "scripts/slurm/lib_code_dir.sh",
            "scripts/slurm/pose_temponet.sbatch", "scripts/slurm/campaign_checkout.sbatch"):
    p = fake / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("# " + rel + "\n")
base = P.source_manifest(fake)
out["fake_base_files"] = sorted(base)
cases = {
    "nuovo pacchetto scripts/tcn_diag/__init__.py + driver.py": ["scripts/tcn_diag/__init__.py", "scripts/tcn_diag/driver.py"],
    "nuovo launcher scripts/slurm/tcn_diag.sbatch": ["scripts/slurm/tcn_diag.sbatch"],
    "nuovo launcher scripts/slurm/pose_temponet_diag.sbatch": ["scripts/slurm/pose_temponet_diag.sbatch"],
    "nuovo file al primo livello scripts/tcn_diag_driver.py": ["scripts/tcn_diag_driver.py"],
    "nuovo file src/wearusfm_new/mod.py": ["src/wearusfm_new/mod.py"],
    "nuovo file scripts/pose_temponet/diag.py": ["scripts/pose_temponet/diag.py"],
    "nuovo file scripts/pose_temponet/notes.txt": ["scripts/pose_temponet/notes.txt"],
    "nuovo file scripts/slurm/other.sh": ["scripts/slurm/other.sh"],
    "nuovo file docs/x.md e results/y.json e tests/cpu/test_z.py": ["docs/x.md", "results/y.json", "tests/cpu/test_z.py"],
}
res = {}
for name, rels in cases.items():
    made = []
    for rel in rels:
        p = fake / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# new\n")
        made.append(p)
    now = P.source_manifest(fake)
    res[name] = {"cambia_manifest": now != base, "aggiunti": sorted(set(now) - set(base))}
    for p in made:
        p.unlink()
    # rimuovi cartelle vuote create
    for rel in rels:
        d = (fake / rel).parent
        while d != fake and d.exists() and not any(d.iterdir()):
            d.rmdir(); d = d.parent
    assert P.source_manifest(fake) == base
out["casi"] = res
print(json.dumps(out, indent=1, ensure_ascii=False))
