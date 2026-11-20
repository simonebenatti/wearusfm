"""Apply exact, unique producer-side schema mutations in a throwaway clone."""
import sys
from pathlib import Path
root = Path(sys.argv[1])
MUTS = {
 "L1_parity_summary_key": ("scripts/contrastive_target_parity.py",
   '"summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest() if summary_path.exists() else None,',
   '"summary_digest": hashlib.sha256(summary_path.read_bytes()).hexdigest() if summary_path.exists() else None,'),
 "L2_numeric_phase_key": ("scripts/contrastive_run.py",
   '"phase": release.get("phase"),',
   '"phase_label": release.get("phase"),'),
 "L3_makejob_drop_review": ("scripts/contrastive_release.py",
   'id=cell_id, package_review=binding, review=binding["review"], out=str(out), arm=cell.get("arm"))',
   'id=cell_id, package_review=binding, out=str(out), arm=cell.get("arm"))'),
 "L6_readout_compare_ranks": ("scripts/contrastive_benchmark_readout.py",
   "sha(Path(folder)/'checkpoint_step0000544.pt'), ranks=4)",
   "sha(Path(folder)/'checkpoint_step0000544.pt'), ranks=1)"),
}
for name in sys.argv[2:]:
    rel, old, new = MUTS[name]
    path = root / rel
    text = path.read_text()
    assert text.count(old) == 1, (name, text.count(old))
    path.write_text(text.replace(old, new))
    print("applied", name, rel)
