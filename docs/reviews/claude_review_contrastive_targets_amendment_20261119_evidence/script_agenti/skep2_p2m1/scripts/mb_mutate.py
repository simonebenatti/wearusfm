"""MB: widen the historical compatibility whitelist to the current tree (scenario only).

Usage: python -B mb_mutate.py <root> edit     -> rewrite the helper's allowed sets once
       python -B mb_mutate.py <root> manifest -> regenerate manifest current inventory (run after ALL code edits)
"""
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
helper = root / "scripts/contrastive_source_compat.py"
if sys.argv[2] == "edit":
    changed = ['scripts/contrastive_benchmark_readout.py', 'scripts/contrastive_cuda_check.py', 'scripts/contrastive_job.py',
               'scripts/contrastive_prepare.py', 'scripts/contrastive_run.py', 'scripts/contrastive_source_compat.py',
               'scripts/train_jepa.py', 'src/wearusfm/data/pretraining_loader.py', 'src/wearusfm/harness/backbone_depth.py',
               'src/wearusfm/harness/finetuning.py', 'src/wearusfm/harness/fm_features.py', 'src/wearusfm/harness/probe.py',
               'src/wearusfm/model/anchor_targets.py', 'src/wearusfm/model/spectral_branches.py',
               'src/wearusfm/model/spectral_step1.py', 'src/wearusfm/model/spectral_targets.py',
               'src/wearusfm/training/run.py', 'src/wearusfm/training/spectral_calibration.py']
    text = helper.read_text()
    old_allowed = ('ALLOWED_CHANGED_SOURCES = frozenset({\n'
                   '    "scripts/contrastive_cuda_check.py", "scripts/contrastive_run.py",\n'
                   '    "scripts/contrastive_benchmark_readout.py", "scripts/contrastive_prepare.py", HELPER_PATH,\n'
                   '})\n')
    assert text.count(old_allowed) == 1
    text = text.replace(old_allowed, "ALLOWED_CHANGED_SOURCES = frozenset(" + repr(changed) + ")\n")
    old_added = '_require(set(current) - set(old) == {HELPER_PATH} and not set(old) - set(current),'
    assert text.count(old_added) == 1
    text = text.replace(old_added, '_require(set(current) - set(old) == {HELPER_PATH, "src/wearusfm/harness/backbone_depth.py", '
                                   '"src/wearusfm/harness/finetuning.py"} and not set(old) - set(current),')
    helper.write_text(text)
    print("helper widened")
else:
    code = ("import sys,json;sys.path[:0]=[sys.argv[1]+'/src',sys.argv[1]+'/scripts'];import contrastive_source_compat as S;"
            "inv=S._source_inventory();print(json.dumps({'inv':inv,'digest':S._inventory_digest(inv)}))")
    data = json.loads(subprocess.check_output([sys.executable, "-B", "-c", code, str(root)], text=True))
    path = root / "results/infra/contrastive_comparison_20261116/compatibility_20261118/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["current_sources"] = data["inv"]
    manifest["current_source_sha256"] = data["digest"]
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("manifest regenerated", data["digest"])
