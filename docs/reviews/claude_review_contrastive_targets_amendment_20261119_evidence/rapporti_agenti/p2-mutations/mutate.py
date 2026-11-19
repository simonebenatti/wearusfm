"""Apply one named set of production mutations to the TEMPORARY clone only.

Usage: python -B mutate.py <mut_clone> <comma separated mutation names>
Mutations (production code only, tests untouched):
  MA  revert the 100/125 assignment in build_review_configs (recipe stays 25/50)
  MC  calibration_report compares configs ignoring rms/fast windows and target_version
  MD  validate_audit no longer checks audit config_sha256 against the calibration recipe
  ME  calibration_report no longer calls the historical-source guard (any source accepted)
  MF  authenticate_calibration no longer calls the historical receipt guard
  MB  compatibility whitelist broadened to the current sources (manifest regenerated last)
"""
import hashlib, json, sys
from pathlib import Path

MUT = Path(sys.argv[1]).resolve()
names = [n for n in sys.argv[2].split(',') if n]
assert '/scratchpad/rev_targets/tmp/' in str(MUT), 'refusing to mutate outside the temporary area'


def replace(rel, old, new):
    path = MUT / rel
    text = path.read_text()
    assert text.count(old) == 1, (rel, old[:60], text.count(old))
    path.write_text(text.replace(old, new))


for name in names:
    if name == 'MA':
        replace('scripts/contrastive_prepare.py',
                '    base["loader"]["rms_window_ms"] = 100.\n    base["loader"]["ms_fast_window_ms"] = 125.\n    base["target_version"] = target_version_for_fast_window(125.)\n',
                '')
    elif name == 'MC':
        replace('scripts/contrastive_prepare.py',
                '    if R.config_to_dict(config_from_dict(cfg)) != expected:\n        raise ValueError("calibration run config differs from the frozen 512-batch recipe")',
                '    _strip = lambda c: {**{k: v for k, v in c.items() if k != "target_version"}, "loader": {k: v for k, v in c["loader"].items() if k not in ("rms_window_ms", "ms_fast_window_ms")}}\n'
                '    if _strip(R.config_to_dict(config_from_dict(cfg))) != _strip(expected):\n        raise ValueError("calibration run config differs from the frozen 512-batch recipe")')
    elif name == 'MD':
        replace('scripts/contrastive_run.py',
                '    if (report.get("steps") != 1024 or report.get("world_size") != 4 or report.get("workers_per_rank") != 6\n'
                '            or report.get("config_sha256") != hashlib.sha256(json.dumps(calibration_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()):',
                '    if (report.get("steps") != 1024 or report.get("world_size") != 4 or report.get("workers_per_rank") != 6):')
    elif name == 'ME':
        replace('scripts/contrastive_prepare.py',
                '        from contrastive_source_compat import validate_historical_evidence\n        validate_historical_evidence("calibration_run", folder)\n',
                '        pass\n')
    elif name == 'MF':
        replace('scripts/contrastive_prepare.py',
                '        from contrastive_source_compat import validate_historical_evidence\n        validate_historical_evidence("calibration_receipt", receipt)\n',
                '        pass\n')
    elif name == 'MG':
        replace('scripts/contrastive_benchmark_readout.py',
                "            audit.get('workers_per_rank') != 6 or audit.get('config_sha256') != canonical_calibration):",
                "            audit.get('workers_per_rank') != 6):")
    elif name == 'MH':
        replace('scripts/contrastive_prepare.py',
                '    if _jsonable(receipt) != comparable:\n        raise ValueError("calibration receipt differs from authenticated recalculation")\n',
                '')
    elif name == 'MB':
        pass
    else:
        raise SystemExit('unknown mutation ' + name)

if 'MB' in names:
    sys.path[:0] = [str(MUT / 'src'), str(MUT / 'scripts')]
    manifest_path = MUT / 'results/infra/contrastive_comparison_20261116/compatibility_20261118/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    old = {r['path']: r['sha256'] for r in manifest['historical_sources']}
    replace('scripts/contrastive_source_compat.py',
            '    _require(set(current) - set(old) == {HELPER_PATH} and not set(old) - set(current),',
            '    _require(set(current) - set(old) <= ALLOWED_CHANGED_SOURCES and not set(old) - set(current),')
    # Two passes: the helper's own bytes change when the whitelist literal changes.
    for _ in range(3):
        import importlib
        import contrastive_source_compat as C
        importlib.reload(C)
        actual = C._source_inventory()
        changed = sorted({r['path'] for r in actual if old.get(r['path']) != r['sha256']} | (set(old) - {r['path'] for r in actual}))
        text = (MUT / 'scripts/contrastive_source_compat.py').read_text()
        start = text.index('ALLOWED_CHANGED_SOURCES = frozenset({')
        end = text.index('})\n', start) + 3
        literal = 'ALLOWED_CHANGED_SOURCES = frozenset({\n' + ''.join('    %r,\n' % p for p in changed) + '})\n'
        (MUT / 'scripts/contrastive_source_compat.py').write_text(text[:start] + literal + text[end:])
    importlib.reload(C)
    actual = C._source_inventory()
    manifest['current_sources'] = actual
    manifest['current_source_sha256'] = C._inventory_digest(actual)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'applied': names}))
