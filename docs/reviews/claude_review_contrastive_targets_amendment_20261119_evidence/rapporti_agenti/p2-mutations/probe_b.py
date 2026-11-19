"""P2 probe: try every historical artifact against the current (100/125) production readers.

Read-only on the clone: copies/forgeries are written only under TMP.
Usage: python -B probe_b.py <clone> <tmpdir>
"""
import copy, glob, gzip, hashlib, json, os, shutil, sys, traceback
from pathlib import Path

CLONE, TMP = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
TMP.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(CLONE / 'src'), str(CLONE / 'scripts')]
import contrastive_prepare as P
import contrastive_source_compat as S
import contrastive_run as RUN
import contrastive_job as J
import contrastive_benchmark_readout as B

H = CLONE / 'results/infra/contrastive_comparison_20261116'
CAL = H / 'calibration_completed_20261117'
NEW = CLONE / 'results/infra/contrastive_targets_20261118'
out = {}


def attempt(name, fn):
    try:
        r = fn()
        out[name] = {'accepted': True, 'result': (r if isinstance(r, (str, int, float, bool)) else str(type(r)))}
    except Exception as e:  # noqa: BLE001 - we record the rejection reason
        out[name] = {'accepted': False, 'error': type(e).__name__ + ': ' + str(e)[:240]}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canon(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


current = P.source_digest()
pre = json.loads((NEW / 'preflight_plan.json').read_text())
out['source_digest_current'] = current
out['preflight_source_matches_current'] = pre['source_sha256'] == current
new_cfgs = P.build_review_configs(P._read(P.BASELINE))
new_cal = new_cfgs['calibration']
hist_cal_raw = P._read(H / 'configs/calibration.config.json')
out['new_cal_windows'] = [new_cal['loader']['rms_window_ms'], new_cal['loader']['ms_fast_window_ms'], new_cal['target_version']]

# ---------- 1. historical audit (job 64652559, TIMEOUT exception) ----------
audit = TMP / 'hist_audit.json'
audit.write_bytes(gzip.decompress((H / 'audit_completed_20261117/audit.json.gz').read_bytes()))
manifest = json.loads((H / 'compatibility_20261118/manifest.json').read_text())
out['hist_audit_sha_equals_compat_pin'] = sha_file(audit) == manifest['artifacts']['audit']['sha256']
ar = json.loads(audit.read_text())
out['hist_audit_config_sha_equals_hist_cal_config'] = ar['config_sha256'] == canon(hist_cal_raw)
out['hist_audit_config_sha_equals_new_cal_config'] = ar['config_sha256'] == canon(new_cal)
attempt('1a compat_guard(audit)', lambda: S.validate_historical_evidence('audit', audit))
dummy = {}
for name in ('manifest', 'scales', 'window_seconds', 'pose_stats'):
    p = TMP / ('dummy_' + name + '.json'); p.write_text('{"%s": 1}\n' % name); dummy[name] = p
attempt('1b validate_audit(hist audit, new calibration)', lambda: RUN.validate_audit(audit, dummy, new_cal))
# Forgery that bypasses the source guard: claims current source and dummy inputs.
forged = dict(ar); forged['source_sha256'] = current
forged['inputs'] = {k: sha_file(v) for k, v in dummy.items()}
fa = TMP / 'forged_audit_current_source.json'; fa.write_text(json.dumps(forged))
attempt('1c validate_audit(relabelled source, new calibration)', lambda: RUN.validate_audit(fa, dummy, new_cal))
attempt('1d CONTROL validate_audit(relabelled source, historical 25/50 calibration)', lambda: bool(RUN.validate_audit(fa, dummy, hist_cal_raw)))

# ---------- 2. historical calibration (lambda 0.1194 / 0.3304) ----------
receipt = P._read(CAL / 'calibration_report.json')
out['hist_receipt_weights'] = receipt['weights']
out['hist_receipt_canonical_sha_equals_compat_pin'] = hashlib.sha256(S._canonical_json(receipt)).hexdigest() == manifest['artifacts']['calibration_receipt']['canonical_sha256']
attempt('2a compat_guard(calibration_run)', lambda: S.validate_historical_evidence('calibration_run', CAL / 'run'))
attempt('2b compat_guard(calibration_receipt canonical)', lambda: S.validate_historical_evidence('calibration_receipt', receipt))
attempt('2c calibration_report(hist run)', lambda: P.calibration_report(CAL / 'run'))
attempt('2d authenticate_calibration(hist receipt, hist run)', lambda: P.authenticate_calibration(receipt, CAL / 'run'))
attempt('2e calibrated_configs(hist receipt)', lambda: P.calibrated_configs(receipt, CAL / 'run'))
# Relabel only the config (100/125) keeping historical provenance: must hit the guard.
rel = TMP / 'relabel_config_run'; shutil.copytree(CAL / 'run', rel)
(rel / 'config.json').write_text(json.dumps(new_cal, indent=2))
summ = P._read(rel / 'summary.json')
summ['contrastive']['config'] = new_cal['contrastive']; (rel / 'summary.json').write_text(json.dumps(summ))
attempt('2f calibration_report(config relabelled to 100/125, old provenance)', lambda: P.calibration_report(rel)['status'])
# Deliberate full forgery (config + self-declared source): reader only sees file contents.
rel2 = TMP / 'relabel_config_and_source_run'; shutil.copytree(rel, rel2)
s2 = P._read(rel2 / 'summary.json'); s2['contrastive']['provenance']['source_sha256'] = current
(rel2 / 'summary.json').write_text(json.dumps(s2))
def forged_full():
    r = P.calibration_report(rel2); return r['status'] + ' weights=' + json.dumps(r['weights'])
attempt('2g calibration_report(config AND source relabelled: deliberate forgery)', forged_full)

# ---------- 3. releases and job files (bf16_20261118 and older) ----------
for f in sorted(glob.glob(str(H / 'releases/**/*.job.json'), recursive=True)):
    job = json.loads(Path(f).read_text()); mode = job.get('mode')
    kind = 'gpu' if mode in J.GPU_MODES else 'cpu'
    attempt('3a build_command(' + str(Path(f).relative_to(H)) + ')',
            lambda job=job, kind=kind, mode=mode: J.build_command(job, kind, J.ALLOCATED_SECONDS.get(mode)))
for f in sorted(glob.glob(str(H / 'releases/**/*.release.json'), recursive=True)):
    rl = json.loads(Path(f).read_text())
    attempt('3b validate_release(' + str(Path(f).relative_to(H)) + ')',
            lambda rl=rl: RUN.validate_release(rl, rl.get('stage'), H / 'calibrated_cohort_20261118_bf16/C0.config.json', dummy,
                                               H / 'calibration_completed_20261117/calibration_report.json', rl.get('time_limit_s'),
                                               audit_path=audit))
# A brand-new calibration release (new source/config) pinning the OLD audit.  Data
# bindings are target independent and the real manifests are only on Leonardo,
# so validate_input_bindings is neutralised for this probe only.
RUN.validate_input_bindings = lambda pins: None
calcfg = TMP / 'new_calibration.config.json'; calcfg.write_text(json.dumps(new_cal, indent=2))
newrel = {'status': 'APPROVED_FOR_PREPARATION', 'stage': 'calibration', 'source_sha256': current,
          'config_sha256': sha_file(calcfg), 'world_size': 4, 'workers_per_rank': 6,
          'inputs': {k: sha_file(v) for k, v in dummy.items()}, 'review_pin': 'x', 'time_limit_s': 1770,
          'audit_sha256': sha_file(audit)}
attempt('3c validate_release(NEW calibration release pinning OLD audit)',
        lambda: RUN.validate_release(newrel, 'calibration', calcfg, dummy, None, 1770, audit_path=audit))
# Synthetic NEW audit (as if a new 100/125 audit had passed) + NEW benchmark release
# pinning the OLD calibration receipt and OLD bf16 cohort.
na = dict(ar); na['source_sha256'] = current; na['inputs'] = newrel['inputs']; na['config_sha256'] = canon(new_cal)
nap = TMP / 'synthetic_new_audit.json'; nap.write_text(json.dumps(na))
attempt('3d CONTROL validate_audit(synthetic new audit)', lambda: bool(RUN.validate_audit(nap, dummy, new_cal)))
cohort = H / 'calibrated_cohort_20261118_bf16/cohort.json'
c0 = H / 'calibrated_cohort_20261118_bf16/C0.config.json'
benchrel = {'status': 'APPROVED_FOR_PREPARATION', 'stage': 'benchmark', 'phase': 'full', 'arm': 'C0', 'source_sha256': current,
            'config_sha256': sha_file(c0), 'world_size': 4, 'workers_per_rank': 6, 'inputs': newrel['inputs'],
            'review_pin': 'x', 'time_limit_s': 1770, 'audit_sha256': sha_file(nap),
            'calibration_sha256': sha_file(CAL / 'calibration_report.json'), 'cohort_sha256': sha_file(cohort)}
attempt('3e validate_release(NEW benchmark release, new audit, OLD receipt+bf16 cohort)',
        lambda: RUN.validate_release(benchrel, 'benchmark', c0, dummy, CAL / 'calibration_report.json', 1770,
                                     audit_path=nap, cohort_path=cohort, calibration_run_dir=CAL / 'run'))
attempt('3f validate_cohort(bf16 cohort)', lambda: RUN.validate_cohort(cohort, CAL / 'calibration_report.json', CAL / 'run', c0))
attempt('3g validate_recipe(bf16 C0 config, calibration stage)', lambda: RUN.validate_recipe(P._read(c0), 'calibration'))
attempt('3h validate_recipe(hist calibration config)', lambda: RUN.validate_recipe(P._read(CAL / 'run/config.json'), 'calibration'))

# ---------- 4. benchmarks, CUDA report, canonical preflight receipt ----------
bdir = TMP / 'bench_C0'; shutil.copytree(H / 'benchmarks_completed_bf16_20261118/benchmark/C0', bdir)
(bdir / 'metrics.jsonl').write_bytes(gzip.decompress((bdir / 'metrics.jsonl.gz').read_bytes()))
# Most favourable expected config for the old run: new recipe with the OLD lambdas.
exp = copy.deepcopy(new_cfgs['C0']); exp['max_steps'] = 576
exp['contrastive'].update(nce_weight=receipt['weights']['nce'], rnc_weight=receipt['weights']['rnc'])
hcfg = P._read(bdir / 'config.json')
out['hist_bench_C0_windows'] = [P.R.with_legacy_target_windows(hcfg)['loader'].get(k) for k in ('rms_window_ms', 'ms_fast_window_ms')] + [hcfg.get('target_version')]
pcohort = {'initial_state_sha256': receipt['initial_state_sha256'], 'source_sha256': current}
attempt('4a _run_evidence(bf16 C0 benchmark, new C0 recipe)',
        lambda: B._run_evidence(bdir, 'C0', pcohort, exp, {'source_sha256': current}))
attempt('4b _cuda_evidence(bf16 cuda.json, new cohort/source)',
        lambda: B._cuda_evidence(H / 'cuda_completed_bf16_20261118/cuda.json', pcohort, receipt['input_sha256'], new_cal))
ev = P._read(H / 'benchmarks_completed_bf16_20261118/preflight_readout.json')
out['bf16_preflight_status'] = [ev.get('status'), ev.get('failure_reasons', [])[:3]]
paths = {'cohort_path': cohort, 'full_dirs': {a: H / 'benchmarks_completed_bf16_20261118/benchmark' / a for a in B.ARMS},
         'resumed_dirs': {a: H / 'benchmarks_completed_bf16_20261118/resume' / a for a in ('CN', 'CR')},
         'cuda_path': H / 'cuda_completed_bf16_20261118/cuda.json', 'cpu_audit_path': audit,
         'calibration_path': CAL / 'calibration_report.json', 'calibration_run_dir': CAL / 'run'}
attempt('4c read_preflight(all bf16_20261118 evidence, local copies)', lambda: B.read_preflight(**paths))
# The archived receipt itself, as the training gate would read it.
pr = TMP / 'preflight_readout_localpaths.json'
ev2 = dict(ev); ev2['evidence_paths'] = {k: (str(v) if not isinstance(v, dict) else {a: str(p) for a, p in v.items()}) for k, v in paths.items()}
ev2['evidence_paths']['rank_exit_path'] = str(H / 'cuda_completed_bf16_20261118/cuda.json.rank_exit_probe/report.json')
pr.write_text(json.dumps(ev2))
attempt('4d authenticate_preflight(bf16 preflight_readout, local paths)', lambda: B.authenticate_preflight(pr))
attempt('4e authenticate_preflight(bf16 preflight_readout, archived remote paths)',
        lambda: B.authenticate_preflight(H / 'benchmarks_completed_bf16_20261118/preflight_readout.json'))

print(json.dumps(out, indent=1, default=str))
