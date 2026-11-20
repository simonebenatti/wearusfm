"""Cross-check: historical compare_resume (438c8d65) vs historical_numeric_pair (fbfde797) on the same CPU checkpoints."""
import importlib.util, json, sys, tempfile
from pathlib import Path
C = Path(sys.argv[1]); EXP = [Path(p) for p in sys.argv[2:]]
sys.path[:0] = [str(C / 'src'), str(C / 'scripts')]
import torch
spec = importlib.util.spec_from_file_location('readout_438', str(Path(__file__).resolve().parents[1] / 'r2_equiv/scripts/readout_438.py'))
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
import contrastive_target_resume as N
from contrastive_prepare import config_from_dict
from wearusfm.training.pairing import state_sha256
tmp = Path(tempfile.mkdtemp(dir=str(Path(__file__).resolve().parents[1])))
for exp in EXP:
    cfg = config_from_dict(json.loads((exp / 'continuous/config.json').read_text()))
    initial = N.checkpoint(exp / 'calibration/checkpoint.pt')['student']
    isha = state_sha256(initial)
    with torch.random.fork_rng(devices=[]):
        model = N.attach_contrastive_head(N.WearUsFM(cfg.model))
    params = {n for n, _ in model.named_parameters()}
    finals, patched = {}, {}
    for ph in N.PHASES:
        st = torch.load(str(exp / ph / 'checkpoint.pt'), map_location='cpu', weights_only=False)
        finals[ph] = st
        st2 = dict(st); st2['step'] = 576
        patched[ph] = tmp / f'{exp.name}_{ph}.pt'; torch.save(st2, patched[ph])
    for a, b in [('continuous', 'reload_a'), ('continuous', 'reload_b'), ('reload_a', 'reload_b')]:
        try:
            o = old.compare_resume(patched[a], patched[b], cfg, isha, initial)
            ores = (o['passed'], {k: float(v) for k, v in o['relative_to_full_displacement'].items()}, sorted(o['floating_state_errors']))
        except ValueError as e:
            ores = ('RAISED', str(e))
        try:
            n = N.historical_numeric_pair(finals[a], finals[b], initial, params)
            nres = (n['passed'], {k: float(v) for k, v in n['relative_to_left_displacement_from_initial'].items()}, sorted(n['floating_state_errors']))
        except ValueError as e:
            nres = ('RAISED', str(e))
        same = ores == nres
        print(exp.name, a + '/' + b, 'old=', ores, 'new=', nres, 'IDENTICAL' if same else 'DIFF')
for p in tmp.iterdir(): p.unlink()
tmp.rmdir()
