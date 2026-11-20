"""Proxy CPU (Mac) dei costi lato host di UN probe dell'osservatore con il modello di produzione (T100-G 30M).
Non misura forward/backward (GPU). Misura: costruzione WearUsFM+testa, load_state_dict studente/teacher,
make_teacher, impronte state_sha256 di studente e teacher, copia+impronta+norma double del gradiente.
"""
import json, sys, time
from pathlib import Path
import torch
clone = Path(sys.argv[1])
from contrastive_prepare import BASELINE, build_review_configs, config_from_dict
from wearusfm.model.fm import WearUsFM
from wearusfm.training.contrastive_objectives import attach_contrastive_head
from wearusfm.training.jepa import make_teacher
from wearusfm.training.pairing import state_sha256
raw = build_review_configs(json.loads(Path(BASELINE).read_text()))['calibration']
cfg = config_from_dict(raw)
torch.manual_seed(0)
live = attach_contrastive_head(WearUsFM(cfg.model))
teacher_live = make_teacher(live)
n = sum(p.numel() for p in live.parameters())
s_state = {k: v.detach().clone() for k, v in live.state_dict().items()}
t_state = {k: v.detach().clone() for k, v in teacher_live.state_dict().items()}
grads = {k: torch.randn_like(p) for k, p in live.named_parameters()}
rows = []
for rep in range(3):
    t = {}
    a = time.perf_counter(); m = attach_contrastive_head(WearUsFM(cfg.model)); t['build'] = time.perf_counter() - a
    a = time.perf_counter(); m.load_state_dict(s_state, strict=True); tt = make_teacher(m); tt.load_state_dict(t_state, strict=True); t['load+teacher'] = time.perf_counter() - a
    a = time.perf_counter(); state_sha256(m.state_dict()); state_sha256(tt.state_dict()); t['fingerprint_student_teacher'] = time.perf_counter() - a
    a = time.perf_counter()
    g = {k: v.detach().to('cpu', copy=True) for k, v in grads.items()}
    state_sha256(g); bool(all(torch.isfinite(v).all() for v in g.values()))
    sum(float(torch.linalg.vector_norm(v.double())) ** 2 for v in g.values())
    t['gradient_copy_hash_norm'] = time.perf_counter() - a
    t['total'] = sum(t.values())
    rows.append(t)
    del m, tt, g
print('parameters', n)
for r in rows:
    print({k: round(v, 3) for k, v in r.items()})
