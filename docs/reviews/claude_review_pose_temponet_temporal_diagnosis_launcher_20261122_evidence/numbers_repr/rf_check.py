import numpy as np, torch, json
from pose_temponet import controls as C, heads
torch.manual_seed(0)
x = torch.randn(1, 160, 384)
def rf(fn):
    base = fn(x); x2 = x.clone(); x2[0, 80] += 1.0
    d = (fn(x2)-base).abs().amax(dim=-1)[0]
    idx = torch.nonzero(d > 0).flatten()
    return int(idx.min()), int(idx.max()), int(idx.max()-idx.min()+1)
pupil = heads.make_head('TCN_post', 0).eval()
with torch.no_grad():
    for m in pupil.deep:
        if isinstance(m, torch.nn.Conv1d) and m.weight.abs().sum() == 0:
            torch.nn.init.normal_(m.weight, std=0.1)   # l'ultimo strato nasce a zero: lo riempio solo per misurare il campo
    r_pupil = rf(lambda t: pupil(t))
xs = np.random.default_rng(0).standard_normal((2, 160, 384)).astype(np.float16)
a = C.generate_components(xs, [{'user': 'u0', 'window': 0}, {'user': 'u1', 'window': 1}], np.ones((2, 148, 20), bool), np.array([0]), np.array([1]), ['u0', 'u1'], outer_fold=0)
xs2 = xs.copy(); xs2[0, 80] += 1.0
b = C.generate_components(xs2, [{'user': 'u0', 'window': 0}, {'user': 'u1', 'window': 1}], np.ones((2, 148, 20), bool), np.array([0]), np.array([1]), ['u0', 'u1'], outer_fold=0)
# confronto su T non normalizzata: rimuovo la normalizzazione
def raw(c, k): return c[k]*np.array(c['normalization'][k]['std'])+np.array(c['normalization'][k]['mean'])
dT = np.abs(raw(a, 'T')-raw(b, 'T'))[0].max(axis=-1); dL = np.abs(raw(a, 'L')-raw(b, 'L'))[0].max(axis=-1)
iT = np.flatnonzero(dT > 1e-9); iL = np.flatnonzero(dL > 1e-9)
print(json.dumps({"pupil_TCN_post_rf": r_pupil, "teacher_T_rf": [int(iT.min()), int(iT.max()), int(iT.max()-iT.min()+1)],
                  "teacher_L_rf": [int(iL.min()), int(iL.max()), int(iL.max()-iL.min()+1)],
                  "teacher_params": int(sum(v.numel() for v in a['generator_states']['T'].values())),
                  "pupil_params": sum(p.numel() for p in pupil.parameters()),
                  "pupil_deep_params": sum(p.numel() for p in pupil.deep.parameters()), "pupil_affine_params": sum(p.numel() for p in pupil.affine.parameters())}))
