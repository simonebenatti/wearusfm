# ANALISI DESCRITTIVA LOCALE, NON CONGELATA (D5a resta com'e'): stabilita' delle feature CONTINUE prima della quantizzazione sotto il rumore di V4.
# Stesso ambiente e stessi input di step1bis_margins.py: vedi results/step1bis/NOTA_TOKENIZER_LOCALE.md.
"""Stabilita' delle FEATURE continue prima della quantizzazione, sotto lo stesso rumore di V4, contro la variazione naturale fra token."""
import sys, json
from pathlib import Path
import numpy as np, torch
from einops import rearrange
sys.path.insert(0, '/Users/simonebenatti/dev/wearusfm/src')
from wearusfm.tokenizer_checks.neurorvq import NeuroRVQRunner
S = Path('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/57944b46-716a-4677-b1f7-a66f65922403/scratchpad')
runner = NeuroRVQRunner(S / 'neurorvq_repo', S / 'models' / 'NeuroRVQ_EMG_tokenizer_v1.pt', device='cpu')
m = runner.model; torch.set_num_threads(8)
rep = json.load(open('/Users/simonebenatti/dev/wearusfm/results/step1bis/step1bis_59048369.json'))

def feats(x):
    b = x.shape[0]
    t_ix, s_ix = runner._indices([0], 16, b)
    x4 = rearrange(torch.from_numpy(x), 'B N (A T) -> B N A T', T=200).contiguous()
    with torch.no_grad():
        f = m.encoder(x4, temporal_embedding_ix=t_ix, spatial_embedding_ix=s_ix, return_patch_tokens=True)
        return [getattr(m, f'encode_task_layer_{k+1}')(f[k].float()).reshape(-1, 128).numpy() for k in range(4)]

def cos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)

for name in ['emg2pose', 'capgmyo', 'putemg']:
    z = np.load(S / 'step1bis_arrays' / f'{name}.npz')
    wins = z['tokens'].reshape(-1, 3200).astype(np.float32)
    sel = np.arange(0, len(wins), 6)[:200]
    X = wins[sel][:, None, :]
    floor = rep['v4_per_dataset'][name]['noise_floor_rms']
    rng = np.random.default_rng(0)
    out = {}
    for label, sigma in (('rumore = soglia (V4)', floor), ('rumore = 0,1 x soglia', 0.1 * floor)):
        Xn = X + rng.normal(scale=sigma, size=X.shape).astype(np.float32)
        Fc, Fn = feats(X), feats(Xn)
        out[label] = [cos(Fc[k], Fn[k]) for k in range(4)]
    Fc = feats(X)
    perm = np.random.default_rng(1).permutation(len(Fc[0]))
    ref = [cos(Fc[k], Fc[k][perm]) for k in range(4)]   # coseno fra token DIVERSI: quanto sono gia' simili fra loro
    print(f'\n{name}')
    for label, cs in out.items():
        print(f'  {label:24s} coseno feature pulite-vs-rumorose, mediana per ramo: {np.round([np.median(c) for c in cs],3)} | 5% piu\' basso: {np.round([np.quantile(c,0.05) for c in cs],3)}')
    print(f'  coseno fra token DIVERSI (riferimento):          mediana per ramo: {np.round([np.median(c) for c in ref],3)} | 95%: {np.round([np.quantile(c,0.95) for c in ref],3)}')
