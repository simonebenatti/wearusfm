# Uso: python3 scripts/step1bis_v3_extras.py <cartella_npz>   (arrays_<jobid> del run di V1-V4)
"""ANALISI DESCRITTIVA, NON CONGELATA (D5a resta com'e'): riproduce V3 dai dati salvati del run vero
e prova sonde diverse per capire QUALE parte dei codici porta l'identita' del dataset e quanto e'
debole la baseline delle 5 bande. Non cambia nessun verdetto."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from wearusfm.tokenizer_checks import metrics as M

A = Path(sys.argv[1])  # cartella con i .npz del run (fuori dal repo)
names = sorted(p.stem for p in A.glob('*.npz'))
data = {n: np.load(A / f'{n}.npz') for n in names}
codes_l, tokens_l, y, units, ds_subj = [], [], [], [], {}
for di, n in enumerate(names):
    z = data[n]
    gs = z['group_subject']
    for k, subj in enumerate(gs):
        sl = slice(k * 256, (k + 1) * 256)
        codes_l.append(z['codes'][:, :, sl]); tokens_l.append(z['tokens'][sl])
        y.append(di); units.append(f'{n}/{subj}')
    ds_subj[n] = sorted(set(gs.tolist()))
part_map = M.split_subjects(ds_subj, np.random.default_rng([0, 3, 0]))
part = np.array([part_map[(names[yy], u.split('/', 1)[1])] for yy, u in zip(y, units)])
y = np.array(y); units = np.array(units)
print('datasets', names, 'n', len(y), 'test', int((part == 'test').sum()))

def probe(X, label):
    r = M.dataset_id_probe(X, y, units, part, np.random.default_rng([0, 3, 1]))
    print(f'{label:52s} acc={r.balanced_accuracy:.3f} ci=[{r.ci95[0]:.3f},{r.ci95[1]:.3f}] model={r.model}', flush=True)
    return r

probe(M.band_power_features(tokens_l), '[congelata] 5 bande (deve dare 0.655)')
probe(M.code_histogram_features(codes_l), '[congelata] codici, tutti i rami e livelli (0.964)')

# --- baseline piu' forte: spettro fine e statistiche di ampiezza (NON congelate)
def fine_spec(tokens, lo=20, hi=400):
    p = np.mean(np.abs(np.fft.rfft(tokens.astype(np.float64), axis=-1)) ** 2, axis=0)
    f = np.fft.rfftfreq(200, 1 / 1000.0)
    sel = (f >= lo) & (f < hi)
    return np.log10(p[sel] + 1e-12)
def amp_stats(tokens):
    rms = np.sqrt(np.mean(tokens.astype(np.float64) ** 2, axis=-1))
    q = np.quantile(np.log10(rms + 1e-9), [0.05, 0.25, 0.5, 0.75, 0.95])
    return np.concatenate([q, [np.log10(rms.mean() + 1e-9), np.log10(rms.std() + 1e-9)]])
spec = np.stack([fine_spec(t) for t in tokens_l])
stats = np.stack([amp_stats(t) for t in tokens_l])
probe(spec, '[extra] spettro fine 5 Hz (76 bin, 20-400 Hz)')
probe(np.hstack([spec, stats]), '[extra] spettro fine + statistiche di ampiezza')

# --- quali codici portano l'identita' del dataset (NON congelato)
def sub(codes_list, branches=None, levels=None):
    out = []
    for c in codes_list:
        cc = c
        if branches is not None: cc = cc[list(branches)]
        if levels is not None: cc = cc[:, list(levels)]
        out.append(cc)
    return out
def hist_sub(codes_list, branches, levels):
    """istogramma sparso solo su (rami, livelli) scelti: si rimappa a un array (B', L', n) e si usa la stessa funzione
    su una versione a 4x16 con il resto azzerato ai codici 0 e poi si tolgono le colonne: piu' semplice, si costruisce a mano"""
    from scipy import sparse
    rows, cols, vals = [], [], []
    nb, nl = len(branches), len(levels)
    for i, c in enumerate(codes_list):
        cc = c[list(branches)][:, list(levels)]
        bl = (np.arange(nb)[:, None] * nl + np.arange(nl)[None, :])[..., None]
        flat = (bl * M.N_CODE + cc).ravel()
        rows.append(np.full(flat.shape, i)); cols.append(flat); vals.append(np.full(flat.shape, 1.0 / cc.shape[-1]))
    return sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(codes_list), nb * nl * M.N_CODE)).tocsr()
for label, br, lv in [('codici: solo livello 0, 4 rami', range(4), [0]),
                      ('codici: livelli 0-1, 4 rami', range(4), [0, 1]),
                      ('codici: livelli 0-3, 4 rami', range(4), range(4)),
                      ('codici: livelli 8-15, 4 rami', range(4), range(8, 16)),
                      ('codici: solo ramo 0 (>20 Hz), 16 livelli', [0], range(16)),
                      ('codici: solo ramo 3 (>250 Hz), 16 livelli', [3], range(16))]:
    probe(hist_sub(codes_l, list(br), list(lv)), '[extra] ' + label)
