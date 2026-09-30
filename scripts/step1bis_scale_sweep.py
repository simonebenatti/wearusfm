# ANALISI DESCRITTIVA LOCALE, NON CONGELATA (D5a resta com'e'): errore di ricostruzione (NMSE mediano) al variare della scala di ingresso.
# Richiede l'ambiente ~/.venvs/wearusfm-tok (torch CPU, einops, pyyaml), il clone di NeuroRVQ (commit 926e770) e il
# checkpoint NeuroRVQ_EMG_tokenizer_v1.pt (sha256 0d255bcc...dce49) e gli array del run 59048369. I percorsi
# in testa allo script sono quelli dello scratchpad della sessione: vanno adattati.
"""Errore di ricostruzione (NMSE mediano per token, dominio standardizzato) al variare della SCALA d'ingresso."""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/Users/simonebenatti/dev/wearusfm/src')
from wearusfm.tokenizer_checks.neurorvq import NeuroRVQRunner
from wearusfm.tokenizer_checks import metrics as M
S = Path('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/57944b46-716a-4677-b1f7-a66f65922403/scratchpad')
runner = NeuroRVQRunner(S / 'neurorvq_repo', S / 'models' / 'NeuroRVQ_EMG_tokenizer_v1.pt', device='cpu')
torch.set_num_threads(8)
mult = [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]
print('moltiplicatore rispetto alla scala scelta nel run (21,7 = scala nativa emg2pose):', mult)
res = {}
for name in ['emg2pose', 'capgmyo', 'putemg', 'camargo2021']:
    z = np.load(S / 'step1bis_arrays' / f'{name}.npz')
    wins = z['tokens'].reshape(-1, 3200).astype(np.float32)          # gia' scalate x factor
    sel = np.arange(0, len(wins), 6)[:200]
    row = []
    for mu in mult:
        x = (wins[sel] * np.float32(mu))[:, None, :]
        out = [runner.run(x[i:i + 50], [0], want_recon=True) for i in range(0, len(x), 50)]
        nm = M.token_nmse(np.concatenate([o['std_x'] for o in out]), np.concatenate([o['std_xrec'] for o in out]))
        row.append(float(np.median(nm)))
    res[name] = row
    print(f'{name:12s}', ' '.join(f'{v:.4f}' for v in row), '| minimo a x', mult[int(np.argmin(row))], flush=True)
json.dump({'multipliers': mult, 'median_nmse': res}, open(S / 'analysis' / 'exp3_scale.json', 'w'))
