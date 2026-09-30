# ANALISI DESCRITTIVA LOCALE, NON CONGELATA (D5a resta com'e'): CPU (Mac) contro A100: accordo dei codici RVQ sugli stessi ingressi del run vero di V1-V4.
# Richiede l'ambiente ~/.venvs/wearusfm-tok (torch CPU, einops, pyyaml), il clone di NeuroRVQ (commit 926e770) e il
# checkpoint NeuroRVQ_EMG_tokenizer_v1.pt (sha256 0d255bcc...dce49) e gli array del run 59048369. I percorsi
# in testa allo script sono quelli dello scratchpad della sessione: vanno adattati.
import sys, time, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, '/Users/simonebenatti/dev/wearusfm/src')
from wearusfm.tokenizer_checks.neurorvq import NeuroRVQRunner
S = Path('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/57944b46-716a-4677-b1f7-a66f65922403/scratchpad')
runner = NeuroRVQRunner(S / 'neurorvq_repo', S / 'models' / 'NeuroRVQ_EMG_tokenizer_v1.pt', device='cpu')
print('checkpoint caricato; quantizzatori inizializzati OK (il controllo interno non ha sollevato)', flush=True)
torch.set_num_threads(8)
out = {}
for name in ['emg2pose', 'capgmyo', 'putemg']:
    z = np.load(S / 'step1bis_arrays' / f'{name}.npz')
    tokens, codes_gpu = z['tokens'], z['codes']            # (n_tok, 200), (4, 16, n_tok)
    n_win = tokens.shape[0] // 16
    wins = tokens.reshape(n_win, 16 * 200).astype(np.float32)
    sel = np.arange(0, n_win, 6)[:200]                       # 200 finestre per dataset: basta per stimare la frazione
    t = time.time()
    res = []
    for i in range(0, len(sel), 50):
        r = runner.run(wins[sel[i:i + 50], None, :], [0], want_recon=False)
        res.append(r['codes'])
    cpu = np.concatenate(res, axis=2)                       # (4, 16, len(sel)*16)
    tok_idx = np.concatenate([np.arange(w * 16, w * 16 + 16) for w in sel])
    gpu = codes_gpu[:, :, tok_idx]
    same = (cpu == gpu).mean(axis=2)                        # (4, 16)
    out[name] = same.tolist()
    print(f'{name}: {len(sel)} finestre in {time.time()-t:.0f}s | livello 0 per ramo {np.round(same[:,0],3)} | livello 3 {np.round(same[:,3],3)} | livello 8 {np.round(same[:,8],3)} | livello 15 {np.round(same[:,15],3)}', flush=True)
json.dump(out, open(S / 'analysis' / 'exp1_cpu_vs_gpu.json', 'w'))
