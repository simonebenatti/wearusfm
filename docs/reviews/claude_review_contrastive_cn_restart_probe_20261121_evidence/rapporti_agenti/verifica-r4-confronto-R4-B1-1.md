# Verifica avversaria di R4-B1 (cuda_rng di primo livello nel confronto della riscaldata)

Esito: il rilievo REGGE, gravita' bloccante confermata. Non sono riuscito a confutarlo.

## Che cosa ho dedotto dal codice (file:riga, clone a 9d34dc69)
- run.py:1383 `torch.manual_seed(cfg.seed)` (seed=0 nel config.json archiviato): manual_seed_all, tutti i dispositivi visibili a seme 0.
- run.py:1446-1450 (ripresa DDP): ripristina CPU e solo `torch.cuda.set_rng_state(..., device)` del dispositivo del rank. Per il rank 0 i dispositivi 1..3 restano a seme 0, offset 0.
- run.py:1461-1463 (solo senza ripresa): `torch.manual_seed(cfg.seed*1_000_003 + rank + 1)`: per il rank 0 tutti i dispositivi a seme 1.
- Il continuo storico parte da zero (metrics.jsonl inizia al passo 1, summary steps 576): passa da 1463.
- run.py:1482 il payload salva `cuda_rng = torch.cuda.get_rng_state_all()` dal rank 0; sbatch `--gres=gpu:4`, torchrun senza maschera per rank (contrastive_cn_restart.py:130): il rank 0 vede 4 dispositivi.
- Quindi a 576 `cuda_rng[1..3]` della riscaldata (seme 0) differisce sempre da quello del continuo (seme 1), qualunque sia la traiettoria. Vale anche col seeding pigro di torch (il tracker conserva l'ultimo seed_all).
- compare.py:241 `checkpoint_differences(checkpoint(ref), state)`; `checkpoint()` (contrastive_target_resume.py:55-56) non filtra campi; tensori uint8 non in virgola mobile -> `metadata_or_schema_different` (compare.py:54-55); compare.py:248 `require(not ...metadata_or_schema_different)` -> ValueError -> main() compare.py:317-318 scrive INCONCLUSIVE. La riga 248 sta prima di tutte le letture per rank e per tensore: nessuna lettura scientifica, nemmeno HYPOTHESIS_UNSUPPORTED.
- Nessun altro controllo lo intercetta o lo neutralizza. Il codice storico lo sapeva: contrastive_target_resume.py:262-265 e contrastive_benchmark_readout.py:187 confrontano `rank_rng` ed escludono torch_rng/cuda_rng di primo livello. Il test del diff (tests/cpu/test_cn_restart_compare.py:41-50) usa payload senza cuda_rng di primo livello.
- L'ordinaria non ne soffre (due riprese, stesso seeding).

## Eseguito (simulazione dichiarata)
`agents/verifica-r4-confronto-R4-B1-1/sim.py`: funzione vera `checkpoint_differences` del clone, stato CUDA simulato come 16 byte uint8 (seed u64 + offset u64, formato Philox di torch 2.2; niente CUDA sul Mac), dispositivo 0 e rank_rng identici, dispositivi 1..3 a seme 1 (continuo) contro 0 (riscaldata):
```
{'floating_tensors_different': [], 'metadata_or_schema_different': ['cuda_rng.1', 'cuda_rng.2', 'cuda_rng.3'], 'exact_payload': False}
riga 248 solleverebbe: warm endpoint metadata/schema or nonfloat state differs
True   (ordinaria: due riprese -> exact_payload)
```

## Attenuante (non cambia la gravita')
Le uscite delle gambe restano nella radice del run: un comparatore corretto potrebbe essere rieseguito su CPU senza GPU, ma richiederebbe nuova review/release; il report ufficiale del tentativo unico sarebbe INCONCLUSIVE.

## Correzione minima
In `compare()` per la sola riscaldata: confrontare `cuda_rng` di primo livello solo all'indice 0 (dispositivo del rank 0, ridondante con rank_rng[0].cuda_rng) e riportare 1..3 a parte con le impronte, escludendoli da `metadata_or_schema_different`; torch_rng di primo livello puo' restare (coincide con rank_rng[0].torch_rng se la traiettoria coincide). Ordinaria invariata (confronto integrale + SHA). Test: payload completo in cui differiscono solo cuda_rng[1..3] -> riscaldata exact_payload vero, campi riportati a parte.
