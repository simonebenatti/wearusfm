# R4 - regole di lettura (contrastive_cn_restart_compare.py), commit 9d34dc69

## Bloccante
R4-B1 compare.py:241,248 - checkpoint_differences confronta anche il `cuda_rng` di primo livello del payload
(run.py:1482, `torch.cuda.get_rng_state_all()` del processo rank 0, 4 dispositivi visibili). Nel continuo
(avvio da zero) i dispositivi 1..3 del rank 0 hanno l'ultimo seme `cfg.seed*1_000_003+1` (run.py:1463, ramo else);
in ogni ripresa hanno `cfg.seed` (run.py:1383) e solo il dispositivo 0 viene ripristinato (run.py:1450).
cfg.seed=0 nel config archiviato: seme 1 contro 0. Foglie uint8 diverse -> metadata_or_schema_different =
['cuda_rng.1','cuda_rng.2','cuda_rng.3'] -> require di riga 248 -> ValueError -> INCONCLUSIVE dopo le GPU,
anche con tensori scientifici identici. Simulazione: agents/r4-confronto/payload_sim.py (stato CUDA simulato
seed+offset). Il confronto storico compare_resume (contrastive_benchmark_readout.py:159-190) e
historical_numeric_pair escludevano torch_rng/cuda_rng di primo livello e confrontavano rank_rng.
Correzione: per la riscaldata escludere (e riportare a parte con le impronte) torch_rng e cuda_rng di primo
livello, confrontando rank_rng; oppure confrontare solo cuda_rng[0]. Aggiungere un test con questo payload.

## Importanti
R4-I1 test: 20 mutazioni su 30 sopravvivono (compare() mai eseguito end-to-end): soglia x10, any_reduced>=0,
any_local senza zeri con segno, gate di fedelta' dell'ordinaria, gate metadata riscaldata, posizionamenti,
uguaglianza fra rank dei ridotti e post-Adam, cold!=hot, first_exact, grad_none, TRACE_EQUAL, stato storico,
join boundary/traccia, non finiti, directory distinte, origin, SHA storici. Script agents/r4-confronto/mutate.py.
R4-I2 soglia 1e-6 e cancellazione: simulazione fp32 a 4 addendi (cancel_sim*.py): rumore indipendente ~5e-8;
servono rapporti sum|g|/|sum| dell'ordine di 10^3 per superare 1e-6 con l'ordine ad anello simulato. Rischio
basso ma un eccesso da sola cancellazione verrebbe letto HYPOTHESIS_UNSUPPORTED. Il report ha le norme locali
per rank e ridotte per tensore; aggiungere un campo descrittivo con il limite elemento per elemento
(|d_j| <= gamma*sum_r|g_rj|/4) senza cambiare lo status.

## Note
- warm: differenza non float a 576 -> INCONCLUSIVE (non registrato; il criterio scritto direbbe parziale).
- any_reduced usa L2>0: differenze di solo segno dello zero nel ridotto non contano (coerente con "nonzero").

## Verifiche positive (eseguite)
- 13 test compare passano. branch_traces(CN_A, CN, warm, r) passa sui 4 rank; metrics_evidence(CN_A, CN, warm):
  245 differenze, prime al 545 (grad_norm 4.336711902194546 contro 4.336711902204087); metrics_evidence(CN_B, CN_A,
  ordinary)=0. history(): SHA del readout e stati storici passano, poi FileNotFound sui .pt (attesi, non sul Mac).
- layouts() con il vero trace-0.ddp.jsonl del continuo e il modello vero (270 parametri, fp32+fp64): passa;
  un'inversione di due ID viene rilevata. Taglie loggate = numel*element_size del modello.
- CN_A al 547 logga lo stesso layout ricostruito del continuo (rischio di layout diverso dopo un backward sul batch 545 escluso).
- classify corrisponde alla tabella; training_approved=false e FAILED_REQUIRES_REVIEW sempre nel report.
- config JSON == config_to_dict (nessuna tupla); source_sha256 dei profili = production_source_sha256 del piano.
