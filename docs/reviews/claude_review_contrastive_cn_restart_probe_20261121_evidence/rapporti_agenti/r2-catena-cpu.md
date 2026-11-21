# R2 - Catena intera su CPU (4 rank Gloo) con le funzioni vere

Clone: 9d34dc69 (sola lettura). Script: agents/r2-catena-cpu/{chain.py,analyze.py,shim.py}. Evidenze: agents/r2-catena-cpu/run3 (res/*.json, osservazioni, post_update, metriche).

## Setup (eseguito)
- trainer vero `R.train` (loader, encoder, InfoNCE CN, AdamW, EMA, checkpoint, ripresa), DDP Gloo 4 rank, config `_config(pose_root,"CN")` dei test, batch globale 8, wsd 9 passi, fork = `checkpoint_step0000005.pt` del continuo (keep_checkpoints), fill_steps=2 (coda vuota e peso 0 al primo update), allarme disattivato, `bucket_cap_mb=0.02` (unica modifica di DDP, per avere 5 bucket ricostruiti); 106 parametri, 98 fp32 + 8 fp64.
- run3: metriche globali del gradiente solo da F+1 (come nel cluster: assenti all'update 1, dense a 545).
- ordinary/warm: `leg.instrument_leg` vero con FORK_STEP=5, `device_identity` sostituita (CPU), conteggio 106 nell'osservatore e in `record_optimizer_step`. Hook `_average_observed` vero.
- tempi: 7-12 s per modo, ~52 s a catena.

## Risultati
1. Hook contro nativo: continuo con `allreduce_hook` dall'inizio = continuo nativo, gradienti ridotti identici a tutti i 9 update, endpoint esatto. Ripresa ordinary (hook) = ripresa plain (senza hook): gradienti identici a tutti e 4 gli update, endpoint esatto, metriche identiche.
2. Riavvio: plain/ordinary differiscono dal continuo dal primo update dopo il fork (6): 84/106 tensori al 6, poi 101-102; endpoint: 300 tensori float diversi. Locali identici, scarto ridotto relativo massimo 9,5e-8.
3. Warm: identica bit per bit al continuo (gradienti ad ogni update, endpoint esatto, metriche identiche). Bucket all'update 6: ordinary 2 (82140 fp32 + 27008 fp64), warm 5 = layout ricostruito del continuo; `layouts()` vero superato contro il log DDP del continuo (ricostruito all'update 1 con coda vuota). Gli indici dell'hook coincidono con gli indici del logger DDP.
4. Shim torch 2.2 (simulazione): `load_observation` solleva ValueError su ordinary e warm (B-1 riprodotto, catturato da compare.main -> INCONCLUSIVE). `contrastive_target_resume.checkpoint` funziona (usa str). Con `str(tensor_path)` load_observation funziona e conserva gli offset. Nessun altro torch.load con mmap su Path nel perimetro. `device_identity` senza `uuid`, con libcuda finta: percorso ctypes corretto.
5. classify sulle funzioni vere: SUPPORTED. Primo tentativo con fork da run separato: alarm_counts diverso (valutazione finale extra del run corto, artefatto della fixture), checkpoint_differences lo marca come metadato e compare lo renderebbe INCONCLUSIVE: nel cluster il fork viene da checkpoint_step0000544 del continuo, quindi non si applica.

Test esistenti: leg/observe/compare 25 passati in 8,5 s.
