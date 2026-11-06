# Review degli adattatori T100-LQ dopo 1d6f96a

Lavora in sola review sulla nuova consegna del ramo
`origin/codex/tokenizer-logcontrast-20261104`, diff da `6d5b353`.
È il seguito della tua review `1d6f96a`: R1, R2, controller e disegno sono
già approvati. GPT ha implementato gli adattatori e recepito le note.
Nessun job o deploy nuovo; training ancora `REVIEW_REQUIRED`, release nulla.
Simone ha già autorizzato l'esecuzione dopo convergenza: non serve un altro via.

Leggi prima:

- `docs/logsquare_adapters_execution_20261104.md`;
- `docs/logsquare_preflight_readout_20261104.md` e protocollo aggiornato;
- `docs/reviews/logsquare_adapters_critic_20261104.md`;
- piani e risorse in `results/infra/logsquare_evaluation_20261104/`;
- `component_review.json` e piano in `results/infra/logsquare_training_20261104/`.

Rivedi codice, quattro launcher e relativi test:
`logsquare_evaluation.py`, `logsquare_infer_benchmark.py`,
`logsquare_pose_readout.py`, `logsquare_probe.py`.
I moduli numerici storici e il controller training approvato non cambiano.

Verifiche richieste:

1. **R1-a/R1-b e N1–N4:** rimosso l'argomento tautologico sul limite analitico;
   norme delle 64 colonne W_log da checkpoint, per student e teacher, famiglie
   e confronto MLP/non-MLP nei readout dei due endpoint, solo descrittive.
   Note su time_limit senza retry, rilascio dedicato pinnato, conservazione
   SCRATCH/WORK e override esplicito 4 h della continuazione. Epsilon .001
   resta fissato; cambiarlo dopo i risultati richiederebbe un nuovo candidato.
2. **D1/D2:** benchmark train sul percorso reale, posa FP32 e classificazione
   BF16 con decoder; scale, crop e supporti invariati. Massimo dei rapporti
   per i tetti, media pesata dai conteggi storici di finestre per la stima.
   Verifica selezione rappresentativa, pin e ricostruzione del preventivo,
   compreso il legame fra ricevuta benchmark e piano aggiornato senza cicli.
   Benchmark al 5.000; la continuazione non aspetta il budget delle estrazioni.
3. **D3 e integrità:** test con loader e percorsi reali, W_log zero uguale
   al lineare/nonzero diverso, gain e pesi restaurati senza calibrazione,
   solo due default legacy, rifiuto dei pin mancanti o incoerenti.
4. **Lettori:** fit solo LQ; tre controlli originali, senza riscrittura delle
   provenienze. Ridge e logistica storiche, supporti/fold/target identici,
   bootstrap condiviso a quattro celle, criteri e griglie invariati,
   convergenza irrisolta trattata esplicitamente. Rivedi anche il contratto
   tra ricevute dell'estrattore e lettori.
5. **Risorse:** training approvato 22 GPUh; nuovo benchmark fino a 0,5 GPUh;
   estrazioni ancora senza tetto finché rho_infer non è misurato. Lettori CPU:
   posa 16 coreh con nuovo benchmark, sonde 34,3333 coreh. Per le sonde valuta
   esplicitamente il riuso delle misure storiche per shard con identità di
   ambiente/ricetta/sorgenti/conteggi: è una proposta, non un benchmark LQ
   già fatto. Feature diverse possono cambiare il tempo del solver.
   Checkout 0,3333 coreh ciascuno, da contare nel piano esecutivo effettivo.

GPT ha verificato 417 test (276 regressioni + 141 degli adattatori), sintassi
Bash, pin locali e inventario dei 102 sorgenti training immutati. Comandi e
limiti in `local_validation.json`; nessuna verifica CUDA nuova dichiarata.

Consegna un verdetto sugli adattatori e sui loro gate di risorse, distinguendo
blocchi reali e note. Se approvi, indica se la consegna soddisfa i prerequisiti
perché GPT prepari il commit dedicato `APPROVED_FOR_TRAINING`, con release
pinnata alla tua review: non confondere questa consegna preparatoria con un
training già rilasciato o lanciato. Pubblica review e registro sul tuo ramo.
Nessun lancio da parte tua, nessuna contrastiva o estensione scientifica.
