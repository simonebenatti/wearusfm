# Recupero P1 a 19.618 — pacchetto approvato del 16/11/2026

> Aggiornamento 17/11: dopo l'emendamento esplicito di Simone, il singolo
> P1-r2 64620434 e' COMPLETED 0:0. Raccolta, verifica e prosecuzione documentate
> in `../p1_recovery_readout_20261117/` e
> `docs/reviews/gpt_scaling_p1_recovery_readout_20261117.md`.
> La frase «Esito finale» sotto descrive la sola diagnostica, prima dell'emendamento.

**Esito finale:** diagnostico 64610181 COMPLETED 0:0, NOT_REPRODUCED /
READERS_CLEAR_IN_TEST. P1-r2 non rilasciato, nessun secondo checkout o retry.
Readout completo: `docs/reviews/gpt_scaling_p1_diagnostic_readout_20261116.md`.
I passi condizionali sotto sono la sequenza approvata, non lavori ancora in esecuzione.

Simone: «approvo tutto, correggi e prepara tutto e lancia tu se non hai altre
cose da correggere». Registrato in `81f1f76`, pubblicato e verificato su origin.
GPT e' l'operatore di questo pacchetto; eccezione al ruolo ordinario di revisore
della linea 1. Un solo operatore, submitter e ledger comuni. Nessun nuovo via
fra diagnostica e P1 se tutti i criteri approvati sono soddisfatti.

La review e il protocollo in `docs/reviews/` restano conservati testualmente.
Questa decisione rende approvata la loro proposta e i relativi tetti; non
allarga l'esperimento scientifico. I due P1 falliti restano FAILED e le loro
cartelle non vengono modificate o adottate.

## Sequenza e limiti

1. Pubblicare codice, test CPU, sei input, budget, review interna e pin.
2. Pubblicare un ref di deploy nuovo; creare un checkout nuovo tramite il job
   versionato `campaign_checkout.sbatch`: 2 CPU, 4 GiB, 5 min (8 con margine).
3. Un diagnostico Booster: 1 GPU allocata inutilizzata, 8 CPU, 32 GiB,
   QoS `boost_qos_dbg`, 27 min richiesti, massimo 30 min col margine verificato
   `OverTimeLimit = 3 min`. Deadline applicativa 1.500 secondi. Output nuovo:
   `/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/diag/scaling_p1_identity_20261116_v1`.
4. Raccogliere report, eventi e accounting testuale. Solo con matrice completa,
   `METADATA_ONLY_REPRODUCED` e `READERS_CLEAR_IN_TEST`, preparare il secondo
   checkout con i pin effettivi del risultato e un solo P1-r2 (1 GPU, 8 CPU,
   100 GiB, 2 h richieste). Lo strumento di recupero verifica questi criteri.
5. Con un errore, TIMEOUT, INCOMPLETE, NOT_REPRODUCED o READERS_AFFECTED:
   nessun altro job discende automaticamente da questo piano. Preservare
   prove e spiegare quale nuova decisione occorre. Nessun retry.

I job pose 64571999 e P2 64572001 sono preservati. I manifest successivi
mantengono il codice reale per ogni cache: pose/P2 `df2de35`, P1-r2 il suo nuovo
commit. I lettori congelati non vengono modificati; validazione e benchmark
restano necessari prima dei fit. Il tetto dei soli due checkout aggiuntivi
non autorizza altri checkout di preparazione fuori dal budget.

## Precisazioni esecutive prima dei risultati

- Le due letture principali sono: sequenza storica stat/hash/stat; pausa di
  2 s; validazione NPZ sintetica; hash su un descriptor con fstat e legame al
  percorso. La validazione e' collocata **fra** i due hash, precisazione
  rispetto all'elenco del protocollo, per coprirne entrambi i confini senza
  aggiungere una terza lettura. La prima osservazione rimane identica a quella
  che ha rifiutato P1. Nessuna apertura aggiunta la precede.
- Il riferimento NPZ piccolo viene scritto anche dopo una pausa di 2 s per
  verificare che l'oracolo BytesIO non dipenda dall'orario del contenitore ZIP.
- SIGALRM fa rispettare la deadline anche nelle funzioni di hash congelate,
  senza cambiarne il codice o l'ordine delle operazioni. Un timeout produce
  INCOMPLETE; I/O kernel non interrompibile puo' ancora terminare in TIMEOUT
  SLURM, che resta inconclusivo.
- La politica del nuovo P1 e' fissata prima della diagnostica: **nessuna
  barriera aggiunta** (`output_barrier: none`), controllo su descriptor,
  tolleranza dei soli mtime/ctime con oggetto e byte invariati e seconda
  lettura stabile. Massimo una rilettura aggiuntiva nell'intera validazione
  di una cache. A/F/S servono alla diagnosi, non a scegliere una barriera
  a posteriori. Nessuna modifica a `tokenizer_preflight.py` o ad altre guardie
  degli input, ne' ai sorgenti scientifici congelati.

## Evidenze e budget

`evidence/preparation/` contiene letture remote di metadata, ricevute,
provenienze, configurazione dello scheduler, quota e SHA dei sorgenti. I payload
grandi non vengono hashati sul login node: li verifica il job diagnostico.
I pin delle cache accettate derivano dalle ricevute e dalle provenienze;
quelli dei file di job falliti restano soltanto diagnostici.

`budget_amendment.json` distingue consumi passati e massimi futuri. Il massimo
aggiornato dello stadio e' 7,264444 GPUh / 58,115556 coreh su 7,30 / 58,40;
P1 cumulativo massimo 3,070833 GPUh su 3,10, tre tentativi totali. CPU-only
350,451111 coreh su 351, con due checkout aggiuntivi al massimo. Il congelamento
storico non viene sovrascritto. `sacct.txt` accompagna i conteggi.

Codice e test del recupero sono preparati insieme al diagnostico. L'assenza
di un manifest P1 con pin reali e' intenzionale fino alla sua conclusione:
non si inventano ricevute o esiti favorevoli e non si accoda P1 prima del gate.
