# Seguito automatico della campagna tokenizer — 03/11/2026

Simone ha autorizzato tutti e tre i bracci nuovi fino a19.618, valutazioni incluse, lancio
diretto tramite submitter/ledger e superamento del precedente budget. **Nessun nuovo consenso**
per letture, commit/push, prosecuzioni o valutazioni di questo confronto. Non introdurre loss,
altri semi, altre dosi o altre domande scientifiche. Non modificare i checkout in uso.

Worktree Mac: `/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science`,
ramo `codex/r3-long-collaudo-20261028`. Codice training congelato
`afb644ea10ecc2a716160b4d8a4904e805c366fb`, pubblicato e verificato su origin e deploy Leonardo.
Checkout remoto `/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/tokenizer-training-20261103`.
Checkout63385655 completato0:0 in32s, pulito e in sola lettura.

| Braccio | Training5k | Continuazione19618 | Dipendenza |
|---|---:|---:|---|
| T25-L |63385800|63386315|afterok63385800|
| T100-L |63386190|63386337|afterok63386190|
| T100-G |63386228|63386403|afterok63386228|

Al controllo dopo invio: tre iniziali PENDING/Priority, tre continuazioni PENDING/Dependency.
Non c’erano ancora log perché i job non erano partiti. **Non risottomettere questi sei job.**
Ricevute locali in `results/infra/tokenizer_execution_20261103/submission/`, indice
`training_submissions.json`. Il ledger comune è `~/wearusfm_local/jobs/`.

Prima di decidere qualunque seguito, leggere:

- `docs/tokenizer_training_execution_20261103.md`: dose, tetti, ripresa, interpretazione.
- `docs/tokenizer_evaluation_execution_20261103.md`: matrice esatta, API e criteri.
- `results/infra/tokenizer_execution_20261103/full_resource_policy.json`: risorse per fase.
- `docs/reviews/tokenizer_training_launch_20261103.md`: review e107test passati.

## Quando ci sono checkpoint completi

Le continuazioni sono già in coda; non richiedono un nuovo invio. Per ogni fase completata,
leggere `sacct` parent e ricevuta in
`$WORK/wearusfm_runs/results/tokenizer_training_20261103/<ARM>/<STEP>/receipt.json`.
I run sono in `$SCRATCH/wearusfm_runs/runs/tokenizer_20261103/<ARM>/`.
Solo COMPLETED0:0, ricevuta completa, SHA/dose/configurazione coerenti e assenza STOP consentono
le estrazioni. Il checkpoint nominato5000 e lo snapshot della sua configurazione sono immutabili;
non pinnare `checkpoint.pt` o il `config.json` mobile per estrarre la dose intermedia.

I valutatori sono già implementati/revisionati nel codiceafb644e:

- `scripts/tokenizer_evaluation.py`: `--stage validate|extract-pose|extract-classification`.
  `make_plan(comparison, inputs, paths, pose_references, classification_references, checkpoints=None)`.
  Pin checkpoint per dose/braccio: path,sha256,config_path,config_sha256,receipt_path,receipt_sha256.
  Preparare i piani dagli artefatti effettivi; le guardie ricontrollano teacher, GELU/gain,
  configurazione completa, firma dati, target/maschere e identità native della classificazione.
- `scripts/tokenizer_pose_readout.py`: `validate|benchmark|ridge|compare` posizionale.
  `make_plan(caches,references,fold_users,step)` richiede tutti i6pin train/val della dose.
  Dopo le tre estrazioni della dose, benchmark prima della ridge entro8CPU/30G/30min per report.
- `scripts/tokenizer_probe.py`: `fit-shard|aggregate|compare` posizionale.
  `make_plan(caches,datasets,step=19618)` richiede tutti i15pin endpoint. Cinque shard per braccio
  come politica, sei aggregati P1/P2, confronto comune. Conservare anche fit non convergenti;
  in quel caso descrittivi sì, decisioni di non inferiorità irrisolte.

Restano da concretizzare i piani con gli hash ancora inesistenti e i launcher delle valutazioni;
non fingere che siano già sottomessi. Usare comandi scientifici e API già verificati, launcher
versionati e checkout congelato; nessun editing SSH sottoWORK. Se serve un nuovo checkout per
aggiungere piani/launcher, dichiarare prima anche il relativo costo CPU tramite normale procedura;
l’autorizzazione corrente copre questo adattamento operativo. Pubblicare e verificare originSHA
prima di ogni nuova fase. Valutazioni GPU: sei posa45min, treP1 30min, treP2 45min, ciascuna1GPU.

## Controllo e consegna

Usare `ssh leonardo` per letture e lo stesso submitter assoluto già autorizzato per gli invii.
Non duplicare job, non ripetere ciecamente dopo errori o risposte ambigue. Un problema tecnico
si diagnostica dai log prima di cambiare codice o dichiarare risorse aggiuntive. Non usare un
checkpoint incompleto e non cambiare protocollo per far superare una soglia.

Raccogliere report/receipt/hash/log e consuntivi senza segnali grezzi in Git. Review del critico
sui risultati, aggiornamento della tabella e del registro in entrambe le sezioni, decisioni,
commit/push con verificaSHA. Screening5k descrittivo; confronto endpoint19.618 a pari dose,
posa ed EPN separati, P1/P2 secondari e IC condizionati a un seme/val già osservata.

Silenzio quando lo stato non cambia o è soltanto attesa in coda. Comunicare soltanto un lancio
significativo, un problema che richiede intervento o risultati conclusivi. Disattivare il seguito
automatico alla conclusione o quando serve una nuova scelta scientifica di Simone.

Seguito automatico attivo: `completa-tokenizer-wearusfm`, controllo ogni15min nella stessa chat, silenzioso sugli stati invariati. Disattivarlo alla chiusura.

## Aggiornamento delle11:49UTC:5k completo

Tutti e tre i5k COMPLETED0:0,3834/3926/3916s. Evidenze e pin verificati in
results/infra/tokenizer_execution_20261103/5000/ e
results/infra/tokenizer_evaluation_20261103/extraction_plan_5000.json.
Leggere docs/tokenizer_evaluation_5000_execution_20261103.md per i prossimi tre
extract-pose e il nuovo checkout. Continuazioni19618 PENDING/Priority, non duplicare.
Prima di inviare controllare gli ID TOKENIZER-eval5k-checkout-20261103 e
TOKENIZER-<ARM>-pose5000-20261103 nel ledger: la sottomissione può essere avvenuta
dopo questo documento. Ricevute e ID vanno raccolti dopo gli invii.

## Sottomissioni eseguite

Checkout **63399280 COMPLETED0:0**,18s,0,010000coreh CPU-only, SHA
**f2061b8bc91a6355213793a9719121135ad48fa2** verificato, pulito e in sola lettura.
Pubblicato prima dei lanci su origin e nuovo deploy/tokenizer-eval5k-20261103.

| Braccio | Estrazione posa5k | Primo stato verificato |
|---|---:|---|
| T25-L |63399428|RUNNING|
| T100-L |63399479|RUNNING|
| T100-G |63399484|RUNNING|

Tre invii riusciti tramite submitter/ledger comuni, allocazioni1GPU/8CPU/100G.
Ricevute originali in results/infra/tokenizer_evaluation_20261103/submission/,
indice submissions_5000.json. Continuazioni63386315/63386337/63386403 ancoraPENDING.
Nessun risultato scientifico delle estrazioni o ridge ancora disponibile.

**Non duplicare queste tre estrazioni.** Il checkout di valutazione resta f2061b8;
le modifiche di soli documenti successive non vanno deployate sopra il checkout attivo.
Dopo tutte e tre le ricevute complete, raccogliere e verificare cache/provenienze,
pinnare sei cache e proseguire con benchmark train-only e ridge entro i tetti approvati.
Per stato e log preferire comandi semplici: ssh leonardo sacct/squeue/tail/cat con
argomenti letterali. Il 03/11 gli script multilinea annidati hanno prodotto richieste
di permesso tecnico; gli invii e le letture semplici successivi sono passati usando
le regole persistenti esistenti. Nessuna nuova firma scientifica o operativa necessaria.
