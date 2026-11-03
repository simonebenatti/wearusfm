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
