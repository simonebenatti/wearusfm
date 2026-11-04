# Seguito automatico della campagna tokenizer — 03/11/2026

## Ripresa esplicita del04/11: endpoint pronto per le estrazioni

Simone: «han finito, provvedi / procedi». Training63386315/63386337/63386403 tutti
COMPLETED0:0 a19618; ricevute, config,15hash e checkpoint5k immutabili verificati.
Vedi docs/tokenizer_endpoint_execution_20261104.md e piano/policy/verification_19618.json
in results/infra/tokenizer_evaluation_20261103/. Restano nove estrazioni, non dodici
nuove: le tre5k sono concluse. Nuovo checkout congelato prima dei nove invii.
ID da riconciliare nel ledger: TOKENIZER-eval19618-checkout-20261104 e
TOKENIZER-<ARM>-<pose|p1|p2>19618-20261104. Il seguito avviene nella chat attiva;
il monitor automatico resta spento. Nessun nuovo consenso richiesto.

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

Seguito automatico inizialmente attivo ogni15min: `completa-tokenizer-wearusfm`. **Eliminato il03/11 dopo il lancio della ridge63405791 su richiesta di Simone.**

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

## Launcher CPU pronto durante le estrazioni5k

Preparato scripts/slurm/tokenizer_pose_readout.sbatch senza modificare il reader
scientifico:validate2CPU/30min, benchmark8CPU/15min, ridge8CPU/30min, memoria massima30G.
Usa checkout congelato verificato prima/dopo, piano contenuto nel checkout, uscita nuova
assoluta fuori dal codice; la ridge comprende già il confronto appaiato dei tre bracci.
La CLI compare resta disponibile ma non rappresenta un job aggiuntivo prenotato.

Sintassi bash e contratto CLI verificati. Nessun job CPU o checkout aggiuntivo lanciato:
prima servono tutte e sei le cache valide con ricevute complete, poi piano con pin reali,
preventivo del checkout e benchmark train-only. La sola presenza di train_subset.npz
non autorizza a consumare risultati parziali della catena.

## Aggiornamento: estrazioni5k complete, fase CPU e arresto monitor richiesto

63399428/63399479/63399484 COMPLETED0:0; sei pin reali nel piano
results/infra/tokenizer_evaluation_20261103/pose_readout_plan_5000.json (9efbf827).
Leggere docs/tokenizer_pose_cpu_5000_execution_20261103.md e relativa resource_policy.
Prossimi ID: TOKENIZER-posecpu5k-checkout-20261103 e TOKENIZER-pose5000-validate/
benchmark/ridge-20261103. Prima di qualunque invio leggere il ledger: nessun duplicato.
Completare validate+benchmark, leggere il gate, inviare ridge se valido; poi **eliminare
completa-tokenizer-wearusfm**, come richiesto ora da Simone. Non proseguire in background
verso i risultati o i valutatori19618 dopo l'eliminazione; i job continuano su Leonardo.

## Fase CPU5k: gate superato, ridge da sottomettere

Checkout63405286, validate63405498 e benchmark63405502 tutti COMPLETED0:0.
SHA runtime70fd080149c11d10917059875d31a5a2820f176a; gate raccolto in
results/infra/tokenizer_evaluation_20261103/cpu_5000/pre_ridge_gate.json.
Previsione420s entro1800s; ridge8CPU/30G/30min autorizzata. Leggere il ledger
TOKENIZER-pose5000-ridge-20261103 prima di inviare: il lancio segue questo commit.
Poi disattivare il monitor come richiesto, lasciando i job attivi su Leonardo.

## Ridge sottomessa; monitor disattivato

**63405791** è la ridge dei tre bracci teacher5k, sottomessa tramite il ledger
TOKENIZER-pose5000-ridge-20261103 dopo pubblicazione e verifica origin del gate
49f91a244632e7546ed88618ee3300da61f443c6. Runtime congelato70fd080;8CPU/30G/30min,
0GPUh e4coreh massime. Il report includerà il confronto appaiato.

L'app conferma l'eliminazione di **completa-tokenizer-wearusfm** dopo il lancio,
come richiesto da Simone. Nessun job SLURM cancellato. Non sono previsti controlli,
raccolte o lanci automatici successivi. La lettura del report ridge e le valutazioni
all'endpoint19618 restano da eseguire in una successiva attività esplicita della chat.
Non risottomettere la ridge o le continuazioni63386315/63386337/63386403.

Primo controllo dopo invio: ridge63405791 **RUNNING**,8CPU/30G; le tre
continuazioni19618 restano **PENDING**. Nessun altro controllo di completamento
programmato dopo questo snapshot di consegna.
