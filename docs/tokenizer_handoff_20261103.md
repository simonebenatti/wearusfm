# Seguito automatico della campagna tokenizer — 03/11/2026

## Stato attuale04/11, aggiornamento12:19CET: tutti15fit sottomessi

Tutti i15shard delle sonde sono ora inviati. Ultimi due T100-G: DB2 fold2-3
**63504830**, P2 **63504862**. Otto shard COMPLETED0:0 e raccolti: tutti5diT25-L,
DB3/DB6 diT100-L e DB3 diT100-G. Verificati hash report/audit/OOF, identita
runtime/piano/cache e224fit/224convergenti.36warning di scoring per classi assenti
nei target di fold DB3, previsti dalla convenzione del protocollo; zero warning
di convergenza. Evidenze in probe_cpu_19618/completed_shards/.

Aggregati T25-L P1 **63505189** e P2 **63505230** accodati dopo la verifica degli
input completi. I vecchi ID non sono piu tutti disponibili a scontrol: accounting
e hash verificati sostituiscono afterok su job gia finiti, nessun rilancio.
Snapshot:1fitRUNNING+6fitPENDING+2aggregatiPENDING,9attivi entrocap10.
Restano da inviare4aggregati dei dueT100 e confronto finale su input completi.
Consuntivo8fitcompleti18,242222coreh; zeroGPU. ETA15:30-17:00CET resta indicativa
e condizionata a coda/invii. Nessun risultato finale classificazione ancora.
Monitorautomatico spento. Multinodo accantonato su richiesta di Simone:
per eventuali training successivi resta la configurazione4GPUperrun.

## Stato precedente04/11, aggiornamento10:26CET:13/15fit sottomessi

Liberati due posti, invii autorizzati eseguiti: T100-G DB6 **63497058** e
T100-G DB2 fold0-1 **63497284**. Rimangono soltanto T100-G DB2 fold2-3 e P2
(EPN+UCI), poi6aggregati e confronto finale, tutti già autorizzati. Ledger e
indice cpu_19618/submissions.json aggiornati. Nessun duplicato.
Tre fit COMPLETED0:0: T25-L DB3 63495005 (596s),T25-L DB6 63495008 (749s),
T100-L DB3 63495819 (523s). Per gli ultimi due il dato è ancora solo accounting:
leggere/verificare i report prima degli aggregati. Un fit in esecuzione e9in coda
dopo gli ultimi invii; cap10attivi raggiunto. Nessun hold, monitorautomatico spento.
Stima complessiva indicativa15:30–17:00CET da tempi storici, condizionata alla
continuità della coda e al completamento degli ultimi invii/aggregazioni; non è
una promessa di raccolta automatica mentre il monitor resta disattivato.

## Stato precedente04/11: posa endpoint conclusa; sonde in corso

Ridge63495299 COMPLETED0:0,142s; risultati e critica in
reviews/tokenizer_pose_19618_{results,critic}_20261104.md. E_dyn
0,851975/0,843363/0,838933; benefici1,01%/0,53%/1,53%, sotto2% pratico.
Non inferire EPN200Hz da questa regressione2kHz. Cache/fit scientifici invariati.

Runtime CPU **b315dadfedc09c522deeb56af829d34ebc04f4b4**. Primo shard
T25-L/DB3 **63495005 COMPLETED0:0**,596s,28/28fit convergenti, gate3526a95.
T25-L ancora: DB2-01=63495003,DB2-23=63495004,DB6=63495008,P2=63495009.
T100-L: DB2-01=63495813,DB2-23=63495814,DB3=63495819,DB6=63495821,P2=63495823.
T100-G: soloDB3=63495895 sottomesso. In totale11/15shard sottomessi.
**Tutti gli hold temporanei sono stati rilasciati**, verifica registrata.

Restano da sottomettere, quando la coda ha spazio, T100-G DB2-01/DB2-23/DB6/P2,
poi6aggregati e confronto secondo probe_resource_policy_19618.json. Non duplicare
job esistenti. Indice/ricevute e pending_actions.json in cpu_19618/. Non superare
10job RUNNING+PENDING complessivi sulla seriale; QOSMaxCpuPerUserLimit permette
un solo fit8CPU alla volta. Autorizzazione già acquisita, nessun nuovo consenso.
Nessun monitor automatico attivo o da ricreare; le raccolte avvengono nella chat.


## Stato ultimo04/11: primo gruppo CPU sottomesso

Runtime **b315dadfedc09c522deeb56af829d34ebc04f4b4**, checkout comune verificato
63494914 COMPLETED0:0. Validate63494990 + benchmark63494999; cinque shard T25-L
DB2-01=63495003,DB2-23=63495004,DB3=63495005,DB6=63495008,P2=63495009.
Non duplicare. Leggere gate posa prima ridge e primo completamento tecnico sonde
prima degli altri due bracci. Queuecap10RUNNING+PENDING, spazio anche alla posa.
Indice/ricevute in results/infra/tokenizer_evaluation_20261103/cpu_19618/.
Gli aggiornamenti documentali successivi non modificano questo runtime.

## Stato precedente04/11: estrazioni concluse, fase CPU da inviare

Tutti9job63491925/35/37/40/41/43/45/47/48 COMPLETED0:0, ultimo08:50:01CET.
Costo3,581111GPUh/6; la matrice12estrazioni è conclusa, non duplicare.
Piani CPU19618 con pin reali in results/infra/tokenizer_evaluation_20261103/:
pose_readout_plan_19618.json e probe_plan_19618.json, con policy/verifiche.
Checkout unico nuovo tokenizer-endpointcpu-20261104,2CPU/4G/10min; SHA da
pubblicare/verificare prima del deploy. Posa7coreh, sonde101coreh, zeroGPU;
aggiunte esplicite2coreh persecondobenchmark e0,333333checkout.
Leggere docs/tokenizer_{pose,probe}_cpu_19618_execution_20261104.md e review
reviews/tokenizer_endpoint_cpu_prelaunch_20261104.md prima degli invii.
Ledger prima di ogni invio; gate pose da leggere prima ridge, shard in gruppi
entro10 job seriali RUNNING+PENDING. Seguito nella chat attiva, monitor sempre spento.
Nessun nuovo consenso, training, loss, seme, dose o candidato log-quadratico.

## Stato storico04/11: nove estrazioni endpoint in esecuzione

RuntimeGPU453ce6c496745f2d73aa3007406721586e6dd767, checkout63491904 completo18s.
Il precedente63491851 è fallito3s prima di creare cartelle per invio anticipato
rispetto alla fine del push; recupero diagnosticato e pubblicato904187c.
Estrattori tutti RUNNING al primo controllo: T25-L posa63491925/P163491935/P263491937;
T100-L posa63491940/P163491941/P263491943; T100-G posa63491945/P163491947/P263491948.
**Non duplicare nessuno di questi job.** SeiGPUh già nella matrice autorizzata.
Indice/ricevute in results/infra/tokenizer_evaluation_20261103/submissions_19618.json
e submission/. Piano/policy e seguito in docs/tokenizer_endpoint_execution_20261104.md.
Il launcher CPUsonde è pronto e revisionato,18test passati; i fit non sono ancora
sottomessi, perché servono tutte le cache/provenienze reali e il relativo piano.
Monitor spento. Lo screening5k è letto/revisionato e pubblicato453ce6c.

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
