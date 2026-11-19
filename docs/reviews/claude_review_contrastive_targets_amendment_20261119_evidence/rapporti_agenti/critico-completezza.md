# Critico di completezza: adeguamento target della linea 3 (438c8d65, consegna e388a1de..0efdc29c)

Metodo: confronto fra la sintesi dei sette revisori e i sei punti richiesti da GPT, poi controlli rapidi nella copia git
(sola lettura; un solo script di sonda in `tmp/critico/old_review_pin.py`; copia git pulita, nessun file ignorato creato).

## Lacune confermate (in ordine di peso)

1. **Punto 2 e richiesta finale (non copiare autorizzazioni): il pin `review` dei job non è legato alla nuova coorte.**
   `contrastive_job.py:64-69` controlla solo lo stato `APPROVED_FOR_PREFLIGHT` scritto nel job, il digest delle sorgenti
   e lo sha del file di review. L'audit, prima cella, non ha release (righe 103-104).
   Sonda: job audit nuovo con `review` = `docs/reviews/contrastive_preflight_release_20261117.md` (rilascio del 17/11,
   «APPROVED_FOR_PREFLIGHT ... per questo giro») e config di calibrazione nuova. Esito: i pin di review e config sono accettati,
   lo stop avviene solo sugli input (`FileNotFoundError` sul Mac; su Leonardo gli input sono identici allo storico).
   P2 e P6 dicono "procedurale", ma nessuno ha provato che un'autorizzazione vecchia passa. Le ricevute vecchie sono invece rifiutate meccanicamente.
   Verifica: `sed -n 62,69p scripts/contrastive_job.py`; esecuzione di `tmp/critico/old_review_pin.py <clone> <work>`.

2. **Punti 6 e 5: la matrice nuova è identica a quella fallita.** Ci sono le stesse 9 celle (modo, braccio) e le stesse dipendenze.
   Cambiano solo i tre prerequisiti procedurali dell'audit (script di confronto dei due `preflight_plan.json`), e
   `contrastive_split_benchmark.py` è invariato da 8905e346. Le due riprese restano due training indipendenti confrontati con i completi.
   GPT stesso ha dichiarato confuso questo disegno (registro `docs/claude_GPT_shared` r.9422-9431: norme diverse già al passo 1,
   proposta "ripresa da un unico stato salvato"). Nessun revisore ha verificato quale esito della diagnostica permetta di
   riusarlo; l'emendamento dice solo "risoluzione o emendamento esplicito". Se la deriva resta, le celle split
   (4 GPUh, 32 core-ore) ripetono la matrice negativa e il preventivo di 13 allocazioni non è definitivo.

3. **Punto 3: l'appaiamento è verificato solo nei config.** La parità numerica a peso zero (256 update) è il cancello fallito
   (registro r.9440-9447: CN 146 e CR 135 norme oltre rtol 1e-4/atol 1e-6). I nuovi config hanno ancora `amp_bf16: true`, come T100-G.
   Il punto 3 dipende quindi dalla chiusura del punto 6; P3 lo cita solo come fatto storico.

4. **Punto 5: checkout, oltre alla memoria (P5-M1).** `campaign_checkout.sbatch:5-7` ha di default 1 CPU, 2G e 10 min.
   Tutti i checkout storici sono stati inviati con `--cpus-per-task=4 --mem=2G --time=00:04:00` (uno con 00:05:00; vedi `*checkout*state.json`).
   Il preflight dichiara 4 CPU/30 GiB/240 s ed elenca solo l'override dell'audit.
   Con 4 CPU e i 600 s di default: 4x4x600/3600 = 2,666667 core-ore e totale 138,666667 > 137,066667.

5. **Punto 5: mancano le ore locali e il residuo del budget (CLAUDE.md:263).** P5-N4 copre solo il residuo.
   Nella contabilità storica il billing coincide con le CPU: GPU billing=32 (16 ore locali per cella), audit 8 CPU/30G billing=8,
   checkout 4 CPU/2G billing=4. Le 137,066667 core-ore valgono come ore locali solo con i checkout a 2G; per 4 CPU/30G non c'è contabilità.

6. **Punti 1-2: `local_validation.json` non è riproducibile.** Nessuno script del repo genera i 18 controlli né il blocco
   `independent_preparation_review` (`git log --all -S storedconfigmatchesgenerator` trova solo 0efdc29c). P7-F3 copre solo
   i 121 test mirati. `cli_explicit` è dichiarativo; è coerente con `train_jepa.py:201-212` (125 -> fast125_v3), verificato leggendo il codice.

7. **Punti 1-2: documenti operativi vecchi senza rimando.** Solo il protocollo ha la nuova intestazione.
   `contrastive_preflight_execution_20261116.md`, `contrastive_handoff_20261117.md` e `prompt_3_launcher_contrastive_readout_20261118.md`
   hanno 0 occorrenze di amendment/100/125/fast125. Rischio basso: i job vecchi sono rifiutati per sorgenti/ALLOCATED_SECONDS.

8. **Punto 6: stato reale non visto dai revisori.** `rev_targets/leonardo_state.out.txt` (lettura delle 10:07, non riverificata)
   non mostra ref `deploy/contrastive-common-*`, checkout `contrastive-common-20261118`, né uscite della diagnostica o della nuova coorte.
   La diagnostica quindi non è partita e la condizione del punto 6 è tutta da soddisfare. L'operatore deve rileggere stato e ledger prima del rilascio.

9. **Punto 2 (test): il titolo di P2-M1 è troppo stretto.** Le mutazioni di P2 mostrano che sono senza test tre barriere di secondo livello:
   `contrastive_run.py:99-101` (config_sha256 dell'audit), `contrastive_benchmark_readout.py:331-334` (lo stesso controllo in read_preflight)
   e la guardia della ricevuta in authenticate_calibration. Con MD+MG+MF insieme passano 337 test su 337 (`reports/p2-mutations/all_contrastive_MD_MG_MF.txt`).

## Verifiche aggiunte che chiudono voci "non verificate"

- `docs/claude_GPT_shared` in 438c8d65: confronto per multiinsieme delle righe (afb89b45 contro 438c8d65). È stato tolto solo il titolo
  duplicato "## 2) Problemi trovati" e sono state aggiunte 23 righe; il resto è uno spostamento, le voci restano testuali.
- Moduli contrastivi invariati da 8905e346: compat, audit, run, cuda_check, split, rank_exit, compare, objectives, memory
  (`git diff --stat` vuoto). Quindi non ci sono soglie implicite nuove e la whitelist non è stata ampliata.
- `authenticate_preflight` ricalcola `read_preflight` (readout:401-412) e le riprese passano da `_run_evidence` con
  config uguale (r.201-209, 377). Anche una vecchia ricevuta PASSED urterebbe la barriera di ricetta.
- Nessun codice legge `plan.resources` storici: le allocazioni del training vengono solo dal readout misurato (readout:361-372, run:137-145).
- `scales` è la normalizzazione MAD per sessione (pretraining_loader.py:246-249), indipendente dai target: il riuso dei pin d'ingresso è legittimo.
- Requeue: nei launcher non c'è `--no-requeue`, ma tutte le istantanee storiche di Leonardo, anche su lrd_all_serial,
  hanno `Requeue=0`. Un retry automatico di SLURM non è atteso.

## Esito
Nessun blocco nuovo. Le lacune 1-3 dicono che cosa il rilascio deve dichiarare e controllare. Le 4-5 sono correzioni del preventivo,
le 6-9 sono note di riproducibilità e documentazione.
