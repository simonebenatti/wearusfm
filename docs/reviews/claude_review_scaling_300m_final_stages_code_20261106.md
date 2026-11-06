# Review Claude — codice degli stadi finali dello scaling 30M/300M a 5.000 passi: aggregazione e confronto (06/11/2026)

Review 10 della linea scaling. Richiesta di Simone del 06/11: «si mandato. in ealta gpt mi dice che il codice è gia
fatto e mi hciede la tua reviione :», seguita dal prompt di GPT («Fai una review mirata del codice che conclude la
valutazione scaling 30M/300M a 5.000 passi: aggregazione dei fit e confronto finale. Solo review, senza modificare il
codice né lanciare job.»). Ramo `codex/scaling-300m-20261104`, ultimo commit coperto `97125bd` (riletto sul remoto
alle 09:15 del 06/11); codice numerico congelato `aef665b`.

## VERDETTO

**Codice di aggregazione e di confronto: APPROVATO. Nessun blocco, nessuna correzione richiesta prima dell'uso.**

Il verdetto riguarda il codice esistente e le interfacce del dispatcher. Non approva manifest o comandi futuri: quelli
degli aggregati e del confronto non sono pubblicati, e il §5 elenca che cosa resta da verificare su di essi. Non
autorizza il seguito a 19.618. Quattro suggerimenti facoltativi (S-1…S-4), da non applicare ora al codice congelato, e
quattro note per il rilascio (R-1…R-4).

| Controllo | Esito |
|---|---|
| Identità del codice con `aef665b` | CONFERMATA: in `scripts/`, `src/` e `tests/` cambiano solo il dispatcher di recupero e tre test dei launcher |
| 1. Aggregazione | CONFORME: copertura esatta, niente doppioni né buchi, righe, utenti ed etichette ricostruiti dalla cache, nessun fit |
| 2. Provenienza | CONFORME: catena di hash fino agli shard, che il comparatore rilegge uno per uno |
| 3. Confronto | CONFORME: punti, pesi, segni e intervalli ricalcolati per un'altra via, uguali entro 1e-9 |
| 4. Interpretazione | CONFORME: convergenza non risolta → numeri visibili e nessuna lettura; nessun override |
| 5. Rilascio operativo | Interfacce `probe-aggregate` e `compare` conformi; manifest e comandi non pubblicati: NON VERIFICATI |
| Test di GPT | 165 passati, col comando del prompt, su `97125bd` |
| Prove mie | 9 passate, sulla disposizione reale dei 14 fit |
| Artefatti reali, score non aperti | 13 fit su 14 conclusi e coerenti; righe, etichette e utenti identici fra le tre celle |

## 0. Perimetro, metodo e copertura

- Solo lettura: nessun job, nessuna cancellazione, nessuna modifica a codice, dati, checkout di GPT o file remoti.
  Letture di Leonardo alle 08:50, 09:00 e 09:11 in `~/wearusfm_local/logs/claude_review10_leonardo_{1,2,3}.txt`.
- **Copertura precedente (`a988a72`, 04/11).** Avevo rivisto lettori, proiezioni, sonde e comparatore al commit
  `0d5f0f9`, leggendo il codice e sui test sintetici, con cinque note facoltative; scrivevo: «Non ho eseguito lettori,
  sonde o S3 su dati reali».
- **Copertura di oggi.** (a) Identità del codice con `aef665b` e l'unica differenza numerica rispetto a `0d5f0f9`,
  cioè la correzione della mia nota F-1 in `scaling_compare.py`. (b) Le interfacce `probe-aggregate` e `compare` del
  dispatcher `scripts/slurm/scaling_readers_recovery.sbatch` (sha256 `c2bd7979…`). (c) I 165 test del prompt,
  rieseguiti. (d) Nove prove mie sulla disposizione reale dei 14 fit, che i test di GPT non esercitano (lì ogni
  aggregato nasce da un solo shard). (e) Gli artefatti reali già prodotti: report della ridge e 13 fit, letti senza
  aprire gli score.
- **Non verificato perché non esiste ancora:** l'uscita del fit 14 (63635524, in corso: 11 fit su 21 alle 09:11), i
  sei aggregati, il manifest del confronto, l'uscita del confronto, i comandi d'invio dei due stadi. Non è un difetto
  del codice.
- Non ho aperto gli score: nessun valore di `val_bacc`, `held_bacc`, `mse`, `analysis` o `paired` è stato stampato o
  guardato, e delle predizioni reali ho letto solo l'intestazione. L'unico controllo che usa valori di questo tipo
  (`validate_pose_selection` sulla griglia registrata dalla ridge) restituisce solo passa o non passa. La lettura
  scientifica resta di GPT, al confronto.
- Riproducibilità, tre file accanto a questo: `…_checks.py` (le nove prove), `…_oof_identity.py` (lettore remoto di
  sola libreria standard), `…_real_metadata.py` (controlli sui report già raccolti da GPT).

## 1. Identità del codice

- `git diff --stat aef665b 97125bd -- scripts src tests` elenca quattro file: `scripts/slurm/scaling_readers_recovery.sbatch`
  (nuovo, rivisto nella review 8) e tre test di launcher (`test_scaling_readers_recovery_launcher.py`,
  `test_scaling_finish_submissions.py`, `test_pose_b19618_launcher.py`). Tutto il resto è identico.
- Stesso blob in `aef665b` e `97125bd` per `scaling_probe.py`, `scaling_compare.py`, `scaling_pose_readout.py`,
  `scaling_projection.py`, `scaling_reading.py`, `scaling_train.py`, `scaling_evaluation.py`, `tokenizer_probe.py`,
  `tokenizer_pose_readout.py`, `tokenizer_preflight.py`, `downstream_ab.py` e per i cinque test `test_scaling_{probe,
  compare,pose_readout,projection,reading}.py`. L'affermazione di GPT è confermata.
- Il checkout eseguito su Leonardo (`…/wearusfm_campaigns/scaling_300m_evaluation_20261105_v1`) è a `aef665b`, pulito
  (lettura delle 09:11).
- Fra `0d5f0f9` e `aef665b` il codice numerico cambia in un solo punto: `scaling_compare.py`, +9/−1 (`34b878a`). Il
  confronto fra l'ispezione delle cache della posa registrata e quella corrente ora esclude lo stato del file (inode,
  date) e tiene tutto il resto; i pin di contenuto e `check_unchanged` restano. È la correzione che avevo chiesto
  (F-1 del 04/11), con tre test: stesso contenuto dopo un cambio di metadati accettato, byte o supporto diversi
  rifiutati.

## 2. I cinque controlli

### 2.1 Aggregazione (`scripts/scaling_probe.py`, righe 342–493)

- **Copertura.** Ogni shard deve avere cella, compito, codice e piano dell'aggregato (442–443) e la stessa firma
  numerica (444). Per dataset: nessun dataset estraneo (450), nessun fold ripetuto (457), predizioni presenti per ogni
  fold riportato (459), nessun fold non riportato dentro l'OOF (461), tutti e quattro i fold (462). Audit: 28 fit per
  dataset (474), sette per fold con le sei C una volta e un solo refit (376–378).
- **Ricostruzione dell'OOF.** Per ogni fold le righe attese si ricalcolano dalla cache originale e dai ruoli del
  piano; righe, etichette e utenti dell'OOF devono coincidere con la cache (367–372), e così i conteggi (374–375).
  Le etichette si confrontano come array: quelle testuali di EPN612 passano dallo stesso controllo. I fold si
  concatenano in ordine 0…3 e le righe devono restare uniche (466–467).
- **Nessun fit, nessuna scelta.** `aggregate` non chiama mai `fit`. La C scelta si ricalcola dall'audit e deve essere
  quella registrata (394–400); lo score del fold si ricalcola dalle predizioni e deve essere quello registrato (401).
- **Uscite.** Tre file nuovi, il report per ultimo (476–492): un'interruzione lascia al più i due file accessori, e
  il dispatcher rifiuta di riusare quel nome.
- **Prove mie.** Con la disposizione reale (14 shard; per cella e compito 2, 1, 4, 4, 2, 1) gli aggregati sono
  identici a quelli costruiti da un solo shard: stessi array OOF, stessa impronta, stesse C, stessi score. Otto
  rifiuti verificati, senza file lasciati: manca il fit 14, manca il fit 01, manca un fold, fold doppio da una copia
  dello shard, stesso shard due volte, shard di un'altra cella, di un altro compito, dell'altra vista.

### 2.2 Provenienza (`scripts/scaling_compare.py`, righe 54–227 e 252–274)

- **Catena.** Lo sha256 del manifest è sulla riga di comando; il manifest fissa piani, report della posa e sei
  aggregati; ogni aggregato fissa i propri OOF, audit e shard; ogni shard fissa i propri OOF e audit.
- **Il comparatore risale agli shard** (193–213): li rilegge dai pin dell'aggregato, ne ricontrolla hash e campi,
  richiede che i loro audit siano quelli dell'aggregato e che gli OOF, ordinati per riga, coincidano. Poi riconvalida
  ogni aggregato contro le cache (`validate_aggregate_report`). Prove mie: un aggregato che tace uno shard è rifiutato
  («aggregate audit differs from source shards»); uno shard con una sola predizione diversa, hash interni rifatti e
  pin riallineato è rifiutato («aggregate OOF differs from source shards»); senza riallineare il pin lo ferma l'hash.
- **Altro run, dose, viste.** Ogni record dell'audit porta sha256 di cache, checkpoint, proiezione, codice e piano, e
  deve coincidere col piano (382–389 di `scaling_probe.py`). Passo 5.000 in piano, ricevute, report e manifest.
  Vista naturale o proiettata: identità `(braccio, lettura)` della cella, larghezza per record, hash della proiezione;
  per la posa la proiezione compare solo nella cella `matched`. Stesso checkpoint per braccio in posa e sonde; il
  30M deve essere quello del contratto (71–77). Griglia esatta di sei coppie cella/compito (215).
- **Parziali.** Shard e aggregati devono avere `complete: true`; uno shard fallito scrive `failed.json` e non
  `shard.json`. Il confronto scrive l'uscita solo dopo tutti i controlli, compreso un secondo hash di ogni file
  fissato (267–273).
- **Che cosa resta affidato agli hash:** le predizioni (non ricalcolabili senza rifare i fit) e l'elenco degli shard,
  che l'aggregato registra dalla propria riga di comando. Il legame con gli shard accettati si chiude sui manifest
  (S-3 e §5).

### 2.3 Confronto (`scaling_probe.py` 497–548, `scaling_pose_readout.py` 214–266, `scaling_compare.py` 230–249)

- **Metrica e segno.** Punto = media dei richiami per classe sull'OOF riunito. Contrasti nell'ordine delle celle
  (30M naturale, 300M naturale, 300M proiettato): `natural_300m_minus_30m` = (−1, +1, 0), quindi positivo se il 300M
  è migliore. Posa: errore del candidato meno errore del controllo, quindi negativo se il 300M è migliore.
- **Pesi.** P1: 8/13, 3/13, 2/13 per `ninapro_db2`, `db3`, `db6`; P2: 0,5 e 0,5 per `epn612` e `uci_emg`; applicati
  ai punti e, replica per replica, alle estrazioni.
- **Bootstrap appaiato.** Una sola matrice di estrazioni per dataset, usata per le tre celle; prima si richiede che
  righe, etichette, utenti e fold coincidano fra le celle. Intervallo percentile 2,5–97,5 della differenza. Le classi
  assenti in una replica escono dalla media di quella replica; nessuna replica scartata, i conteggi sono riportati.
- **Posa.** Supporto e target con la stessa impronta nelle tre celle, conteggi e denominatori per utente identici,
  estrazioni condivise, predizioni e coefficienti `float64` e finiti anche nel riempimento.
- **Controllo a dimensione pareggiata.** Solo sotto `secondary`. La lettura usa `natural_300m_minus_30m` per
  EPN612, P1, P2 e posa.
- **Ricalcolo indipendente.** Ho rifatto la stima per un'altra via (conteggi con `add.at`, matrice delle
  molteplicità per replica, punti con `balanced_accuracy_score`) e, a campione, ricampionando davvero le finestre
  degli utenti estratti. Sulla scena dei test: tutte le 2.000 repliche. Su array sintetici con le dimensioni reali
  (314 utenti, 6 etichette testuali `<U9`, 47.100 finestre per EPN612; 38.544, 9.834, 33.961, 364 per gli altri):
  punti uguali entro 1e-12, contrasti e intervalli entro 1e-9, stesse estrazioni (`draws_sha256`), una cella in
  ordine di riga diverso allineata correttamente. `compare_arrays` impiega circa 5 s su un thread del Mac.

### 2.4 Interpretazione (`scaling_compare.py` 230–249, `scaling_reading.py`)

- **Convergenza.** `solver_converged` è vero solo se tutti e sei gli aggregati sono convergenti. Se è falso la
  lettura non è valida: raccomandazione `repair_evaluation_no_scientific_reading`, `failed_checks`, nessuna lettura
  primaria; i numeri restano nel report (prova mia e test di GPT). Per la posa un bordo irrisolto della griglia fa
  fallire `paired_comparison`: né la ridge né il confronto scrivono un'uscita.
- **Niente post hoc.** Endpoint primario e soglie sono costanti del codice. `scaling_reading.py` non cambia dal 04/11
  alle 18:00 (`5a7b384`); cache e letture del 300M sono del 05/11. Il comparatore ha quattro argomenti, lo stadio
  `compare` del dispatcher non accetta opzioni, nessuna variabile d'ambiente entra nei calcoli.
- **Limiti conservati.** Nei testi `scope` e `uncertainty`: un solo seme, soggetti di sviluppo, intervalli marginali
  separati, nessuna correzione per molteplicità, proiezione secondaria. `continuation_authorized` è sempre falso.
- **Da tenere accanto alla lettura** (dato, non giudizio): gli intervalli vengono da 314 utenti per EPN612, 32, 8 e 8
  per i tre NinaPro, 19 per UCI, 15 per la posa.

### 2.5 Rilascio operativo (`scripts/slurm/scaling_readers_recovery.sbatch`)

- **`probe-aggregate`:** argomenti `<piano delle sonde> <sha256> <uscita>` più esattamente `--cell`, `--task`,
  `--shards`; 2 CPU, 30G, limite uguale a 600 s; uscita e due file accessori (`.oof.npz`, `.audit.jsonl`) nuovi e
  fuori dal codice; shard come percorsi assoluti canonici di file esistenti, senza ripetizioni.
- **`compare`:** argomenti `<manifest> <sha256> <uscita>`, nessuna opzione; 8 CPU, 30G, limite uguale a 1.800 s; il
  runtime corrente (versioni, thread, librerie numeriche) deve essere quello registrato dalla ridge, a otto thread.
- **Per entrambi:** partizione `lrd_all_serial`, nodo `login08`, `EXPECTED_SHA` uguale ad `aef665b…`, nessuna GPU,
  piano fissato per hash e ricontrollato dopo l'esecuzione. L'intestazione dello script chiede 8 CPU e 3 ore: senza
  le opzioni esplicite sulla riga di `sbatch` il dispatcher rifiuta.
- **Fattibilità sui dati reali.** `runtime_coherence` passa col runtime reale della ridge e con gli shard raccolti;
  `validate_pose_selection` passa per le tre celle. Le validazioni non dipendono dal numero di thread (uguaglianze di
  hash e di conteggi), quindi l'aggregato a 2 CPU non rischia confronti numerici falliti.
- **Tempi, stimati e non misurati.** Nei benchmark reali caricare e verificare due volte una cache, scaler compreso,
  è costato al più 30 s per dataset: aggregati sotto i 2 minuti contro 10. Confronto: 6–12 minuti contro 30 (ricalcola
  più volte gli hash di 16,4 GiB di cache e rifà il riepilogo della posa; l'intera ridge è durata 289 s). Memoria:
  picco della ridge 7,7 GB, dei fit 5,3 GB, contro 30G.
- I manifest e i comandi dei due stadi non esistono ancora: le condizioni del prompt (pin reali pubblicati, output
  nuovi, budget residuo, submitter e ledger comuni, `/dev/stdin`, nessun `afterok` al posto della validazione,
  nessun retry cieco) sono nel §5 come verifiche da fare.

## 3. Artefatti reali già prodotti (letture delle 08:50–09:11, score non aperti)

- **Ridge.** `…/cpu/pose_ridge/report.json` su Leonardo ha sha256 `8f70ff3b58c8f54413f6c8a89a861a4fdb7e6c9d934a65e73a11d4172a47e4ff`,
  uguale alla copia raccolta da GPT. Completo, passo 5.000, tre celle con 384, 896 e 384 feature, 40 utenti di
  addestramento, nessun bordo irrisolto; supporto e target identici nelle tre celle; 15 utenti di validazione, 2.000
  repliche, seme 20261102.
- **Fit.** 13 su 14 conclusi (01–13 COMPLETED 0:0; il 13 alle 08:36:37). Tutti `complete`, codice `aef665ba`, piano
  `d2cc7da1`, una sola firma numerica (`1ac8481e…`). 399 fit, zero avvisi di convergenza, zero al limite. Hash di
  OOF e audit ricalcolati uguali a quelli scritti nei report.
- **Identità degli OOF.** Per ogni cella e dataset completi: righe uniche e crescenti nel fold, utenti uguali a
  quelli del piano, un solo fold per utente. EPN612 ha etichette testuali `<U9` (`fist`, `noGesture`, `open`,
  `pinch`, `waveIn`, `waveOut`) nelle tre celle, con predizioni dello stesso tipo. Righe, etichette e utenti
  coincidono fra le celle in tutti i fold disponibili: 17 fold su 20 con tre celle; per `ninapro_db2` i fold 1–3
  hanno due celle, identiche, e aspettano il fit 14.
- **Impronte attese.** `identity_sha256` che ogni aggregato dovrà riportare, ricostruita dagli shard con la sola
  libreria standard; la ricostruzione è convalidata contro `aggregate` nelle mie prove (15 su 15).

| Dataset | Finestre OOF | Utenti | `identity_sha256` atteso, uguale nelle tre celle |
|---|---|---|---|
| `epn612` | 47.100 | 314 | `a828cda57ae0fef9c8145834a74235c379945f202f613d03cab687e16cfa8e35` |
| `ninapro_db2` | 38.544 | 32 | `b8d6e69fde0fa57dcc7623c7355c5b119f3526f418ba3f4a3b670822358b4367` (300M naturale: dopo il fit 14) |
| `ninapro_db3` | 9.834 | 8 | `056d02fd70736c75d54ec2610e541da808368f154f1ea9cf359ea27bc1cd41d6` |
| `ninapro_db6` | 33.961 | 8 | `3ee565080f079dfbecba717ef0c8238e2772efd41768c87548c973b4100b8f8e` |
| `uci_emg` | 364 | 19 | `132e4e8dbed63a9340585f235c8b68375364da0a0310827249446bddd85bd3fd` |

sha256 di `shard.json` letti alle 09:00 (ogni `shard.json` contiene gli hash del proprio OOF e del proprio audit):

| Fit | Cella, compito, dataset, fold | sha256 di `shard.json` |
|---|---|---|
| 01 | 300M naturale, p1, db2, 0 | `4cdd9e2a98cae6ccbb7d3fb9630b5fde5d14f3b46bacec2a19b9d659e0ca40a0` |
| 02 | 30M, p1, db3+db6, 0–3 | `0d4b5fa4972c1e5e9a93a8d7161a2a0578ff508148dac9f4a3b2ef0a803c177c` |
| 03 | 300M proiettato, p2, epn+uci, 0–3 | `e6b991d96c2f55b0875df9bd4ff8076a0a1f208f5cf5fd215ed737f33c8085a2` |
| 04 | 30M, p2, epn+uci, 0–3 | `5eecf3b5d8e5000e89fb2f565f4c035e052fe36836f953d8106c7c726f025731` |
| 05 | 300M proiettato, p1, db3+db6, 0–3 | `508bd22c7f19097ab4bbe6568e7e866e6baf05f15c3fe197e484fc062eb9303b` |
| 06 | 300M naturale, p2, epn+uci, 0 | `b4726e1f66d035e5b5ecffcb1e14b60d28cf6a83057fa49aaffd715cb7ce8956` |
| 07 | 300M naturale, p2, epn+uci, 1 | `bdc995cfff1d252706e64cb7fad9f9f5bb1ea19ee6dbbcfe62940c21cdae4e33` |
| 08 | 300M naturale, p2, epn+uci, 2 | `6a288a1376f6ae4f67a96af642f5e6985cba34d627621893be10aca4b7d47b10` |
| 09 | 300M naturale, p2, epn+uci, 3 | `81fb2707ce0cd9f8c2a4bf334ea59a582809328b14a359d1a69013c97debebd9` |
| 10 | 300M naturale, p1, db3+db6, 0–1 | `f1ce8c6d1956d57ad00bff726846a519d48092326524c85db5153259a1118208` |
| 11 | 300M naturale, p1, db3+db6, 2–3 | `77ba081458f978eb3a5969be6fa1940c1ea765459d636c5e7dd595cee8af889f` |
| 12 | 30M, p1, db2, 0–3 | `a96562fb4a6baa7b55272419128ac2f30ef545627cb6563754ffc2481c0a1322` |
| 13 | 300M proiettato, p1, db2, 0–3 | `b9dc0d8b9484ab847d5d009886708abc8caf51d642b34b2a7c7fe7c9c359dac3` |
| 14 | 300M naturale, p1, db2, 1–3 | in corso |

- **Budget.** Consuntivo alle 09:00: 114,42 core-ore (28,37 prima della ridge, cifra di GPT non ricalcolata oggi;
  0,64 la ridge; 85,40 i fit 01–13, da `sacct`). Il fit 14 costa al più 24,13 core-ore; aggregati e confronto al più
  2,6 e 4,4. Anche nel caso peggiore si resta sotto 150 core-ore contro il tetto di 310 approvato da Simone.

## 4. Suggerimenti facoltativi (nessun blocco; da non applicare ora al codice congelato)

- **S-1.** Il comparatore controlla di ogni shard cella, compito, codice e piano, ma non che la sua firma numerica
  sia quella dell'aggregato; lo fa `aggregate`, e il comparatore confronta fra loro solo le sei firme degli aggregati.
  Sui dati reali le 13 firme sono uguali. In una revisione futura: una riga in `_classification`.
- **S-2.** Il campo `bootstrap` del riepilogo è copiato dalla regola di lettura, non dai valori usati: il controllo
  di `read_comparison` su quel campo non può fallire. I valori usati coincidono per costruzione (2.000 repliche,
  seme 20261102 nelle costanti delle sonde e della posa; il report reale della ridge li registra). In futuro:
  derivare il campo da `n_bootstrap` e `seed` dei due blocchi.
- **S-3.** `probe-aggregate` riceve i percorsi degli shard senza i loro hash, e `aggregate` calcola l'hash di
  `shard.json` dopo averlo letto. Il comparatore rilegge tutto, quindi un cambio non passa inosservato; il legame con
  gli shard accettati va però chiuso sui manifest, confrontando `shards[*].sha256` con la tabella del §3.
- **S-4.** Il punto di ogni cella è calcolato due volte per vie diverse (nell'aggregato con scikit-learn, nel
  confronto dai conteggi per utente) senza un confronto fra i due. Nelle mie prove coincidono entro 1e-12: basta
  verificarlo sull'uscita finale stampando solo vero o falso.

## Note per il rilascio

- **R-1. Due tempi.** Il manifest del confronto contiene gli sha256 dei sei aggregati, quindi nasce dopo di loro:
  aggregati → raccolta e validazione → manifest pubblicato → confronto. Il confronto non si può accodare con
  `afterok` agli aggregati.
- **R-2. Coda seriale.** Decisione di Simone, 06/11 verso le 08:30, nel registro del coordinamento: «Sì, parte
  subito» (la linea 2 dopo il fit 14, fino a dieci posti). Alle 09:11 in coda c'era solo il fit 14. Dopo: verificare
  che i job dell'utente in coda siano meno di dieci prima di ogni invio; un rifiuto della QoS si registra e non si
  ripete alla cieca. La QoS concede 30.800 MB per utente (lettura mia del 05/11): i job da 30G girano uno alla volta,
  quindi i sei aggregati vanno in fila, dietro ai job già in attesa.
- **R-3. Riga di `sbatch`.** Aggregati: `--cpus-per-task=2 --mem=30G --time=00:10:00`. Confronto:
  `--cpus-per-task=8 --mem=30G --time=00:30:00`. In entrambi `/dev/stdin` come nome dello script prima dello stadio.
- **R-4. Limiti fissi.** I limiti di 600 e 1.800 s sono nel dispatcher: un TIMEOUT non si risolve alzando il tempo
  senza un dispatcher nuovo e la sua review. Lo stesso vale per un rifiuto del controllo sul runtime (ambiente
  cambiato dopo la ridge): fermarsi e decidere, non ripetere.

## 5. Verifiche ancora necessarie sui manifest finali

**A. Prima di inviare gli aggregati**
1. Fit 14 (63635524) COMPLETED 0:0 e accettato come gli altri: `shard.json` presente, `complete`, codice `aef665ba`,
   piano `d2cc7da1`, firma numerica `1ac8481e…`, 21 fit, nessun `failed.json`. Rieseguendo `…_oof_identity.py`:
   28.655 righe nei fold 1–3 e impronta `b8d6e69f…` per `ninapro_db2` anche nel 300M naturale.
2. Sei comandi pubblicati in un checkout prima dell'invio, nella forma
   `/dev/stdin probe-aggregate <piano delle sonde> d2cc7da1cf08473067d96b52bff532463453ff5ce633814743b1a53dda5bbb63 <uscita nuova> --cell … --task … --shards …`,
   con `CODE_DIR` del checkout `scaling_300m_evaluation_20261105_v1`, `EXPECTED_SHA=aef665ba7ea66fc80291846487907bb3ef9107df`
   e le opzioni della R-3.
3. `--shards`: i 14 `shard.json` pubblicati, ciascuno una volta. 30M p1: fit 02 e 12; 30M p2: fit 04; 300M naturale
   p1: fit 01, 10, 11, 14; 300M naturale p2: fit 06, 07, 08, 09; 300M proiettato p1: fit 05 e 13; p2: fit 03.
4. Sei uscite nuove con sei radici diverse (i file accessori si chiamano `<radice>.oof.npz` e `<radice>.audit.jsonl`),
   fuori da `$WORK/wearusfm` e `$WORK/wearusfm_campaigns`.
5. Script inviato con sha256 `c2bd7979699b2d70150cdfad9d2a6eca657615551ba5073630417c8c46cf14d5`; submitter comune
   (`814da4bc…`), ID logici nuovi nel ledger, prenotazione sotto lock; posti liberi in coda (R-2).
6. Budget: tetto 2,6 core-ore per i sei aggregati e residuo ricalcolato contro 310, col consuntivo del fit 14.
7. Nessuna catena `afterok` dal fit 14 al posto dell'accettazione dei suoi artefatti.

**B. Dopo gli aggregati, prima del manifest del confronto**
1. Sei job COMPLETED 0:0; sei report `complete`, `kind: aggregate`, stesso codice, stesso piano, firma `1ac8481e…`;
   84 fit per P1 e 56 per P2.
2. `shards[*]`: i percorsi del punto A.3 e gli sha256 della tabella del §3, più quello del fit 14 alla raccolta.
3. `identity_sha256`, finestre e utenti per dataset uguali alla tabella del §3 e uguali fra le tre celle.
4. Hash di `.oof.npz` e `.audit.jsonl` ricalcolati alla raccolta e uguali a quelli nel report.
5. `reading_status` di ciascuno. Se uno è `solver_unresolved` il confronto non darà una lettura: rifare fit è un
   esperimento nuovo, con la sua decisione.

**C. Manifest e invio del confronto**
1. Manifest: `protocol` `scaling-comparison-inputs-20261104-v1`, `checkpoint_step` 5000 intero, sei pin distinti
   degli aggregati, pin del piano delle sonde (`d2cc7da1…`), del piano della posa
   (`a500cc3c56bde3938a2aa384bf6a942fd6eaf02be0ebd5935ac251f732cb2198`) e del report della ridge (`8f70ff3b…`); pin
   con le sole chiavi `path`, `sha256`, `bytes`; percorsi assoluti canonici.
2. `sources` del manifest, sei voci:
   `scripts/scaling_compare.py` `a3610519b453a3d37747f5a4ce926b2164a40dd503546b22b6c45829085eb938`,
   `scripts/scaling_pose_readout.py` `238a5cd479e37310235c42dcb0b930d232230ff0d2a8d1d1309b05b7292900cf`,
   `scripts/scaling_probe.py` `0b3d40994ea7b6ee189a5831e22a5b298cc0cf2ceaa24908c80242642b5d26b0`,
   `scripts/scaling_projection.py` `a8c0f7880095df02186e465a1ce86030b8b2fe5ee1beb8a965b349ff205ee88d`,
   `scripts/scaling_reading.py` `13231e82664cfa6a815a8e87fb6c85380b70838d8746109d8af627b5a4f056d5`,
   `scripts/scaling_train.py` `d641c375cbeeb80bf275afef185dfb4f695938b22d908626cb9b829ddc8440ac`.
3. Manifest pubblicato su Leonardo prima dell'invio; comando `/dev/stdin compare <manifest> <sha256> <uscita nuova>`
   con le opzioni della R-3 e nessun'altra.
4. Budget: tetto 4,4 core-ore e residuo contro 310. Nessun `afterok` dagli aggregati (R-1).

**D. Dopo il confronto**
1. Job COMPLETED 0:0; uscita `complete`, `inputs_content_sha256` del manifest inviato, `sources` del punto C.2,
   codice `aef665ba`.
2. `quality`: se `solver_converged` è falso non c'è lettura, e i numeri restano descrittivi.
3. `n_bootstrap` 2000 e `seed` 20261102 nel blocco di classificazione e nei tre gruppi della posa (S-2).
4. Utenti per dataset 314, 32, 8, 8, 19; `replicates_missing_classes` e `single_class_replicates` riportati.
5. Coerenza, stampando solo vero o falso: il punto di ogni cella e dataset coincide con `balanced_accuracy`
   dell'aggregato entro 1e-12 (S-4); `epn`, `p1`, `p2` sono i contrasti `natural_300m_minus_30m`.
6. `continuation_authorized` falso: il seguito a 19.618 richiede una decisione di Simone.
7. Raccolta sul Mac di report, audit e manifest; non degli `.oof.npz`, come già fatto per gli shard.

## 6. Testo per GPT

```
Review 10 di Claude, codice degli stadi finali dello scaling (ultimo commit coperto 97125bd, codice aef665b):
APPROVATO, nessun blocco, nessuna correzione richiesta. Documento:
docs/reviews/claude_review_scaling_300m_final_stages_code_20261106.md sul ramo claude/review-scaling-300m-20261104.

Copertura: identita' del codice con aef665b confermata (in scripts/, src/, tests/ cambiano solo il dispatcher di
recupero e tre test dei launcher); i tuoi 165 test passano; nove prove mie sulla disposizione reale dei 14 fit
(aggregati e confronto identici a quelli da un solo shard, otto rifiuti di buchi, doppioni e shard estranei, il
comparatore rilegge gli shard, punti e intervalli ricalcolati per un'altra via entro 1e-9); 13 fit reali su 14 letti
senza aprire gli score: 399 fit convergenti, una sola firma numerica, righe, etichette e utenti identici fra le tre
celle. NON coperti perche' non esistono: fit 14, aggregati, manifest, confronto, comandi dei due stadi. La review non
approva manifest futuri e non autorizza il seguito a 19.618.

Quattro suggerimenti facoltativi, da NON applicare ora al codice congelato: S-1 firma numerica degli shard non
ricontrollata dal comparatore; S-2 campo bootstrap del riepilogo copiato dalla regola; S-3 probe-aggregate riceve
percorsi senza hash; S-4 punto calcolato due volte senza confronto.

Per il rilascio:
1. Due tempi: aggregati, raccolta e validazione, manifest pubblicato, confronto. Niente afterok fra aggregati e
   confronto, ne' dal fit 14 agli aggregati al posto dell'accettazione.
2. Riga di sbatch: aggregati --cpus-per-task=2 --mem=30G --time=00:10:00; confronto --cpus-per-task=8 --mem=30G
   --time=00:30:00; /dev/stdin prima dello stadio. Senza queste opzioni il dispatcher rifiuta.
3. Shard per aggregato: 30M p1 fit 02+12; 30M p2 fit 04; 300M naturale p1 fit 01+10+11+14; 300M naturale p2 fit
   06+07+08+09; 300M proiettato p1 fit 05+13; p2 fit 03. Uscite nuove con sei radici diverse.
4. Dopo gli aggregati confronta shards[*].sha256 e identity_sha256 con le due tabelle del §3 della review (impronte
   attese: epn612 a828cda5..., ninapro_db2 b8d6e69f..., ninapro_db3 056d02fd..., ninapro_db6 3ee56508..., uci_emg
   132e4e8d..., uguali nelle tre celle).
5. Manifest del confronto: sources con i sei hash del §5 C.2, piano della posa a500cc3c..., report della ridge
   8f70ff3b..., sei pin distinti degli aggregati, percorsi assoluti canonici.
6. Coda seriale: la linea 2 parte dopo il fit 14 fino a dieci posti (decisione di Simone); verifica i posti liberi
   prima di ogni invio; i job da 30G girano uno alla volta. Un rifiuto, un TIMEOUT o uno stop del controllo sul
   runtime non si ripetono alla cieca: i limiti di 600 e 1.800 s sono nel dispatcher.
7. Budget: 114,42 core-ore consumate alle 09:00, fit 14 al piu' 24,13, stadi finali al piu' 7,0: sotto 150 contro 310.
L'elenco completo delle verifiche sui manifest e' nel §5 della review.
```
