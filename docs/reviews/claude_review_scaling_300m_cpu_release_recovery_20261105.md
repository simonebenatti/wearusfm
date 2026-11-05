# Review Claude — rilascio CPU dei lettori 30M/300M a 5.000 e recupero dopo il TIMEOUT (05/11/2026)

Review 7 della linea scaling. Due richieste di Simone del 05/11 sera: «Sì, rivedili» sui quattro commit del rilascio CPU
(`1bde83e`, `26781ee`, `8cdabd6`, `76a3126`) e, a review in corso, il prompt di GPT
`docs/prompt_claude_scaling_cpu_recovery_20261105.md` sul commit `03cbd18` (TIMEOUT e piano di recupero).

## VERDETTO

**Lavoro eseguito: AFFIDABILE, niente da rifare. Diagnosi di GPT: CONFERMATA con ricalcolo indipendente.
Piano di recupero: APPROVATO nell'impostazione, con tre condizioni da chiudere prima degli invii (B-1, B-2, B-3).**
Questa review non è un via ai job: il dispatcher del recupero non esiste ancora e va rivisto quando c'è.

| Area | Esito |
|---|---|
| Piani CPU, checkout, validazione e benchmark della posa (`1bde83e`…`76a3126`) | CONFORMI, verificati sugli originali remoti |
| Diagnosi del TIMEOUT 63593508 e conto delle 148,13 core-ore (`03cbd18`) | CONFERMATI |
| Riuso dei tre report NinaPro del job in TIMEOUT | CORRETTO, con le guardie di F-1 |
| Recupero dei soli EPN612 e UCI 300M naturale, tetti 120 e 10 minuti | ADEGUATO |
| Tetto dei fit e tetto CPU complessivo | DA DECIDERE: 136 e 180 core-ore non bastano (B-1) |
| Limiti cablati nel launcher congelato | Riguardano anche i fit: servono almeno 16 shard, non 15 (B-2) |
| Congelamento condizionale S8 a 19.618 (`76a3126`) | Calcolo GPU confermato; due correzioni prima del primo fit (B-3) |

## 0. Perimetro, copertura e metodo

- **Copertura:** i commit `1bde83e`, `26781ee`, `8cdabd6`, `76a3126` e `03cbd18` del ramo `codex/scaling-300m-20261104`.
  In `c74d207..03cbd18` nessun file fuori da `docs/` e `results/` è cambiato: codice numerico e launcher sono quelli della
  review 6 (launcher `scripts/slurm/scaling_readers.sbatch`, sha256 `2da66c15…`, uguale ad `aef665b`, `c74d207`, `76a3126`).
- **Fuori copertura:** `6e57e19` e `d2c099b`, pubblicati alle 19:07–19:10 a review in corso. Toccano solo
  `tests/cpu/test_pose_b19618_launcher.py` e due voci di registro; non riguardano la catena CPU e non li ho rivisti.
- **Metodo, tutto in sola lettura:** git; su Leonardo `sacct`, `squeue`, `scontrol show node/partition/config`,
  `sacctmgr show qos/assoc`, `find`, `sha256sum`, `git rev-parse/status` nei due checkout di campagna, copia sul Mac della
  directory dei benchmark (317 KB). Script miei fuori dal repo; i conti sono rifatti dai file originali, non dai riepiloghi.
- **Non fatto:** nessun job, nessuna modifica; le cache non sono state aperte (solo hash); la suite intera non è stata
  rieseguita; il dispatcher del recupero non è stato rivisto perché non è ancora scritto.

## 1. Rilascio CPU (`1bde83e`, `26781ee`, `8cdabd6`, `76a3126`): conforme

- **Piani e checkout.** `pose_plan.json` `a500cc3c…`, `probe_plan.json` `d2cc7da1…`, `submission_plan.json` `f6d0c0e7…`,
  uguali a `1bde83e` e alla punta. Su Leonardo il checkout dei piani è a `1bde83e`, quello del codice ad `aef665b`,
  entrambi staccati e puliti; gli hash remoti dei due piani e del launcher coincidono.
- **Pin.** Ho ricalcolato su Leonardo, alle 19:12–19:15, lo sha256 delle **quattordici cache** (16,44 GiB): 14 su 14 uguali
  ai pin dei piani. Le ricevute pinnate sono quelle che ho verificato nella review 6 (`4634ac05…`, `0cf0d507…`,
  `03abef46…`, `172aa79b…`, `a57c9420…`); checkpoint `49dd518b…` (30M) e `37279995…` (300M); piano estrazioni `8b358951…`.
- **Job** (`sacct`, tutti su `login08`, `lrd_all_serial`, inviati col submitter comune e `--agente gpt`):

| Job | Stadio | Stato | Durata / limite |
|---|---|---|---|
| 63591447 | checkout dei piani | COMPLETED 0:0 | 44 s / 5 min |
| 63591791 | validazione delle quattro cache posa | COMPLETED 0:0 | 45 s / 30 min |
| 63592426, 63592917, 63593029 | benchmark posa, tre viste | COMPLETED 0:0 | 22, 30, 29 s / 10 min |
| 63593262 | benchmark sonde 30M naturale | COMPLETED 0:0 | 33:38 / 45:00 |
| 63593593 | benchmark sonde 300M pareggiata | COMPLETED 0:0 | 36:34 / 45:00 |
| 63593508 | benchmark sonde 300M naturale | **TIMEOUT** | 1:33:04 / 1:30:00 |

- **Posa.** Report di validazione `b0af2231…` e i tre report di benchmark: originali remoti uguali ai pin di
  `pose_budget.json`. Ho replicato fuori linea, sulle copie pubblicate, il gate del launcher (righe 186–230): tutti i
  controlli passano (identità del job come stringa, 2 e 8 CPU, supporto 40 utenti e 4.813.620 elementi, 384/896/384
  feature, pin degli input uguali al piano); stima 353,579 s, quindi `ceil60(1,5·stima+120)` = **660 s**, come dichiarato.
  Il runtime numerico registrato è identico nei tre report (8 thread, cinque pool a 8). La prova del runtime vivo resta
  quella dello stadio ridge (O-6 invariata).
- **Nessuno score aperto.** Validazione con `scientific_metrics_computed: false`; benchmark posa «fixed alpha 1e5; train
  only; no validation or CV score»; benchmark sonde solo train, fold 0, nessuna selezione di C. Alle 19:06 nessun job
  SCALING in coda e nell'albero degli output ci sono solo validazione e benchmark: nessun fit è partito.
- **Test.** Comando dei 16 file registrato da GPT, rieseguito su `76a3126` con un mio percorso JUnit: **571 passati in
  177,83 s**. `03cbd18` cambia solo documenti e risultati che nessun test legge.

## 2. TIMEOUT e secondo blocco (`03cbd18`): diagnosi confermata

- **Hash e contenuto.** I 27 file di benchmark e audit raccolti da GPT sono uguali, byte per byte, alle copie che ho
  scaricato io; i 13 `benchmark.json` hanno lo sha256 che ho letto sugli originali remoti.
- **83 fit** (78 nei 13 report, 5 nell'audit parziale di EPN612): tutti convergenti, nessun avviso, nessuna eccezione,
  nessuno a 12.000 iterazioni. Contesto numerico unico `1ac8481e…`. Nessuna metrica di validazione o held-out nei record.
- **Contabilità.** 2.018, 5.584 e 2.194 s per 8 CPU: 4,484444 + 12,408889 + 4,875556 = **21,768889 core-ore**; col resto
  (0,258889) il consuntivo CPU-only è **22,027778 core-ore**. Picchi 2,634 / 3,839 / 2,669 GiB su 30. Confermo che il padre
  in TIMEOUT non è un successo e che il costo va contato per intero.
- **Limite inferiore dei fit.** Formula di `scaling_probe.py::validate_resources` (righe 253–287) sui 13 report:
  66.660,69 s, cioè **148,134869 core-ore** (38,700857 + 67,720770 + 41,713242), prima dei 120 s per shard e dei due
  dataset mancanti. Il termine di ogni coppia cella/dataset è una somma sui fold, quindi il totale non dipende da come si
  spezzano gli shard: spezzare aggiunge solo 120 s per shard. Il tetto 136 è irraggiungibile con qualunque partizione.

Tempi interni dei benchmark, in secondi:

| Dataset | 30M (768) | 300M pareggiata (768) | 300M naturale (1.792) | naturale / 30M |
|---|---:|---:|---:|---:|
| ninapro_db2 | 1.019,8 | 1.100,1 | 1.630,0 | 1,60 |
| ninapro_db3 | 180,1 | 254,8 | 379,8 | 2,11 |
| ninapro_db6 | 254,7 | 304,3 | 1.553,7 | 6,10 |
| epn612 | 524,9 | 495,0 | incompleto: cinque C in 1.422,5 s (30M: 262,6 s) | 5,42 sui cinque C |
| uci_emg | 1,8 | 1,8 | non iniziato | — |

Quattro fatti che aggiungo alla diagnosi:

1. **I 184 s oltre il limite sono spiegati:** `scontrol show config` riporta `OverTimeLimit = 3 min` (5.400 + 180 = 5.580;
   cancellazione a 5.584). Ogni TIMEOUT costa quindi il limite più circa tre minuti.
2. **Il limite di 90 minuti presumeva un rapporto 2 fra 300M naturale e 30M** (5.400 contro 2.700 s, riga 116 del
   launcher). I rapporti misurati vanno da 1,6 a 6,1. Quel limite era una stima fissata nel codice prima di ogni misura
   300M, e nella review 6 ho approvato il launcher senza segnalarlo: l'omissione è mia.
3. **C=100 di EPN612 girava da almeno 559 s** alla cancellazione (ultima scrittura dell'audit alle 17:55:20,
   cancellazione alle 18:04:40). È una mia inferenza dai tempi dei file, non un dato del report.
4. **Più di metà del tempo CPU è tempo di sistema** in tutti e tre i job (`sacct`): 53,7 %, 59,4 % e 55,2 %. Vedi F-3.

## 3. Fabbisogno dei fit con la formula congelata (conto mio)

Riproducibile con `docs/reviews/claude_review_scaling_300m_cpu_release_recovery_20261105_fits.py <directory>`, dove la
directory è `results/infra/scaling_300m_20261104/cpu_readers_20261105/probe_benchmark` del ramo di GPT a `03cbd18`.
Lo script legge solo i tempi dei benchmark.

EPN612 300M naturale, secondo quanto dura C=100 (i cinque C misurati restano 1.422,5 s):

| Ipotesi su C=100 | Quattro fold | Totale dei fit, 16 shard |
|---|---:|---:|
| limite inferiore, 559 s | 35,50 core-ore | **187,90** |
| 1,5 volte C=10 (1.012 s) | 46,07 | 198,47 |
| 2 volte C=10 (1.350 s) | 55,09 | 207,49 |
| 12.000 iterazioni al ritmo di C=10 (1.569 s) | 60,94 | 213,34 |

- **Numero di shard.** Ho cercato per forza bruta la partizione minima in rettangoli dataset × fold, cella per cella:
  con 7.200 s per shard servono **almeno 16 shard** (30M: 2 + 1; pareggiata: 3 + 1; 300M naturale: 5 + almeno 4), contro
  il massimo di 15. Con 10.800 s ne bastano 12, con 14.400 s (il massimo della QoS) 10.
- **Un fold di EPN612 300M naturale entra in 7.200 s** solo se C=100 resta sotto circa 1.640 s: nello scenario peggiore
  del conto il margine è di 208 s.
- **Tetto complessivo.** Speso 22,03; recupero fino a 17,33; ridge 8; aggregati 2; confronto 4; checkout 0,33; fit
  almeno 187,90. Totale dei tetti **almeno 241,6 core-ore, 252–267 negli scenari**, contro 180. Zero GPU. Il consumo
  reale atteso è più basso, perché la formula ha un margine di 1,5.
- **Vincoli della QoS `lrd_all_serial`** (`sacctmgr`): 8 CPU e 30.800 MB per utente, quindi un job alla volta; al massimo
  10 job sottomessi insieme; 4 ore per job. I fit girano in serie: il solo walltime prenotato supera le 23 ore.

## 4. Risposte alle cinque domande del prompt

1. **Hash, 13 report, 83 fit, runtime, accounting:** riprodotti, tutti concordi. D'accordo anche sui limiti della
   verifica fuori linea; in più ho ricalcolato gli hash remoti delle 14 cache e letto lo stato dei checkout.
2. **Limite inferiore:** 148,134869 core-ore riprodotto; l'argomento di additività sopra mostra che non dipende dal
   numero di shard. Con l'audit parziale di EPN612 il limite sale ad almeno 187,9.
3. **Riuso dei tre report NinaPro: corretto.** Ogni dataset è un processo figlio separato che scrive `benchmark.json`
   solo dopo fit, seconda verifica della cache, contesto numerico e identità di codice e piano; il launcher ricontrolla il
   piano prima del figlio successivo, quindi DB2, DB3 e DB6 hanno avuto le stesse guardie di un job riuscito. Il
   controller dei fit ricalcola l'hash di ogni report e ne confronta il contesto numerico con quello vivo. Guardie: F-1.
4. **Recupero limitato e tetti:** adeguati. EPN612 dovrebbe durare 41–51 minuti (cinque C misurati più C=100 fra 1.012
   e 1.569 s): 120 minuti danno un margine di 2,4–2,9. UCI sta sotto il minuto. Conviene dichiarare il caso peggiore con
   i tre minuti di tolleranza: 18,13 core-ore, non 17,33.
5. **Sequenza e formula:** d'accordo su recupero, poi budget completo, poi fit, senza toccare formula, griglia, fold o
   tolleranze. Effetti su S8: vedi B-3.

## 5. Condizioni da chiudere prima degli invii

- **B-1 — Tetto CPU: serve una decisione registrata di Simone sui numeri nuovi.** Il fabbisogno dei fit (almeno 187,9
  core-ore) supera il tetto 136, il totale (almeno 241,6) supera l'inviluppo 180, e la riga «benchmark sonde» passa da
  24 a 39,1 col recupero. `CLAUDE.md`: «Superare il budget di un passo richiede una decisione registrata, non una
  conferma al volo». La frase della diagnosi «Nessun nuovo consenso per ripetere il piano già autorizzato» vale per i
  job già approvati, non per un tetto più alto. Correzione minima: pubblicare l'inviluppo emendato (benchmark, fit,
  totale) e farlo approvare a Simone una volta sola, in `docs/decisioni.md`, prima del primo fit. I due benchmark di
  recupero restano dentro il totale 180 (cumulato fino a 39,4): per quelli basta registrare l'emendamento della riga.
- **B-2 — Il dispatcher deve coprire anche i fit, non solo il benchmark.** I limiti sono nel launcher congelato:
  riga 116 (5.400/2.700 s) e 249 (cinque dataset per job), riga 125 (al massimo 15 shard), riga 129 (7.200 s per shard),
  riga 137 (136 core-ore). Con 7.200 s servono almeno 16 shard, quindi anche il limite di 15 è irraggiungibile.
  Correzione minima: un solo dispatcher derivato dal launcher con un diff minimo e pubblicato (benchmark per singolo
  dataset col suo walltime; limiti nuovi dei fit: totale, numero di shard, walltime per shard entro le 4 ore della QoS),
  tutto il resto identico (moduli, venv, thread, bootstrap, controlli dello scheduler, regole sugli output); `CODE_DIR`
  ed `EXPECTED_SHA` fermi ad `aef665b`, altrimenti i 13 report non passano più `validate_resources`; un test che mostri
  che il diff è solo quello dichiarato; review prima dell'uso. La regola dei nuovi limiti va pubblicata prima di
  misurare EPN612, perché non sembri adattata al risultato.
- **B-3 — Congelamento S8 (`76a3126`): due correzioni prima del primo fit.** (a) `boost_qos_bprod` non è ammissibile
  per un job da un nodo: `sacctmgr` riporta `MinTRES = cpu=2080,gres/gpu=260,mem=31280G,node=65` con `DenyOnLimit`; la
  lettura dello scheduler pubblicata non interrogava `MinTRES`. Le 14:45 ore rientrano nelle 24 della partizione con la
  QoS predefinita. (b) I tetti dei lettori nel JSON (180 totale, 24 benchmark, 136 fit) sono già smentiti dalle misure a
  5.000. Vanno emendati ora, su evidenza solo tecnica, oppure va dichiarato che la continuazione si fermerebbe ai
  lettori: dopo il primo fit il documento stesso non consente più di presentarli come fissati prima degli score. La
  parte GPU è confermata dal mio ricalcolo: 14.665,05 s e 2,7667 s danno 53.100 s, 59 + 7 = **66 GPU-ore**.

## 6. Facoltativi

- **F-1 — Guardie per il riuso.** Nel piano risorse, o in un file di accettazione pinnato, registrare per ciascuno dei
  15 report percorso, sha256, job e stato del padre (TIMEOUT per i tre NinaPro di 63593508) con la prova che il figlio
  era finito (riga `complete` nel log; scritture alle 16:59:02, 17:05:28 e 17:31:26, prima della cancellazione). Il
  recupero scrive in directory nuove, mai dentro `probe_benchmark/T100-G-300M-natural/`, dove l'audit parziale resta
  come prova. Prima dei fit: `numeric_sha256` dei due report nuovi uguale a `1ac8481e…`.
- **F-2 — Tolleranza dello scheduler.** Calcolare i casi peggiori come CPU × (walltime + 180 s).
- **F-3 — Efficienza, ipotesi non misurata.** `login08` ha due thread hardware per core, quindi le 8 CPU allocate
  sono con ogni probabilità 4 core fisici; alle 19:12 il nodo aveva carico 183,9 su 128 CPU. Otto thread numerici su
  quattro core in un nodo sovraccarico spiegherebbero il tempo di sistema sopra il 50 %. Cambiare i thread è però un
  cambio di runtime per tutte e tre le viste, con 15 benchmark da rifare e una decisione: lo segnalo, non lo chiedo.
- **F-4 — C=100 di EPN612 300M naturale può avvicinarsi alle 12.000 iterazioni** (a C=10 ne ha fatte 5.163 contro
  3.715 del 30M, che a C=100 ne ha fatte 7.422). Se le raggiunge vale la regola congelata, senza modifiche.
- **F-5 — `status.json`:** `cpu_reserved_core_hours_so_far = 29.5` non corrisponde più a nulla; il dato vivo è 22,027778.
- **F-6 — Review interne.** `scaling_cpu_release_20261105.md` e `scaling_cpu_timeout_critic_20261105.md` sono review
  interne di GPT, corrette nei fatti che ho ricontrollato; non sostituiscono questa (come O-2 della review 6).

## 7. Testo per GPT

> Review Claude 7 sul ramo `claude/review-scaling-300m-20261104`, file
> `docs/reviews/claude_review_scaling_300m_cpu_release_recovery_20261105.md`. Copertura: `1bde83e`, `26781ee`, `8cdabd6`,
> `76a3126`, `03cbd18`; non coperti `6e57e19` e `d2c099b`.
>
> Verdetto: rilascio CPU conforme, niente da rifare; diagnosi confermata (27 file uguali alle mie copie, 13 report uguali
> agli originali remoti, 83 fit, 21,768889 e 22,027778 core-ore, 148,134869 core-ore riprodotte; 14 cache riverificate
> per hash su Leonardo). Piano di recupero APPROVATO nell'impostazione: riuso dei tre NinaPro corretto, recupero dei soli
> EPN612 e UCI, tetti 120 e 10 minuti adeguati (caso peggiore 18,13 core-ore con `OverTimeLimit = 3 min`).
>
> Tre condizioni prima degli invii. B-1: il fabbisogno dei fit è almeno 187,9 core-ore (198–213 negli scenari) e il
> totale CPU almeno 241,6 contro 180: serve la decisione registrata di Simone sull'inviluppo emendato prima del primo
> fit. B-2: i limiti cablati nel launcher riguardano anche i fit; con 7.200 s per shard servono almeno 16 shard, non 15
> (12 a 10.800 s, 10 a 14.400 s); un solo dispatcher con diff minimo, `EXPECTED_SHA` fermo ad `aef665b`, regola dei
> limiti pubblicata prima di misurare EPN612, review prima dell'uso; la QoS ammette 10 job sottomessi e 8 CPU per
> utente. B-3: nel congelamento S8 `boost_qos_bprod` richiede almeno 65 nodi (`MinTRES`, `DenyOnLimit`) e i tetti dei
> lettori 180/24/136 sono superati: correggere prima del primo fit. La parte GPU (66 GPU-ore) è confermata.
>
> Facoltativi F-1…F-6 nel file. Quando il dispatcher e il budget completo sono pubblicati, mandami il commit.
