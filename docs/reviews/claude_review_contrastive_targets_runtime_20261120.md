# Review Claude: pacchetto operativo della coorte RMS100/fast125, linea 3 (`fbfde797`)

20/11/2026. Revisore: Claude. Linea 3: codice, protocollo e lettura spettano a GPT, i lanci a 3-launcher.
Claude non lancia job.

Nota sul formato: questo file e' volutamente in ASCII puro. `validate_review_binding` legge la copia della review con
la codifica del locale (rilievo R3-N2).
- **Quando un carattere non ASCII fa fallire i job:** con un locale a codifica ASCII che non attiva la modalita' UTF-8 di
  Python, per esempio en_US.US-ASCII.
- **Quando non fa danni:** con C/POSIX Python 3.11 passa da solo a UTF-8, e con ISO8859-1 la lettura riesce.

Per lo stesso motivo le citazioni sono trascritte con apostrofi e virgolette semplici.

## Richiesta

Prompt di GPT passato da Simone. La richiesta e' di rivedere il pacchetto su `codex/contrastive-target-windows-20261118`,
diff da `438c8d65` al `content_commit` del delivery_pin:
- "Pubblica in un solo passaggio i blocchi con prova riproducibile e le altre note";
- "Se approvi il pacchetto operativo, inserisci nella review un unico blocco" di release, "nessuno stato APPROVED prima
  del tuo verdetto".

Perimetro:
- **Contenuti:** `fbfde797b48644744c36547533acf0e8a7793535`, 46 file.
- **Pin:** `0efa4d45` aggiunge solo `amendment_20261119/delivery_pin.json`. Sorgenti, test e piani sono identici
  fra i due commit.
- **Fuori perimetro:** le parti storiche invariate non sono state riaperte.

Metodo:
- **Workflow di 85 agenti:**
  - un revisore per ciascuno dei cinque punti di GPT, uno per i lettori e uno per la governance;
  - da uno a tre scettici indipendenti per rilievo, secondo la gravita';
  - un critico di completezza.
- **Controlli miei:**
  - impronte del pacchetto ricalcolate;
  - API di torch usate dal profilo numerico, lette sul torch 2.2.0a0 di Leonardo;
  - percorsi `str` in ogni `torch.load` con `mmap`;
  - copia della mia review 94f5094 nel ramo, identica all'originale;
  - prova del blocco di release con `bind-review` in un clone temporaneo;
  - scansione delle evidenze pubblicate, perche' nessun file passi come approvazione.
- **Rilettura della bozza.** L'hanno controllata due verificatori indipendenti, su circa 315 affermazioni: errori e
  imprecisioni corretti qui.

## Esito

**Codice: nessun blocco tecnico.**
- **Risposta alla mia review del 19/11.** Il pacchetto implementa la proposta faa4a8d7 e risponde alle condizioni R-1...R-4,
  al rilievo minore M-1 e alle note N-1...N-7.
- **Un'eccezione nella risposta.** Le sue affermazioni corrispondono al codice e ai test, salvo una su R-4: l'accounting non
  lega davvero l'ID SLURM dell'audit, che resta solo dichiarato dall'operatore (G-6, R3-M3).
- **Test.** La ricevuta si riproduce: 465 test passati.

**Un blocco di governance (G-1), confermato da tre scettici su tre, chiuso il 20/11 dalla firma di Simone.**
- Alla prima pubblicazione di questa review (c591bda) la proposta scientifica faa4a8d7 e il nuovo tetto di 18 GPU-ore non
  avevano la firma di Simone.
- Il blocco di release non ha un ambito di fase: una volta pubblicato, rende eseguibile tutta la catena GPU, non solo
  l'audit.
- Per questo la prima versione non emetteva il blocco. Simone ha poi firmato entrambi (citazione in G-1): questa versione
  contiene l'unico blocco di release, nella sezione "Blocco di release".

**Rilievi minori e note.**
- **Conteggi.** Sopravvivono alla verifica 17 rilievi minori e 27 note; G-2 e' contato fra le note, come l'hanno valutato i
  due scettici.
- **Non sopravvissuti: 9 rilievi.** Otto confutati da tutti i loro scettici; R4-M1 con voto diviso 1 a 1, e il suo
  contenuto e' ripreso in G-1.
- **Effetto sui cancelli.** Nessuno altera lo stato complessivo dei cancelli. Due casi locali:
  - R2-N1 attribuisce male l'esito della singola coppia nella ricevuta;
  - R3-N1 farebbe prendere per approvazione una review non favorevole che contenga un esempio APPROVED.
- **Misure operative.** Alcuni rilievi si gestiscono senza toccare il codice: P-1...P-8.
- **Correzioni che cambiano il digest dei sorgenti:** M-a, M-b e M-d, piu' lo strumento di validazione (R5-6). Con il
  blocco emesso su `fbfde797` vanno nella release dei training, che richiedera' comunque una review nuova. Farle prima del
  preflight richiederebbe una review e un blocco nuovi.
- **Correzioni che non lo cambiano:** i test di M-c, CLAUDE.md e le note solo documentali.

## G-1. Blocco di governance: firma di Simone prima del blocco di release

**Cosa dice faa4a8d7:**
- intestazione "Decisione proposta a Simone";
- chiusura: "proposta da discutere prima degli invii";
- voce di `decisioni.md` sul ramo della linea 3: "Proposta per Simone".

**Cosa fa `fbfde797`.** La dichiara attuata (`decisioni.md`, piano, prontuario, prompt per Claude) e cambia due cose:
- la prova decisiva della parita': il gate sulle traiettorie diventa descrittivo;
- il punto di partenza della ripresa: unico fork a 544.

**Le citazioni di Simone del 19/11 registrate:**
- "Ti affido l'implementazione dell'emendamento RMS100/fast125 e la preparazione del pacchetto per 3-launcher";
- "Gli invii e i controlli su Leonardo spettano a 3-launcher".

Nessuna delle due approva il profilo numerico, i nuovi cancelli o il tetto.

**Il tetto:**
- richiesti 18 GPU-ore e 169,6 core-ore; con il margine di SLURM, 21,6 GPU-ore e 202,4 core-ore;
- l'ultimo tetto approvato per la linea 3 e' di 14 GPU-ore e 129,333333 core-ore. E' del 17/11: "direi che se tu sei a
  posto per questo giro lanci tu, ti do fiducia".

**Il blocco di release.** `contrastive_release.py` accetta solo le sei chiavi previste, quindi una condizione non si puo'
scrivere dentro il blocco. Una condizione scritta nel testo della review non vincola il codice.

**Cosa serve: una decisione di Simone registrata con citazione letterale.** Deve coprire:
1. il profilo bf16 autocast con SDPA solo math e TF32 spento per i tre bracci, dalla calibrazione ai training se
   sostenibile;
2. la nuova prova di parita': stato vivo del C0 prima degli update 128 e 256, gradienti ridotti identici per byte; le 256
   traiettorie diventano descrittive;
3. la ripresa da un unico fork a 544 con le regole storiche dell'1%;
4. il tetto di 18 GPU-ore e 169,6 core-ore, inviluppo 21,6 e 202,4.

Le core-ore coincidono con le ore locali, perche' nel billing storico conta solo il numero di CPU.

**Contabilita' del confronto contrastivo, training escluso:**
- consumato: 6,086667 GPU-ore e 57,827778 core-ore;
- massimo complessivo con il nuovo preventivo: 24,086667 GPU-ore e 227,427778 core-ore.

Il saldo del progetto va letto da 3-launcher prima del primo invio.

**Firma di Simone, 20/11/2026, in chat con Claude (risposta letta alle 09:57):**
- domanda di Claude: "Firmi la proposta faa4a8d7 e il tetto di 18 GPU-ore?";
- risposta di Simone: "Si', firmo entrambi". Nell'originale la i e' accentata; qui e' trascritta in ASCII, come dice la
  nota sul formato.

Lettura mia: la firma copre la proposta faa4a8d7 nel suo insieme e il tetto, cioe' i quattro punti elencati sopra. Il
messaggio a cui Simone rispondeva li descriveva cosi':
- profilo bf16 math con TF32 spento per i tre bracci;
- parita' sugli stati vivi del C0 agli update 128 e 256;
- ripresa da un unico fork a 544;
- tetto di 18 GPU-ore e 169,6 core-ore, inviluppo 21,6 e 202,4, con il saldo del progetto letto da 3-launcher.

La registrazione in `decisioni.md` spetta a GPT, responsabile della linea 3, prima del primo invio.

**Valori verificati per il blocco**, emesso dopo la firma:

| Campo | Valore | Verifica |
|---|---|---|
| schema | contrastive-target-release-review-v1 | `contrastive_release.py` |
| reviewed_commit | fbfde797b48644744c36547533acf0e8a7793535 | delivery_pin, `content_commit` |
| source_sha256 | 40d39f568958004415763f9cc683f3be0b379d170b70c4d37588d1ddcec5cde0 | `source_digest()` ricalcolato |
| science_plan_sha256 | b03c0bc679f7247d703635d7065c9fc7bda4a292f2c8c106e29004e0527e021a | sha256 di plan.json, 14.862 byte |
| preflight_plan_sha256 | e4b3959112db86dac806978cf65307d97694da68b64096120487e8f568c6dd92 | sha256 di preflight_plan.json, 9.082 byte |

Sono gli stessi a `fbfde797` e a `0efa4d45`, e coincidono con `preflight_plan.json`.

**Prova del formato:**
- **Formato atteso:** una sola apertura a colonna 0, la chiusura a colonna 0 e le sei chiavi.
- **Con il blocco.** In un clone temporaneo di `0efa4d45`, questa versione della review, letta dal suo commit e copiata nel
  checkout, supera `bind-review` sia con locale UTF-8 sia con locale C.
- **Senza blocco.** La prima versione (c591bda) e' rifiutata ("exactly one ... binding block").

**Scansione delle evidenze.**
- **Il file escluso.** Fra le evidenze degli agenti, una review sintetica di uno scettico, con le impronte vere, sarebbe
  stata accettata come approvazione. E' esclusa dalle evidenze pubblicate e resta nell'archivio locale (R3-N1).
- **Esito.** Nessuno dei 328 file delle evidenze pubblicate passa il parser. Sul mio ramo l'unico file che lo passa e'
  questa review; sul ramo di GPT a `0efa4d45` nessuno.

## Misure operative prima delle fasi GPU (senza toccare il codice)

- **P-1, corsa all'avvio (R2-M1).** Ogni rank controlla che la cartella d'uscita non esista prima di
  `init_process_group`.
  - Il rank 0, che ospita lo store di `torchrun --standalone`, esce da `init` solo dopo che tutti i rank si sono connessi.
  - Fra il controllo di un rank e il `mkdir` di un altro rank non zero, pero', non c'e' sincronizzazione.
  - In calibrazione, full e training un rank in ritardo puo' quindi fallire e far perdere la cella. Nei reload la corsa non
    c'e'.
  - Contro misura operativa: esportare `TORCH_DIST_INIT_BARRIER=1` nelle celle GPU. Dal 2023 PyTorch fa allora una barriera
    alla fine di `init_process_group`.
  - Correzione di codice, per il prossimo giro: una barriera esplicita dopo l'inizializzazione.
- **P-2, controllo del job prima di `sbatch` (R6-1, R4-N7, R5-9).** Il braccio della cella non e' legato a config e
  release eseguiti. Ricevuta CUDA e accettazione della calibrazione si verificano solo dentro l'allocazione GPU.
  - Prima di ogni invio, eseguire sul login node `build_command` e la validazione del runner.
  - Verificare a mano che il braccio di job, config e release coincida.
- **P-3, dipendenze (R4-N3).** Con `--dopo` verso un job gia' finito (MinJobAge di 30 s), `sbatch` rifiuta l'afterok,
  come gia' successo il 17/11.
  - Per prerequisiti finiti: sacct `COMPLETED 0:0` e SHA del checkout, senza afterok.
  - `--dopo` solo verso job ancora in coda.
- **P-4, override dei checkout (R4-N8).** Sono solo procedurali. Se i sei checkout partissero con 4 CPU e il tempo di
  default (10 min), il totale salirebbe a 172 core-ore, sopra il tetto di 169,6. I comandi definitivi devono riportare
  `--cpus-per-task=4 --mem=2G --time=00:04:00`.
- **P-5, orchestrazione mai eseguita.** L'orchestrazione reale di `contrastive_run.main` non e' mai stata eseguita, nemmeno
  su CPU. Comprende profilo, osservatore della parita', traccia della ripresa, `prepare_reload`, barriere e ricevute.
  - Le prime esecuzioni sarebbero quindi le celle di produzione, senza retry.
  - Prima della release della calibrazione conviene un'esecuzione CPU a quattro processi gloo di `main` per full C0 e
    reload. Non cambia il digest.
- **P-6, manutenzione.** Leonardo e' fermo dal 23/11 alle 08:00 al 25/11 alle 08:00. La catena ha circa 6,4 ore di
  allocazioni in serie e cinque rilasci intermedi a mano, con al piu' due job dbg per utente condivisi con le altre linee.
  Il calendario va fatto a ritroso da quella data.
- **P-7, fattibilita' della ripresa.** La regola dell'1% per singolo tensore di Adam e della coda, sulle coppie
  continuo/reload, potrebbe fallire in modo sistematico.
  - Nel continuo i bucket DDP sono gia' ricostruiti. Nei reload si ricostruiscono dalla terza iterazione dopo il fork
    (update 547): gli update 545-546 hanno un ordine di riduzione diverso.
  - Il profilo math non lo elimina.
  - Il piano prevede blocco e localizzazione. Conviene scrivere prima come si legge un fallimento con A e B concordi, come
    per R-1.
- **P-8, costi da misurare.** Memoria e tempo del profilo math a batch 32 si misurano per la prima volta nella
  calibrazione. Il tempo dell'osservatore entra nella misura del C0 (R1-M1 e R6-8): la proiezione del C0 e' prudente,
  circa +7%.
- **Altre lacune del critico,** senza un secondo giro di scettici:
  - in bf16 math il forward non e' deterministico (somme atomiche: loss diverse nei probe della diagnostica), mentre il
    gradiente risulta unico. La parita' per byte regge finche' quelle somme non entrano nel gradiente;
  - il supporto delle loss ai punti 127, 255 e 544-575 si puo' leggere gia' dall'audit, prima di spendere GPU;
  - per l'audit la stima di durata lascia un margine di circa 1,6-1,9 volte sui 7.200 s;
  - le ricevute numeriche non registrano numpy/scipy, cuDNN, NCCL, driver, GPU e host;
  - API di torch non lette sul cluster: `torch.load` con `mmap` e `weights_only`, `fork_rng(devices=[])`, la patch di
    `AdamW.__init__` e `is_bf16_supported` prima di `set_device`;
  - release e accounting si scrivono a mano, senza generatore (R3-N3);
  - il consumo GPU non e' raccolto con sacct, e un full entro l'OverTimeLimit verrebbe accettato (R6-7).

## Rilievi minori

M-a, M-b e M-d cambiano il digest dei sorgenti. M-c lo cambia solo per lo strumento di validazione (R5-6) o un eventuale
generatore (R5-7); i test aggiunti sono fuori inventario.

- **M-a, legame di release e job:**
  - R3-M1: la release non impone la discendenza dal commit rivisto ne' l'assenza di codice fuori inventario (per esempio
    `src/sitecustomize.py`);
  - R3-M3: l'ID SLURM dell'accounting non e' ancorato all'esecuzione dell'audit. `expected_job_id` non e' passato da nessun
    chiamante di produzione: lo passano solo i test;
  - R6-1: il braccio della cella non e' legato a config e release.
- **M-b, impronte della ripresa (R2-M2 e R6-2).** `read_preflight` sostituisce le impronte di `compare_three_paths` con
  quelle dei soli reload. La ricevuta finale non pinna le 24 tracce e i 4 `start.json`, cioe' proprio la prova della
  ripresa.
- **M-c, test:**
  - R3-M4: i test delle guardie di release non rilevano 25 mutazioni su 39;
  - R6-3: nei lettori sopravvivono 30 mutazioni su 40;
  - R5-1...R5-7: legame della review, dispatcher e runtime sostituiti da no-op, confine della ripresa, guardie
    dell'osservatore, accounting, `--verify`, integrazione produttori-lettori;
  - G-3: chiamate di R-2 e R-4 nei consumatori senza test.

  Le guardie principali (byte-identita', 1% e per tensore, sacct, 28 ricevute) sono invece coperte.
- **M-d, misura (R1-M1 e R6-8).** Il tempo dell'osservatore entra nella proiezione del C0.

## Note

- **Parita':**
  - R1-N2: il cancello non distingue le differenze entro braccio da quelle fra bracci;
  - R1-N5: la memoria per caso e' il picco cumulativo;
  - R1-N6: a peso zero i bracci differiscono solo per l'etichetta.
- **Ripresa:**
  - R2-N1: il "passed" di ogni coppia unisce errori di confine e di supporto;
  - R2-N2: la suite prova solo la ripresa a un rank e non legge le tracce DDP.
- **Release:**
  - R3-N1: il parser accetta piu' del formato documentato (regex non ancorata, chiavi duplicate): ancorare la regex e
    rifiutare i duplicati.

    Resta a parte il caso di una review non favorevole che contenga un esempio APPROVED: verrebbe presa per approvazione
    anche con la regex ancorata. Lo evita solo l'assenza di esempi nel file, un limite dichiarato nel prontuario.
    Un caso concreto e' stato trovato nelle mie evidenze, e il file e' stato escluso;
  - R3-N2: leggere la copia in UTF-8;
  - R3-N3: procedura delle ricevute per `make-job` e comando sacct da esplicitare;
  - R3-N4: `audit_sacct` non pinnato, `--result-out` non controllato;
  - R6-5 e G-8: il binding non dichiara la fase che approva.
- **Risorse:**
  - R4-N1: README della cartella dei piani con il preventivo vecchio;
  - R4-N4: i 1.770 s interni non valgono per la cella CUDA;
  - R4-N5: limiti per utente delle code non riportati.
- **Lettori:**
  - R5-8: un rapporto di parita' senza punti solleva un errore invece di INCONCLUSIVE;
  - R6-6: il profilo della calibrazione non viene riletto;
  - R6-7: il limite di tempo e' verificato con sacct solo per l'audit.
- **Documenti:**
  - G-2, CLAUDE.md:
    - il blocco aggiunto in testa al ramo, sotto il titolo "Linea3, incarico del19/11/2026", dice "Simone affida a questa
      chat";
    - la frase sulla precedenza non nomina la linea 3, lo fa solo il titolo;
    - i lanci sono gia' assegnati per nome a 3-launcher.

    Prima di un'integrazione su main conviene nominare la chat (worktree `wearusfm-contrastive-targets`) e limitare la
    precedenza alla linea 3;
  - G-4: il submitter non ricava lo SHA da `CAMPAIGN_SHA`, cerca `EXPECTED_SHA=`;
  - G-5: la ricevuta locale copre solo pytest;
  - G-6: l'ID SLURM dell'audit e' solo dichiarato;
  - G-7: la provenienza della decisione e' incompleta sul ramo che verra' congelato, perche' faa4a8d7 non e' un antenato.

## Verifiche positive, per punto

**1. Parita':**
- **Aggancio corretto.** Lo stato vivo del full C0 e' preso prima degli update 128 e 256: 127 e 255 update completati,
  batch globale 32.
- **Probe isolati.** I probe sono cloni isolati con DDP propri.
  - Prova del revisore: trainer reale, quattro rank gloo, batch 32, dropout 0,4. Il run con l'osservatore e' identico a
    quello senza: studente, teacher, ottimizzatore, coda e RNG.
  - Gli hook del DDP vivo non scattano e i `.grad` vivi restano vuoti.
  - Le due mutazioni che sporcano lo stato vivo vengono fermate.
- **Criterio conforme.** Profilo, supporto o memoria mancanti danno inconcluso; le 256 traiettorie restano descrittive.
- **Profilo applicato ovunque:** tutte le fasi e i probe, con 28 ricevute confrontate.
- **Compatibilita' con il cluster:**
  - le API del profilo numerico esistono nel torch 2.2.0a0 di Leonardo, oppure sono gestite con `getattr`/`hasattr` e
    con il ramo `TypeError` di `is_autocast_enabled`: lettura sul login node del 19/11;
  - le altre API di torch non sono state lette sul cluster (vedi sopra);
  - ogni `torch.load` con `mmap` passa il percorso come stringa.

**2. Ripresa:**
- **Un solo fork.** Il fork a 544 e' salvato dal full continuo, che prosegue fino a 576. 544 entra in `keep_checkpoints` solo
  nei config del collaudo.
- **Reload.** A e B partono dagli stessi byte, con quattro processi nuovi, e fanno solo 545-576.
- **Tracce e confine.** Tracce per rank e per update; confine esatto.
- **Regola storica.** E' identica a quella di `438c8d65`. Le distanze da 544 restano fuori dal cancello.
- **Confondimento risolto.** Il vecchio confronto fra training indipendenti non c'e' piu'.

**3. Release:**
- **Cosa legge bind-review:** i byte della review nel suo commit, la copia nel checkout, sorgenti e piani al
  `reviewed_commit`.
- **Rifiuti provati:** doppioni, stato diverso, chiavi in piu', altro pacchetto, review storiche.
- **Dispatcher e runtime** rifiutano binding mancanti o alterati.
- **Cancello sacct dell'audit:** COMPLETED, 0:0, al massimo 7.200 s, 8 CPU, 120 min. E' applicato in `make_job`, nel
  dispatcher, nel runtime della calibrazione e nella lettura finale; nessuna eccezione TIMEOUT.
- **Nessun hash circolare.**

**4. Risorse:**
- **Piani riproducibili.** `contrastive_targets_plan.py --check` rigenera i piani byte per byte.
- **Conti esatti:**
  - 17 allocazioni, 18 GPU-ore, 169,6 core-ore, 7,9 ore di walltime;
  - consuntivi ricostruiti dalle righe sacct del repository.
- **Dispatcher.** Accetta solo le allocazioni del piano.

**5. Test:**
- **Ricevuta riprodotta.** 465 test passati in 102,84 s, contro i 102,59 s della ricevuta, e `--verify` corrisponde.
- **M-1 chiusa.** Ogni barriera, disattivata da sola, fa fallire un test.
- **N-4 chiusa.** PLAN, testo del readout e campi target sono protetti.

**Lettori e governance:**
- **Lettori.** I rifiuti avvengono prima di consumare GPU. La lettura finale richiede tutte le evidenze, e un esito
  favorevole non rilascia training. Il CUDA cambia solo il contesto numerico, con criteri invariati.
- **Documenti.** La decisione faa4a8d7 e la mia review 94f5094 sono copiate byte per byte. `evaluation` e le soglie a valle
  sono invariate.

## Limiti

- **Esecuzione.** Niente CUDA, NCCL o cluster: osservatore, traccia e orchestrazione distribuita sono provati solo su CPU,
  e solo per parti (P-5).
- **Ultima lettura di Leonardo.** E' del 19/11. Oggi la connessione va in timeout, probabilmente per il certificato
  scaduto: il locale dei job e lo stato corrente non sono verificati.
- **Lacune del critico.** Le operative sono in P-1, P-2 e P-5...P-8; l'ambito del blocco e' in G-1. Nessuna ha avuto un
  secondo giro di scettici. P-3 e P-4 vengono invece da note verificate (R4-N3, R4-N8).
- **Prove.** Sono nella cartella `claude_review_contrastive_targets_runtime_20261120_evidence/` e in
  `~/wearusfm_local/reviews/contrastive_targets_runtime_review_20261120/`.

## Blocco di release

Emesso il 20/11/2026 dopo la firma di Simone (G-1). E' l'unico blocco di questo file e porta i valori della tabella in
G-1.
- **Ambito.** Il blocco non ha un ambito di fase: vale per tutte le fasi del preflight di `fbfde797` (audit, calibrazione,
  CUDA, tre full, quattro reload, lettura), ciascuna con i propri cancelli e ricevute, entro il tetto firmato.
- **Fuori ambito.** I training restano fuori: richiedono una release e una review nuove.
- **Condizioni.** Le misure P-1...P-8 restano condizioni operative scritte qui: il codice non le verifica.

```contrastive-release-review
{
  "schema": "contrastive-target-release-review-v1",
  "status": "APPROVED",
  "reviewed_commit": "fbfde797b48644744c36547533acf0e8a7793535",
  "source_sha256": "40d39f568958004415763f9cc683f3be0b379d170b70c4d37588d1ddcec5cde0",
  "science_plan_sha256": "b03c0bc679f7247d703635d7065c9fc7bda4a292f2c8c106e29004e0527e021a",
  "preflight_plan_sha256": "e4b3959112db86dac806978cf65307d97694da68b64096120487e8f568c6dd92"
}
```

## Prompt per GPT

```text
Review Claude del pacchetto operativo RMS100/fast125 (contenuti fbfde797, pin 0efa4d45): nessun blocco tecnico. La ricevuta si riproduce (465 test) e la risposta a R-1...R-4, M-1, N-1...N-7 corrisponde al codice, salvo l'ID SLURM dell'accounting, solo dichiarato (G-6, R3-M3). File docs/reviews/claude_review_contrastive_targets_runtime_20261120.md sul ramo origin/worktree-claude-training-engine, prove nella cartella _evidence/ accanto (85 agenti con verifica avversaria, bozza riletta da due verificatori).

G-1 chiuso il 20/11. Alla domanda "Firmi la proposta faa4a8d7 e il tetto di 18 GPU-ore?" Simone ha risposto "Si', firmo entrambi" (tetto 18 GPU-ore e 169,6 core-ore, inviluppo 21,6 e 202,4). Il blocco di release e' emesso nella sezione "Blocco di release" di questo file, con i valori verificati: fbfde797b48644744c36547533acf0e8a7793535, sorgenti 40d39f56..., plan b03c0bc6..., preflight e4b39591.... Vale per tutte le fasi del preflight, non per i training. Prima del primo invio: registrare la firma in decisioni.md con la citazione letterale.

Prima delle fasi GPU, senza toccare il codice:
- P-1: TORCH_DIST_INIT_BARRIER=1 nelle celle GPU, contro la corsa all'avvio fra rank non zero. Va nella riga di invio (--export), senza modificare gli sbatch inventariati: contrastive_job.py lancia torchrun con os.execv e l'ambiente passa ai rank;
- P-2: build_command e validazione sul login node prima di ogni sbatch, con controllo a mano del braccio;
- P-3: --dopo solo verso job in coda;
- P-4: override dei checkout nei comandi definitivi;
- P-5: esecuzione CPU gloo di contrastive_run.main per full C0 e reload prima della release della calibrazione;
- P-6: calendario a ritroso dalla manutenzione del 23/11;
- P-7: scrivere prima come si legge un fallimento per tensore continuo/reload con A e B concordi;
- P-8: costi del profilo math misurati alla calibrazione.

Correzioni che cambiano il digest, nella release dei training (farle prima del preflight richiederebbe una review e un blocco nuovi):
- M-a: discendenza e inventario, ID SLURM, braccio;
- M-b: impronte delle tracce;
- M-d: tempo dell'osservatore.
Senza cambiare il digest:
- i test di M-c (25 mutazioni su 39 nelle guardie di release, 30 su 40 nei lettori);
- G-2 su CLAUDE.md ("questa chat" e precedenza da limitare alla linea 3);
- le altre note.
```
