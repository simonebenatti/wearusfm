# Review di Claude — T100-LQ: benchmark d'inferenza ed estrazione di posa a 5.000 passi (GPT cc81430..63eba9a) — 06/11/2026

Richiesta di Simone, 06/11 mattina. Alla mia nota «I commit di GPT su benchmark d'inferenza ed estrazione di posa a 5.000:
non li ho rivisti. Se li vuoi controllati, dimmelo» ha risposto: «controllali, tanto qua c'e tempo da quanto capisco».
Solo review: nessun job, nessuna modifica a codice o piani, su Leonardo soltanto letture.

Ramo `codex/tokenizer-logcontrast-20261104`, punta 63eba9a (05/11, 20:48). Ricontrollata con `git ls-remote` prima del
verdetto: invariata. Linea 3: GPT ne e' autore e operatore, Claude revisore.

**Verdetto: APPROVATO. Nessun problema, ne' bloccante ne' da correggere.** Rapporti di costo, tetti, piano, invii, hash e
consuntivi dichiarati da GPT coincidono con il mio ricalcolo e con lo scheduler. Tre note informative, da E-1 a E-3.
Nessun punteggio scientifico esiste ancora: il benchmark misura solo tempi, le cache di posa non sono state lette.

| Commit | Cosa fa | Esito |
|---|---|---|
| cc81430, parte benchmark | Pin del checkpoint a 5.000 nei due piani, stato `APPROVED_FOR_BENCHMARK` | **CONFORME** al piano approvato il 04/11 |
| 60808d6 | Archivia ricevuta, log e ricevute d'invio del benchmark | **CONFORME**: SHA uguali agli originali sul cluster |
| 99efdef | Test del launcher storico B19618 | Solo test: 15 passati (eseguiti il 05/11) |
| 9720fc8 | Lettura del benchmark, quattro tetti, piano nuovo, rilascio della sola posa a 5.000 | **CONFERMATA** dal mio ricalcolo |
| 5b03e0c, parte estrazione | Checkout 63615611 e invio dell'estrazione 63615952 | **CONFORME** a piano, ledger e `sacct` |
| 63eba9a | Estrazione completata, hash verificati, letture scientifiche rimandate | **CONFERMATO** sul cluster |

## Cosa ho eseguito

- **Codice.** `scripts/logsquare_infer_benchmark.py` e `scripts/logsquare_evaluation.py` sono identici, per SHA256, a quelli
  della mia review del 04/11 (c8f5c12). Gli alberi git di `src/` e `scripts/` sono gli stessi a cc81430, 9720fc8, 5b03e0c e
  63eba9a: dopo cc81430 cambiano solo due file di test.
- **Benchmark ricalcolato da zero** con `docs/reviews/claude_review_logsquare_inference_extraction_20261106_costs.py`: legge
  le 128 chiamate grezze della ricevuta, non importa il progetto, riscrive la regola D1 da `docs/decisioni.md`.
- **Piani confrontati foglia per foglia** con `..._20261106_plans.py`, che ricalcola anche l'hash del contesto del benchmark.
- **Pin ricalcolati** con `docs/reviews/claude_review_logsquare_relaunch_20261105_pins.py` sui commit 9720fc8 e 63eba9a:
  piano nuovo 114 pin, piano d'inferenza 115, piano di estrazione del 04/11 113. Tutti uguali ai file.
- **Leonardo, sola lettura, due comandi alle 11:07 e alle 11:12 del 06/11:** `sacct` dei cinque job, `squeue`, `ls`, `stat`,
  `sha256sum` di ricevute, log, cache e piani nei checkout, `cat` delle provenienze, `git rev-parse` dei due checkout.
  Comandi e output in `..._20261106_leonardo.txt`.
- **Test non rieseguiti alla punta:** l'albero `tests/` di 63eba9a e' identico a quello di 5b03e0c, dove il 05/11 ho ottenuto
  365 passati (363 nei 12 file LQ) e `bash -n` dei sette launcher. Rieseguirli non aggiungerebbe nulla.

## Il benchmark 63614514

Otto record, 16 chiamate ciascuno: ordine alternato L e LQ, due giri di riscaldamento, sei misurati per braccio. In tutte
le 48 coppie misurate LQ e' piu' lento di L, fra lo 0,08% e l'1,74%. Rapporti per gruppo, uguali alla ricevuta fino
all'ultima cifra:

| Gruppo | Precisione | Record | Media L (s) | Media LQ (s) | LQ / L |
|---|---|---:|---:|---:|---:|
| pose | FP32 | 2 | 0,314767 | 0,315657 | 1,002825 |
| ninapro_db2 | BF16 con decoder | 1 | 0,368859 | 0,373554 | 1,012729 |
| ninapro_db3 | BF16 con decoder | 1 | 0,372118 | 0,376883 | 1,012804 |
| ninapro_db6 | BF16 con decoder | 1 | 0,374720 | 0,379340 | 1,012331 |
| epn612 | BF16 con decoder | 2 | 0,810910 | 0,816016 | 1,006297 |
| uci_emg | BF16 con decoder | 1 | 0,356331 | 0,361108 | 1,013405 |

Tetto di ogni cella: limite storico per il rapporto massimo dei suoi gruppi, arrotondato ai 300 s superiori e mai sotto
lo storico. Stima: durata storica per il rapporto pesato con i conteggi storici di finestre. Risultato identico a GPT:

| Cella | Storico: limite e durata (s) | Rapporto massimo | Rapporto pesato | Stima (s) | Tetto (s) | GPU-ore di tetto |
|---|---:|---:|---:|---:|---:|---:|
| pose_5000 | 2.700 e 1.796,21 | 1,002825 | 1,002825 | 1.801,28 | 3.000 | 0,8333 |
| pose_19618 | 2.700 e 1.714,01 | 1,002825 | 1,002825 | 1.718,85 | 3.000 | 0,8333 |
| p1_19618 | 1.800 e 880,05 | 1,012804 | 1,012574 | 891,11 | 2.100 | 0,5833 |
| p2_19618 | 2.700 e 1.667,54 | 1,013405 | 1,006351 | 1.678,13 | 3.000 | 0,8333 |

Somma dei tetti 11.100 s: 3,0833 GPU-ore e 24,6667 core-ore, checkout esclusi. Ogni cella 1 GPU, 8 CPU, 100G.

- Le quattro ricevute storiche di T100-L sul cluster hanno gli SHA256 pinnati e le durate usate nel calcolo. GPT segnala che
  il vecchio riepilogo locale non riporta la durata storica di `pose_5000`: l'ho letta dall'originale, 1.796,2077 s.
- Selezione e aggregazione del piano eseguito sono quelle approvate il 04/11: fra c8f5c12 e cc81430 il piano d'inferenza ha
  176 foglie identiche su 179. Cambiano lo stato e il pin del piano di estrazione, che a sua volta cambia solo per stato,
  piano di training del 05/11 e pin del checkpoint a 5.000 (167 foglie identiche su 172, dieci campi di pin aggiunti).
- Ricevuta: codice cc81430, piano `71fe65bf…`, checkpoint candidato `6233b070…`, controllo `0ec89584…`, 110 sorgenti uguali
  al piano, otto ingressi distinti alle frequenze attese. SHA256 `7e6e665e…` e 198.900 byte, uguali sul cluster.
- **R1-b, norme del ramo log-quadratico**, ricalcolate dalle 64 colonne: tutte positive in student e teacher. Nel teacher le
  medie sono 0,1383 Fourier, 0,0655 spline e 0,1069 MLP; gli MLP pesano il 24,42% della somma dei quadrati. E' una
  descrizione: non dice se il ramo aiuta.

## L'estrazione di posa a 5.000 passi, job 63615952

- **Piano nuovo** `results/infra/logsquare_evaluation_20261105/extraction_plan_5000.json`, SHA256 `56a9b51e…`. Rispetto al
  piano di estrazione su cui ha girato il benchmark: 176 foglie identiche su 181. Cambiano stato (`APPROVED_FOR_EXTRACTION`)
  e commit del benchmark; si riempiono tre campi prima vuoti: pin della ricevuta, pin del piano d'inferenza, allocazioni.
  Le allocazioni sono uguali alla proposta della ricevuta. L'hash del contesto del benchmark, ricalcolato sul piano nuovo,
  e' `9e1a7bb9…`, uguale a quello scritto nella ricevuta: stessi ingressi, riferimenti, sorgenti e checkpoint.
- **I due piani eseguiti dal benchmark non sono stati toccati:** stessi blob a cc81430, 9720fc8, 5b03e0c e 63eba9a.
- **Il 19.618 non e' eseguibile:** nel piano il suo checkpoint e' `null` e il codice rifiuta un pin mancante.
- **Invio.** Comando del submitter: 1 GPU, 8 CPU, 100G, `--time=00:50:00`, cella `5000 extract-pose none`, SHA del piano
  come sesto argomento. `scontrol` e `sacct` riportano le stesse risorse e lo stesso limite. Il codice esige che il limite
  allocato sia uguale al tetto misurato, che ricalcola da ricevuta e ricevute storiche prima di estrarre.
- **Checkout** `logsquare-pose-5000-20261105`: `HEAD` 9720fc8, cartella e i tre file guardati in sola lettura; piano, piano
  d'inferenza, estrattore e launcher con gli SHA256 attesi. Il checkout del benchmark e' a cc81430.
- **Esito.** Ricevuta sul cluster: `complete: true`, codice 9720fc8, piano `56a9b51e…`, checkpoint `6233b070…` invariato.
  Sette file riletti con `sha256sum`: ricevuta `3dca283a…` (3.793.503 byte), cache train `881b58e7…` (257.774.193 byte),
  cache val `ae95cd20…` (2.485.995.330 byte), due provenienze, due log. Tutti uguali alla verifica archiviata da GPT.
- **Stesse finestre e stessi target di T100-L.** Le provenienze LQ e quelle storiche di T100-L a 5.000 hanno 44 foglie
  identiche su 54: fra queste i tre hash dei target (metadati, array, maschera centrale), 1.907 finestre train da 200
  registrazioni e 40 utenti, 18.548 finestre val da 1.950 registrazioni e 15 utenti, seme, statistiche di posa. Le dieci
  diverse sono braccio, protocollo, codice, piano, checkpoint e identita' della cache. In piu', solo in LQ, i due campi
  `tokenizer_logsquare` ed epsilon 0,001. Il confronto appaiato futuro parte quindi dagli stessi dati.

## Job, costi e autorizzazione

| Job | Cosa | Stato | Durata | Limite | Risorse | Costo |
|---|---|---|---:|---:|---|---|
| 63614193 | Checkout del benchmark | COMPLETED 0:0 | 28 s | 10 min | 2 CPU, 4G | 0,015556 core-ore |
| 63614514 | Benchmark d'inferenza | COMPLETED 0:0 | 110 s | 30 min | 1 GPU, 8 CPU, 100G | 0,030556 GPU-ore, 0,244444 core-ore |
| 63615611 | Checkout dell'estrazione | COMPLETED 0:0 | 30 s | 10 min | 2 CPU, 4G | 0,016667 core-ore |
| 63615952 | Posa a 5.000 | COMPLETED 0:0 | 1.866 s | 50 min | 1 GPU, 8 CPU, 100G | 0,518333 GPU-ore, 4,146667 core-ore |
| 63593128 | Continuazione a 19.618 | PENDING, Priority | non partita | 4 h | invariate | tetto 16 GPU-ore |

Le due fasi hanno consumato 0,548889 GPU-ore e 4,423333 core-ore, contro tetti di 1,3333 GPU-ore e 11,3333 core-ore. Le
quattro righe del ledger comune coincidono con le ricevute; stesso `submit_job.py` (`814da4bc…`). Alle 11:07 del 06/11 la
continuazione 63593128 e' in coda da 19 ore e 48 minuti.

- Benchmark e regola dei tetti li avevo approvati il 04/11 (review 6c0ce65), con questa sequenza: pin dell'endpoint,
  controllo A1, benchmark, pubblicazione dei tetti fra le due fasi. GPT l'ha seguita. Vale l'autorizzazione di Simone del
  04/11, riportata in `resource_proposal.json`: «si ma dovra fare altro immagino, lo autorizzo gia a desso».
- Ogni estrazione, e anche la somma delle quattro (3,0833 GPU-ore), sta sotto la soglia del 28/10: «se devi sottomettere
  cose che durano meno di 5h-GPU fallo pure in autonomia».
- Lettura mia: «ripassa da te» (decisione di Simone del 06/11 su N-2) non riguarda questi invii. Non sono rilanci dopo un
  arresto tecnico: sono le fasi successive gia' approvate.

## Note

- **E-1, conservazione (da non dimenticare).** Cache, ricevuta completa dell'estrazione e checkpoint stanno solo su `$SCRATCH`.
  Con le letture rimandate alla fine devono durare fino ad allora. Li copre il rinfresco del 12/11:
  `scripts/slurm/refresh_scratch.sbatch` tocca `wearusfm_runs` nei bersagli di default. Senza rinfresco, contando 40 giorni
  dalla data di modifica, i file del 05/11 scadono verso il 15/12 e i riferimenti storici B19618 del 29/10 verso l'8/12.
  Sul Mac della ricevuta completa (3,8 MB) c'e' solo il riepilogo con gli hash, come per T100-L.
- **E-2, tetti (informativa).** L'arrotondamento ai 300 s porta ogni tetto allo scalino successivo anche per un sovraccosto
  dello 0,3%: 1.200 s in piu' in tutto, 3,0833 GPU-ore contro le 2,75 dei limiti storici. E' la regola congelata, ed e' un
  tetto, non un costo. La posa a 5.000 e' durata 1.848 s nel programma (1.866 s per lo scheduler): 2,6% sopra la stima e
  2,9% sopra T100-L, mentre il solo forward costa lo 0,28% in piu'. Ipotesi mia: il resto viene da lettura dei dati e I/O,
  che il benchmark non misura. GPT dichiara la stima come proxy. Nessuna azione: il tetto e' usato al 62%.
- **E-3, test (minore).** Il piano nuovo non ha un test locale: lo protegge il controllo `allocation_for`, che gira dentro
  il job ed e' passato. Per i piani del 19.618 conviene ripetere quel controllo prima dell'invio, come GPT ha fatto qui.

## Cosa non ho verificato

- Gli hash dei due riferimenti storici di posa (2,7 GB) e dei cinque di classificazione: li attestano le provenienze e il
  controllo del job. I conteggi storici di p1 e p2 non li ho ricontati; per la posa 1.907 + 18.548 = 20.455 torna.
- La frase di Simone riportata da GPT, «possiamo anche fare tutte le letture alla fine»: non l'ho vista.
- Il contenuto numerico delle cache. Nessun lettore CPU e' stato inviato: dopo l'estrazione il ledger non ha altre righe
  LOGSQ.
- Come GPT abbia eseguito `allocation_for` sul cluster alle 19:17, prima che il checkout esistesse
  (`prelaunch_gate.json`). Ho verificato che lo stesso controllo gira dentro il job.
- Se la cancellazione su `$SCRATCH` conti dalla data di modifica o di accesso.

## Prompt per GPT

```
Review di Claude sul ramo worktree-claude-training-engine:
docs/reviews/claude_review_logsquare_inference_extraction_20261106.md (commit nel registro condiviso).
Oggetto: i tuoi commit cc81430..63eba9a su benchmark d'inferenza ed estrazione di posa a 5.000 di T100-LQ.

Verdetto: APPROVATO, nessun problema. Ricalcolati da zero i sei rapporti LQ/L e i quattro tetti dalle 128 chiamate
della ricevuta: identici ai tuoi. Piano nuovo, pin, contesto del benchmark, invii, ledger, sacct e hash di ricevute e
cache sul cluster coincidono. Le provenienze LQ hanno gli stessi hash dei target di T100-L a 5.000.

Tre note, nessuna richiede un'azione ora:
- E-1: cache, ricevuta completa e checkpoint stanno solo su SCRATCH. Con le letture rimandate devono durare fino alla
  fine: li copre il rinfresco del 12/11 (refresh_scratch.sbatch tocca wearusfm_runs). Tienilo presente nel piano finale.
- E-2: la posa a 5.000 e' durata il 2,6% piu' della stima proxy (1.848 s contro 1.801). Tetto usato al 62%.
- E-3: il piano nuovo non ha un test locale; per i piani del 19.618 ripeti il controllo allocation_for prima dell'invio.

Decisione di Simone del 06/11 sulla nota N-2 della review 49f9438, parole sue: «ripassa da te».
Lettura di Claude, da confermare con Simone se serve: su questa linea, dopo un arresto tecnico, pubblichi piano,
release e pin del rilancio e Claude li rivede prima di checkout e invio. Non riguarda i job gia' inviati ne' le fasi
gia' approvate (estrazioni a 19.618 e lettori), che seguono le regole di sempre. Registro di Claude: 7f976ec.

Non serve una risposta. Nessun job nuovo e nessuna modifica richiesta.
```
