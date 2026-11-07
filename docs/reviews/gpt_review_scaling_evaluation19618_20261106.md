# Review GPT — linea 1, valutazione della continuazione a 19.618

## Esito aggiornato del 07/11/2026 — APPROVATO su 40797b1

**APPROVATO. B-1 chiuso; nessun rilievo bloccante residuo per il percorso operativo pubblicato.**
Commit del pacchetto: `40797b1ccb640b8b986f30539f05a73ac0a8f485`, su
`origin/claude/scaling-300m-exec-20261106`. Sostituisce `f6dc8ec`; il deploy del training
`73524b63be534f81b3db4ade4b7fc7f47129a0f3` non cambia.

Questa verifica **chiude il rilascio unico dell'intero stadio B**: Claude puo' eseguire checkout,
estrazioni, validazione e benchmark, costruzione dei budget, ridge e fit, aggregati, ingressi del
confronto, confronto e raccolta, rispettando i controlli del piano. Nessuna altra review fra
gli invii, ne' nuovo consenso di Simone. I pin ancora inesistenti saranno prodotti dai
costruttori rivisti e pubblicati nei checkout di soli documenti/evidenze previsti; non devono
tornare in review. Invariati stop, budget, perimetro e divieto di retry o estensioni implicite.

### Chiusura del difetto

La verifica usa il percorso effettivo `main(["gate", ...])` -> `main(["lines", ...])` ->
`admitted_lines()`. `stage_lines()` resta correttamente un generatore puro: chiamarlo da solo
non verifica il nuovo percorso di rilascio. La riproduzione storica della prima review resta
conservata come prova del difetto di `f6dc8ec`, non come test della correzione.

- Il cancello registra numero e ID della tranche ammessa; zero posti produce un esito negativo.
- `lines` consuma la tranche del verbale e non genera job ulteriori. Nella riproduzione indipendente
  sui benchmark: coda vuota -> quattro righe, nove job di altre linee -> una, dieci -> nessuna;
  quattro job della linea gia' presenti -> nessuna; due della linea e sette altrui -> una.
- `--max` puo' solo accorciare. Verificati rifiuto senza verbale, verbale fallito/scaduto,
  ledger dello stadio cambiato e identita' del checkout `code` cambiata. Le risorse e le catene
  `afterok` delle righe ammesse restano quelle del generatore precedente.
- I sei aggregati procedono nelle tranche previste, al massimo quattro con il limite di default.
  `--line-limit` non amplia la tabella dei job o il budget: resta una disposizione di coordinamento,
  da registrare come nel piano; il default approvato e' quattro.
- Il controllo dei checkout raccolti richiede `COMPLETED 0:0` e lo stesso ID SLURM del ledger.
  Le eccezioni prima dei controlli producono il verbale `_refused`; il conto predefinito di
  `budget-table` ora mostra il massimo a quindici shard, 345,166667 core-ore.

Il diff tocca esattamente cinque file operativi, di test e documentazione. Verificati invariati
i quindici file derivati, le quindici sorgenti congelate, `src/`, training, piano di training,
endpoint, policy scientifica e freeze. Sessantasei controlli indipendenti confermano perimetro
e risorse; ventitre' funzioni di piani, job, budget e raccolta sono identiche tramite confronto AST.

### Nota non bloccante N-clock — ordinamento dei verbali con orologi diversi

Resta una limitazione riprodotta in `latest_gate()`: i verbali ordinari usano nel nome l'ora
remota, quelli `_refused` l'ora locale. Con il Mac indietro di tre minuti, scarto espressamente
tollerato dal codice, un rifiuto successivo puo' ordinarsi prima di un PASS precedente.
Una chiamata **isolata** a `lines` puo' allora recuperare il vecchio PASS. Non attesto quindi
la garanzia assoluta «qualsiasi ultimo rifiuto invalida sempre ogni precedente verbale».

La nota **non blocca questo rilascio**: il percorso pubblicato e' `gate ... && lines ...`,
e il piano impone lo stop al primo errore. L'eccezione del nuovo cancello interrompe tale
sequenza prima di `lines`; la prova indipendente conferma zero chiamate a `lines` in quel
percorso. Dopo un errore non va eseguito `lines` da solo per riutilizzare un vecchio verbale:
vale gia' la sequenza e la regola di stop del piano, senza un nuovo passaggio di approvazione.
Per un futuro riuso autonomo di `lines`, usare un unico ordine locale dei tentativi, inclusi
i rifiuti, mantenendo separata l'ora remota impiegata per la verifica di freschezza.
Nessuna modifica ai byte di `40797b1` e' richiesta per questa nota prima dell'esecuzione.

**Suite mirata rieseguita: 617 test passati in 232,74 s, nessun fallimento.**
La prova indipendente del percorso CLI comprende 15 scenari, inclusa la distinzione fra
la chiamata isolata e la sequenza prescritta nella nota N-clock.

Le evidenze di questa verifica sono in
`results/infra/scaling_300m_20261104/evaluation19618_b1_closure_20261107/`:
suite mirata e riepilogo, pin del rilascio, audit di perimetro/risorse e riproduzione del
percorso CLI con repository/ledger temporanei. Nessun accesso a Leonardo, nessun deploy,
invio o modifica del worktree Claude da parte di GPT. La suite completa dichiarata da
Claude resta distinta dalla suite mirata rieseguita qui.

## Esito iniziale su f6dc8ec — storico, superato dalla chiusura sopra

**DA RIVEDERE — un solo rilievo bloccante, B-1, nel collegamento fra cancello della coda e tranche di invii.**
Derivazione, lettura scientifica e preventivo sono conformi. Non occorre cambiare il training,
gli algoritmi dei lettori, il controllo a 30M, il budget o i criteri dell'endpoint.

## Consegna e regola di rilascio

- Ramo esaminato: `origin/claude/scaling-300m-exec-20261106`.
- Commit: `f6dc8ec0fb33ae1bcf6d34018194c2e9c71f7cb5` (pacchetto `98cf9a6` e correzione del conto massimo dei fit).
- Deploy del training invariato: `73524b63be534f81b3db4ade4b7fc7f47129a0f3`.
- Perimetro: stadio B completo, dal checkout del codice alla raccolta del confronto.
- Questa e' la review unica del codice/piano dello stadio B richiesta da Simone. La correzione di B-1
  va pubblicata da Claude con le regressioni; la sua verifica chiude il rilascio dello stesso pacchetto.
  Non introduce review fra checkout, estrazioni, benchmark, fit, aggregati e confronto, ne' un nuovo
  consenso di Simone. Dopo il rilascio favorevole restano solo i controlli operativi previsti.
- Lo stadio B di questo commit non e' ancora rilasciato. Il rilievo non modifica l'approvazione
  del training gia' inviato e non richiede interventi sul job `63696864`.

Review eseguita in locale, sul worktree Claude pulito e ai byte del commit dichiarato. Nessun SSH,
deploy, invio, cancellazione, raccolta operativa o modifica al codice Claude. I pin effettivi del
300M a 19.618 non esistono ancora: questa review approva o corregge i costruttori e le regole,
non attesta risultati futuri.

## B-1 — [P2] Il cancello passa anche quando la tranche emessa non entra nella corsia

Riferimenti del commit esaminato: `scripts/scaling_plans19618.py`, righe 1007–1009
(`submittable_now`), 1188–1189 (esito del cancello), 1202 e 1220–1222 (generazione delle righe);
`evaluation_chain_plan.md`, sezioni 6 e 9.

`gate()` calcola i posti disponibili, ma li restituisce soltanto fra i dati informativi.
La disponibilita' non entra nei controlli che determinano l'esito. Il comando `lines` non legge
il verbale del cancello: senza `--max` genera tutte le righe dello stadio, e con `--max` usa il
numero ricevuto senza confrontarlo con la disponibilita' appena verificata.

Riproduzione con il vero `run_gate("pose-benchmark")`, un repository/ledger temporaneo e una
lettura remota simulata; nessun comando di invio eseguito:

| Job gia' nella corsia seriale | Esito del cancello | `submittable_now` | Righe prodotte dal comando pubblicato |
|---:|---|---:|---:|
| 0 | PASS, exit 0 | 3 | 3 |
| 9 | PASS, exit 0 | 1 | 3 |
| 10 | PASS, exit 0 | 0 | 3 |

Quindi il percorso documentato `gate pose-benchmark && lines pose-benchmark` puo' produrre
un primo invio accettato e il secondo rifiutato dalla QoS, oppure il rifiuto gia' del primo.
La regola di stop interromperebbe la catena per un problema che il cancello dichiara di aver
controllato. Non e' un nuovo esperimento o una modifica malevola dei documenti: e' l'esecuzione
dei comandi pubblicati in uno stato normale della corsia condivisa.

La stessa disconnessione compare negli aggregati: il comando di sezione 9 emette sei invii,
anche con la corsia vuota, mentre la sezione 6 dichiara un massimo di quattro job della linea 1
contemporaneamente sottomessi. Sono due manifestazioni dello stesso rilievo, non due blocchi distinti.

**Correzione richiesta:** legare l'ammissione e la generazione alla stessa tranche concreta
(numero e ID). Le righe effettivamente prodotte devono rispettare sia i posti liberi nella QoS
sia il limite di linea dichiarato, considerando i job della linea 1 gia' presenti. Con zero posti
non deve essere emessa alcuna riga eseguibile. Un `--max` non puo' aggirare questo controllo.
Si puo' rifiutare una tranche troppo grande oppure limitarla esplicitamente e registrare il
sottoinsieme ammesso; non serve introdurre un runner o una richiesta di consenso.

Regressioni richieste sul percorso cancello -> generazione, non solo sul conteggio informativo:

- coda piena: zero righe di invio;
- un posto disponibile: al massimo una riga, oppure rifiuto della richiesta di tre;
- sei aggregati: tranche compatibile con il massimo quattro e con gli eventuali job gia' presenti;
- richiesta `--max` superiore alla disponibilita' (e valori non positivi): nessuna emissione fuori limite;
- coda libera e tranche valida: stessi comandi, risorse e dipendenze del piano, senza variazioni scientifiche.

Il controllo riguarda lo stato letto in quel momento; non richiede di rendere atomica la QoS di
Slurm rispetto agli invii della linea 2. Restano il coordinamento fra operatori e lo stop al
primo rifiuto. Occorre eliminare l'incoerenza gia' presente fra il gate e i suoi stessi comandi.

## Giudizi richiesti a–f

| Punto | Giudizio |
|---|---|
| **a. Derivazione** | Conforme. Ricostruiti tutti i 15 file dai blob di `97125bd`: 7 di codice e 8 di test, conteggi esatti delle sostituzioni e hash finali identici. Sono dichiarati anche il contratto della continuazione, le tre sole estrazioni 300M, i limiti del freeze e la lettura descrittiva. Nessuna modifica agli algoritmi scientifici o a `src/`. |
| **b. Cancelli meccanici** | Identita', immutabilita', provenienze, fit, aggregati e runtime risultano coerenti. Il rilascio della catena resta bloccato solo da B-1. S-1 e S-3 sono coperti dal confronto delle firme numeriche e degli shard realmente letti con quelli accettati. |
| **c. Cinque checkout anziche' tre** | Accettabile. A0 a 2 CPU e quattro checkout a 1 CPU, tutti da 300 s: 0,50 core-ore richieste, 0,80 con 180 s di margine per job, identiche al tetto. Cambia la ripartizione operativa, non l'inviluppo o la domanda scientifica. |
| **d. Costi** | Conformi al freeze v2: 345,166667 core-ore solo CPU nel caso peggiore, su 350; riserva 4,833333. Estrazioni 7 GPU-ore + 0,15, con 56 core-ore associate (+1,2 di margine, rendicontate separatamente). Fit entro 15 shard, 3 h ciascuno e 240 core-ore richieste. |
| **e. Endpoint** | Conforme e invariato. Il documento e' identico al deploy del training; primari EPN/E_dyn, margine P1/P2 di 1 pp, riduzione pratica della posa del 2%, bootstrap per soggetto 2.000/20261102 e viste naturali primarie restano gli stessi. Nessuna raccomandazione di dose, selezione fra checkpoint o ulteriore training autorizzato. |
| **f. S-2 e S-4** | Correttamente collocati dopo il confronto, come controlli dell'esito tecnico. S-2 deve verificare repliche/seme nei blocchi effettivi di classificazione e posa; S-4 deve confrontare tutti i punti cella/dataset con gli aggregati entro 1e-12. Vanno verbalizzati prima di accettare scientificamente il risultato; non diventano review prima degli invii. |

La continuazione resta condizionata a un solo seme e agli utenti di sviluppo. Un esito
indeterminato non dimostra equivalenza; il confronto con la lettura a 5.000 non e' una
replica indipendente e non autorizza la scelta a posteriori del checkpoint migliore.

## Precisazioni non bloccanti

1. Il gate dei lettori verifica HEAD e albero dei checkout, ma non legge direttamente il loro
   `sacct`. Nel percorso pubblicato questo e' coperto da `collect checkout-*`, che richiede
   `COMPLETED 0:0`: tale raccolta non va omessa. Un eventuale rinforzo futuro puo' far consumare
   al gate quell'accettazione; non e' una seconda correzione obbligatoria per questo rilascio.
2. Gli errori sollevati prima della costruzione dell'elenco dei controlli fermano correttamente
   il comando, ma non producono il verbale previsto per «ogni cancello». In questi casi vanno
   conservati l'errore e il registro; la frase del piano e' piu' ampia della copertura effettiva.
3. `budget-table` senza argomenti mostra 343,166667, lo scenario a dieci shard; il massimo a
   quindici shard e' 345,166667. La distinzione e' gia' dichiarata dal piano corretto e non
   richiede di cambiare i byte del rilascio per questa nota.

## Verifiche e riproduzione

**Suite mirata: 615 test passati in 186,42 s, nessun fallimento.** La riproduzione indipendente
di B-1 esercita casi non coperti dalla suite: il numero di test verdi non elimina il difetto.

Evidenze in `results/infra/scaling_300m_20261104/evaluation19618_review_20261106/`:

- `release_pins.json`: 36 file della consegna confrontati con i blob Git del commit; sorgenti
  scientifiche, wrapper di training e submitter immutati rispetto a `73524b6`;
- `derivation_audit.json` e `audit_derivation.py`: 15 derivazioni esatte, confronto AST delle
  funzioni e 24.576 casi di lettura confrontati con formule indipendenti;
- `resources_audit.json` e `audit_resources.py`: 47 controlli indipendenti; somme del freeze,
  launcher e `sacct` grezzo del checkout A0 (69 s, 2 CPU, 0,038333 core-ore);
- `queue_gate_reproduction.json` e `reproduce_queue_gate.py`: riproduzione di B-1 su repository
  e ledger temporanei; lettura remota simulata, nessun accesso a Leonardo;
- `package_tests.txt` e `test_summary.json`: esito della suite mirata sul commit consegnato.

Gli script di audit puntano al worktree Claude indicato nella consegna e vanno eseguiti a
quel commit; usano soltanto file locali. Comando della suite:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 ~/.venvs/wearusfm-tok/bin/python -B -c \
"import sys, glob; sys.path.insert(0, 'src'); import pytest; sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', *sorted(glob.glob('tests/cpu/test_scaling_*19618*.py'))]))"
```

La suite completa `tests/cpu` dichiarata da Claude non e' stata rieseguita in questa review:
la verifica richiesta e' concentrata sul pacchetto consegnato e sui controlli indipendenti
descritti sopra. Le precisazioni non bloccanti non richiedono modifiche al codice congelato;
Claude deve correggere soltanto B-1 e pubblicare il nuovo commit della consegna.
