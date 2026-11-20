# Rilettura GPT — seconda consegna TCN, utenti del preflight — 20/11/2026

**Esito del diff: APPROVATO. Nessun rilievo bloccante trovato.** La correzione tecnica
è adeguata al difetto del primo preflight. **Il rilascio resta in attesa della
conferma finale di Claude tramite Simone**, come richiesto nell'addendum alla consegna:
due suoi revisori stanno ancora controllando la punta. Questa review non sostituisce
quella conferma e non esegue `release-plan`, deploy o invii.

## Perimetro e identità

- Base: `31877ad85c619ec69214cd5e9fb53df50e452148`.
- Punta riletta: `0e548bf8218f15ba07fc731efcc2ead512e65034`, ramo
  `claude/pose-temponet-preflight-users-fix-20261120`.
- Codice per `--reviewed-code-sha`: `5d6fb2dc855bd3c12788bb6d3c3a66043c34f1ba`.
- Identità dei 246 sorgenti:
  `37985d28c75bf82a6a784106d414825b727b9ba78575b675dd2dac1d4ccfc8de`.
- Piano non rilasciato: `results/infra/pose_temponet_execution_20261120/plan.json`,
  SHA256 `1e8e2b46b046072c706738995e745f608f009d33406d970729a3eb22a73f563e`.

Ho verificato queste identità sul checkout locale, senza operare su Leonardo.
I 246 file coincidono byte per byte con il commit di codice indicato. Cambiano
soltanto `scripts/pose_temponet/provenance.py` e `scripts/pose_temponet/execution.py`
rispetto alla base; gli altri cambiamenti sono test, piani, documenti e prove.
Il piano ricostruito è identico a quello consegnato e resta `REVIEW_REQUIRED`,
con `submission_released: false`.

## Correzione del difetto

Il produttore conserva nella cache l'ID, mentre la ricevuta censisce la cartella
`u` + ID sanitizzato. Il primo preflight confrontava questi due formati diversi.
La correzione avviene dove si costruisce la lista del produttore, prima di tutti
i consumatori. Mantiene distinti `users` e `storage_directories` e li verifica
entrambi durante la lettura; non elimina il controllo della popolazione.

`cache_user` toglie esattamente il prefisso `u` e accetta soltanto nomi per cui
la trasformazione del produttore è invertibile. Un `_` può provenire da caratteri
diversi e viene rifiutato. Il round-trip usa la stessa espressione regolare del
produttore. Questo è corretto per tutti i 55 ID reali, di dieci caratteri esadecimali.
L'import aggiunto non introduce un nuovo caricamento del produttore: il modulo
era già importato dalla catena di `data`.

La verifica indipendente sulle due ricevute e sulle osservazioni di metadati
archiviate conferma **40 utenti / 1.907 finestre train** e **15 utenti / 18.548
finestre val**, senza utenti in comune. I digest degli elenchi coincidono anche
con quelli della diagnosi dell'operatore. Metadati del produttore e dettagli
storici nel piano sono invariati.

Ho inoltre esercitato benchmark, workflow e controllo della partizione con
gli ID reali e 80 finestre di geometria reale, ma array sintetici. Tutti producono
gli stessi due fold da 20 utenti. Offrendo invece i vecchi nomi `u+ID`, tutti e
tre rifiutano l'input. Una verifica enumerativa di 16.104 stringhe ha recuperato
esattamente tutti i 2.800 casi ammessi; tutte le 1.880 cartelle con collisioni
sono state rifiutate. Non è una misura della qualità delle teste.

## Seconda identità operativa

Proposta, ricetta, otto input, soglie, tetti e radice degli output sono invariati.
Il grafo dei 24 job è identico dopo la sola sostituzione del suffisso; nessun ID
nuovo coincide con quelli della prima consegna o dei due inventari CPU.
Campagna e checkout nuovi sono `pose_temponet_20261120`.

Il vecchio piano è rifiutato dai sorgenti nuovi. La review storica `85f0d62` non
identifica il nuovo codice e non soddisfa il controllo di `release-plan`.
Le ricevute vengono cercate per i nuovi ID e autenticate rispetto al nuovo
piano: la ricevuta fallita rimane una traccia, non un predecessore valido.
Il runner rifiuta comunque una destinazione di stadio o una ricevuta già esistente.

Le istruzioni del 20/11 sono coerenti: ancestry conservata, evidenza operatore
nuova, pin della review e del codice effettivo, riferimento di deploy nuovo e sei
controlli locali prima dell'invio. Per collegare la rilettura indipendente è
preferibile usare **questo file** come artefatto della review. L'etichetta fissa
`claude_review_converged` dello schema non cambia l'attribuzione: autore Claude,
revisore GPT, operatore unico **2-TCN head launcher**.

La freschezza dell'evidenza operatore e l'appartenenza degli ID alla seconda
consegna restano controlli procedurali esplicitati, non nuove garanzie automatiche.
La situazione della radice degli output e delle risorse dovrà essere ricontrollata
dall'operatore; non l'ho riletta sul cluster. Il consuntivo finale comprende anche
i tre job già eseguiti (due inventari e il preflight fallito), negli stessi tetti.

## Verifiche e limiti

Riesecuzione conclusa: **271 test TCN passati in 164,47 s**; **125 test storici
passati e un fallimento preesistente**. È la stessa asserzione già documentata
in `baseline_failure.json`: il test si aspetta che il vecchio piano della
valutazione lunga non sia ancora approvato, mentre il piano registra già la
convergenza. Nessuna nuova regressione; nessuna modifica per nascondere il caso.
La suite completa del repository non è stata rieseguita: il diff non cambia
default o sorgenti esterni al pacchetto TCN.

Prove riproducibili in `results/infra/pose_temponet_execution_20261120/gpt_review/`:

- `audit.py` / `audit.json`: identità dei sorgenti, rigenerazione del piano,
  uguaglianza del grafo normalizzato, pin della consegna, ricevute e popolazioni;
- `consumers.py` / `consumers.json`: consumatori reali con ID reali e array
  sintetici, rifiuto del vecchio formato e controllo dell'inversione;
- `validation/local_validation.json` e i due log: riesecuzione dell'intero
  pacchetto TCN e dei cinque file storici con comandi, ambiente e hash.

Ho ricalcolato anche i **93 controlli su 93** dalle intestazioni e dai metadati
archiviati da Claude. Non è una nuova lettura dei file sul cluster. Il test del
corpo vero di `provenance.preflight` usa il formato reale, ricevute reali con
artefatti riassociati, array sintetici e costanti R-1 sostituite esplicitamente:
prova il percorso e il rifiuto del difetto precedente, non la parità numerica
storica sui dati completi.

Restano da verificare nel nuovo job i digest completi degli array, R-1 a `1e-12`,
tempo e memoria della lettura di val. Il confronto scientifico non ha ancora
prodotto risultati. Nessun cambiamento a FM, teste, obiettivi o criteri.

## Seguito

La decisione L3-9 per un nuovo preflight è già documentata e trasmessa da Simone;
non occorre richiederla di nuovo. Prima del rilascio si attende soltanto la
conferma finale di Claude, **tramite Simone**, sui suoi due revisori ancora attivi.
Se comporta cambiamenti ai sorgenti, si ricontrolla il diff incrementale e si
aggiornano le identità prima di usare questa review. Se i sorgenti restano quelli
indicati, l'operatore può seguire il prontuario del 20/11 nei limiti già approvati.
Questa consegna contiene soltanto review e prove locali: nessun nuovo piano
rilasciato, modifica del runtime, deploy, invio o monitor.
