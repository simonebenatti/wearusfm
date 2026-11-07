# Review critica della proposta contrastiva — 07/11/2026

**Verdetto: APPROVATA come proposta da sottoporre a Claude, non come protocollo eseguibile o autorizzazione di training. Nessun rilievo P0/P1.** Le scelte ancora da validare sono esplicite; implementazione, collaudo, copertura e preventivo restano necessari prima di una release.

Perimetro: `docs/contrastive_proposal_20261107.md`; confronto con la configurazione storica T100-G e lettura mirata di `fm.py`, `jepa.py`, loader e contratto DDP. Nessun fit, esperimento, accesso al cluster o modifica del runtime.

## Coerenza scientifica

- **Base e controllo:** scegliere G dopo LQ è una scelta adattiva correttamente dichiarata. Due nuovi bracci da identica inizializzazione, con stesso proiettore e gestione della FIFO, isolano il peso contrastivo meglio del solo confronto con un controllo storico. Il confronto resta a pari esposizione; non promette pari costo con il vecchio trainer.
- **Positivo e livello:** student mascherato e teacher completo leggono la medesima finestra e montaggio. La rappresentazione contrastiva è distinta da pool768 e dalle sonde; normalizzare soltanto il proiettore non impone per costruzione una normalizzazione della posa. È esplicito che mascheratura e pooling globale non garantiscono conservazione del movimento.
- **Negativi:** memoria per dataset e esclusione delle copie fisiche o vicine affrontano i rischi più immediati. Mantenere anche negativi dello stesso soggetto evita di rendere l'identità sufficiente per definizione. Restano falsi negativi e scorciatoie interne al dataset: il testo li riconosce e non promette di risolverli con questi filtri.
- **Peso:** la calibrazione è train-only e seguita da reset, senza griglia o adattamento sulla validazione. Il limite della testa posa inizialmente nulla è dichiarato: il rapporto iniziale riguarda soprattutto le ancore e non certifica il successivo bilanciamento del training. λ, età/FIFO, soglie di copertura e temperatura sono ipotesi progettuali da far valutare a Claude, non valori dimostrati ottimali.
- **Distribuito e ripresa:** chiavi detached, query locali e denominatore globale 32 sono coerenti con una NCE momentum asimmetrica. La coda deve essere identica fra rank e aggiornata una volta sola, senza effetti da checkpointing o valutazione. I test richiesti su rank senza query valide e ripresa sono pertinenti; non si presume il supporto all'accumulo.
- **Lettura:** P1/P2 sono scelti prima dei nuovi esiti, con intervalli al 97,5% per i due endpoint principali e una guardia separata sulla posa. I margini pratici sono proposti come nuovi, senza reinterpretare i criteri LQ. Gli intervalli restano condizionati al singolo pretraining e la correzione non cancella la selezione sullo sviluppo precedente. Preservare la posa supervisionata non prova trasferimento a una nuova regressione.

## Due precisazioni P2 da recepire nel testo operativo

1. **Copertura dei negativi e validità della vista sono controlli distinti.** Un audit limitato a identità, tempi e regole FIFO può verificare il numero dei negativi ammissibili. Per certificare una query NCE valida occorrono anche supporto QC e chiavi visibili della maschera effettiva. Se l'audit ricostruisce questi soli metadati, va scritto; altrimenti il gate completo deve essere verificato nel collaudo e poi nei log del training. Non assumere che il solo numero di negativi certifichi la copertura effettiva.
2. **`trial=None` è un valore lecito per una registrazione continua.** Nel loader `Window.trial` è `int | None`: due finestre con questo valore nella stessa sessione condividono lo stesso asse temporale, se così definito dal dataset. Non confondere questo caso con identità mancante o trial diversi senza coordinate confrontabili. La chiave deve distinguere il valore nullo previsto dallo schema dall'assenza del campo richiesto.

Sono precisazioni della specifica, non richieste di cambiare la domanda, aggiungere un braccio o avviare diagnostiche adesso. Per le soglie 90%/80% e almeno 32 query per dataset, i dataset non valutabili devono rimanere segnalati come previsto; una rinuncia silenziosa al loro contributo cambierebbe la domanda.

**Rilettura dei chiarimenti: entrambi i P2 CHIUSI.** Il documento ora distingue il gate dei negativi dal gate completo QC/chiavi visibili e definisce `trial=None` come asse continuo lecito; gli stessi chiarimenti sono coerenti nel riepilogo `results/infra/contrastive_proposal_20261107/proposal.json`. Nessun cambio della domanda o rilievo residuo da questa rilettura.

## Risorse e verifiche

Ricontrollato SHA256 della configurazione G: `f9d0318462fd30a5592087ef72c63d3036a88ff1d07fc0dad8624e36253c2cfa`. Batch 32, accumulo 1, seme 0, LR 1e−4, warmup 1.962, WSD 49.046, EMA 0,996 e pesi ancore/posa/JEPA coincidono con il riferimento citato.

Il codice corrente calcola entrambe le encode anche a JEPA=0; la proposta può riusarle. Il costo resta da misurare perché proiezione, comunicazione, gestione della coda e diagnostica non sono gratuiti. Il calcolo storico `(3916 + 11334) × 4 / 3600 = 16,944444 GPU-ore` è corretto; circa 33,89 per due bracci è un riferimento prima del sovraccosto, non un preventivo. Il tetto doppio storico e le altre tranche sono correttamente distinti da risorse approvate.

**Conclusione della review:** la proposta è abbastanza concreta per ottenere un parere utile da Claude su scientificità e fattibilità senza lanciare nulla. Le due precisazioni sopra evitano ambiguità nell'implementazione successiva. Non è necessario completare ora trainer o launcher per questa fase di review della proposta.
