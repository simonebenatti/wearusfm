# Review interna del pacchetto consolidamento

Revisore: subagent `consolidation_critic`, due passaggi in sola lettura sul worktree
consolidamento. Non è la review Claude richiesta da Simone e non autorizza lanci.

Prima review: favorevole alla preparazione, nessun difetto bloccante rilevato nella
continuità WSD/optimizer/RNG/stream, nella selezione train/validation/held, nei tre
regimi di inizializzazione o nel forward pre/post. Dodici test indipendenti passati.

Rilievi accolti:

1. Il benchmark salvava la memoria CUDA ma non la usava nel gate; caricamento e
   preprocessing completi erano rappresentati solo da un margine fisso. Ora misura
   caricamento, preprocessing dell'intero train fold0, RSS e memoria CUDA riservata;
   proietta la popolazione interna completa con margini espliciti e applica soglia
   85% della RAM/GPU prima dei fit. Costo tempo completo nel preventivo del benchmark.
2. Il nuovo estrattore di finestre ha preprocessing della classificazione. Esplicitato
   che per posa si usa soltanto depth_readouts sugli input del loader posa originale,
   preservando scala/sessione, margini e griglia. Nessun riuso cieco di subject_scales.
3. Separati nel protocollo i controller disponibili (continuazione e FT), la libreria
   comune pronta e i controller delle future valutazioni lunghe ancora da completare.

Seconda review: favorevole, nessun nuovo guasto individuato; indicizzazione dei128
esempi train corretta, hash ricetta/sorgenti compatibili con pin successivo del
benchmark, gate memoria applicato, semi testP2 0/1/2 confermati. Dodici test ripassati
in4,28s. Nessuna modifica o sottomissione del revisore.

Verifiche autore:54test passati sul nuovo pacchetto e sulle regressioni WSD e
tokenizer_train; dopo l'aggiunta del test che blocca piani non revisionati,13test
del pacchetto passati. Sintassi Bash dei due launcher valida. Le verifiche CPU
non sostituiscono il collaudo CUDA e non misurano tempo/memoria dei fit reali.

## Seguito dopo review Claude63fd9c5

Critico `review_followup_critic`, sola lettura, nessun job/commit. Correzioni FT:
nessun difetto bloccante trovato; griglia testa, persistenza e distinzione numerica,
selezione validation-only, ricevute split, warm-up separato e regola scratch verificati.
21test nuovi +13originali passati nella sua verifica. La funzione scratch richiede
ancora collegamento all'aggregatore finale, limite mantenuto nella consegna.

Continuazione:15test operativi passati nella sua verifica, archivioatomico confsync
e SHA, nessunoverwrite, quota numerica/progetto e recovery coerenti. Root ha aggiunto
anche verifica esplicita dell'età della quota con parametro submission_time (solo
pre-submit, non gate sulla durata della coda) e relativo test. Evidenza realequota
ancora nulla. Test autore complessivi dopo correzioni:123passati, inclusoWSD/wsd_chain/
tokenizer_train;100/100sorgenti scientifici storici byte-identici. Questi controlli
non sostituiscono il follow-up Claude né autorizzano lanci.
