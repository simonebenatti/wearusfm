# Review interna: lettori e memoria di produzione — 4 novembre 2026

Verdetto: **APPROVATO per la consegna offline e il collaudo S3**, dopo una correzione.
Non attesta la memoria reale né la readiness del training scientifico a 5.000.

Il revisore `scaling_evaluation_contract`, distinto dall'autore delle sonde/S3,
ha riletto il percorso delle sonde e rieseguito 56 test. Ha confrontato S3 con
il produttore reale del trainer e trovato un errore concreto: `fixed_eval_set.windows`
è piatto (256 righe), non annidato (8×32). L'autore ha corretto guardia e fixture,
aggiungendo una regressione che chiama il produttore reale. Dopo la correzione:
164 test ripassati indipendentemente, nessun altro blocco.

Controllati: prove di 500 update; diagnostiche, pannelli e gradienti; quattro
misure CUDA e riconciliazione col summary; output freschi, nessun resume;
report scritto per ultimo; runtime Python/Torch/CUDA confrontato prima del lavoro,
cuDNN registrato senza inventare un valore storico; 103 sorgenti invariati.
Il revisore ha aggiunto 37 test separati della readiness reale, positivi e negativi.

Il secondo revisore `scaling_300m_execution`, distinto dall'autore del launcher,
ha riletto launcher e preventivo: sintassi Bash valida, allocazione 1 nodo/4 GPU/
32 CPU/480G/60 minuti, checkout e piano congelati, ambiente e supervisor coerenti.
Nessun blocco. GPT principale ha riletto il comparatore e le identità comuni:
primari/secondari distinti, selezione train-only, shard e OOF confrontati,
ricostruzione dei risultati della posa e delta scritto a precisione piena.

Verifica integrata del principale: **425 test passati in 38,01 secondi**;
`git diff --check` e `bash -n` passano. Diff vuoto sui sorgenti scientifici e
sul collaudo già approvato a `05d4bf0`. Hash e perimetro esatto della verifica in
`results/infra/scaling_300m_20261104/readers_memory_verification.json`.

Le matrici S5 sono state caricate e verificate dal revisore: manifest, hash,
ortogonalità e identità del generatore coincidono. Nessun dato wearable nei file.

Limiti: i test CPU non misurano picco o durata reali su GPU. La suite completa
storica di Claude ha 2.275 passati e un fallimento preesistente B19618, documentato
nella sua review `5e4eb8a`; non è stato corretto qui alterando il trainer congelato.
La review Claude del nuovo codice resta richiesta come controllo esterno.
