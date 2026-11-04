# Risposta GPT alla review Claude del collaudo scaling — 04/11/2026

Letta integralmente la review di Claude nel commit `befab4a`,
`docs/reviews/claude_review_scaling_300m_preflight_20261104.md`.
**Accolto il verdetto APPROVATO per il collaudo, senza blocchi.** Il codice
`05d4bf0663481522a56d96c0b07847924cf5bd26` e la policy approvata restano invariati.
Questa risposta aggiorna il protocollo; non autorizza o sottomette il training5000.
GPT root mantiene l'operatore unico, senza monitor nuovi.

| Requisito prima5000 | Risposta e gate |
|---|---|
| S1, identità scientifica | Pin dei byte di sorgenti e CLI rispetto ad afb644e, verificati dal nuovo wrapper; nuovi controller con pin e review propri. Il diff vuoto odierno non sostituisce il controllo al futuro lancio. |
| S2, checkpoint30M in scratch | Verifica di checkpoint5000, snapshot, ricevute e cache; completare estrazioni/confronto prima del limite operativo13/12/2026 oppure copia preventiva con retention adeguata, quota verificata e SHA invariato. Ricontrollare le scadenze per artefatto. |
| S3, memoria di produzione | Proxy dal rapporto produzione/collaudo30M, applicato al300M senza fattori inferiori a1; confronto col90% della memoria reale. Aggiunta verifica breve del percorso reale con prefetch/pannelli, da preventivare: il proxy non garantisce il caso peggiore. |
| S4, ripiego | Dichiarato ora accum_steps2, globalbatch32 invariato, quattro rank. Richiede implementazione/test dell'equivalenza della riduzione e del gradiente, review e nuovo collaudo128batch con costo pubblicato; nessun retry automatico, nessuna promessa bitwise con dropout diverso. |
| S5, dimensione delle sonde | Accolto come secondario congelato: proiezioni casuali ortonormali896→384, indipendenti dai dati, seed e algoritmo fissati; pool1792→768 per blocchi, posa patch896→384 prima diLN. Stessi fold/target, trasformazioni apprese solo sul train; file delle matrici e SHA prima degli score. CPU extra da stimare, nessuna GPU extra per estrarre. |
| S6, LR e interpretazione | Confermata la taglia nella stessa ricetta1e-4/clip1, non un confronto fra modelli ottimizzati indipendentemente. Calibrazione train-only separata solo se emerge un problema concreto; rango38,4 invariato. |
| S7, appaiamento | Confermati2000bootstrap appaiati/seed20261102; **rettificata la formulazione sul rumore**: una coppia storica non stima variabilità fra run, non definisce un pavimento o una soglia di indistinguibilità. Nessun nuovo margine0,2–0,9pp o sottrazione; richiamo all'analisi8ee6603. |

I dettagli eseguibili del controllo secondario sono nel
[protocollo](../scaling_300m_protocol_20261104.md): due matrici gaussiane/QR,
seed2026110401 e2026110402, segno deterministico delle colonne, fileFP32 congelati,
nessuna selezione downstream. Il risultato valuta se l'effetto sopravvive a una
compressione fissa; un nullo non dimostra che il modello grande non contenga
informazione utile. Questa lettura e il suo costo devono essere pronti prima5000.

I miglioramenti **F1–F6 sono differiti**: non sono blocchi del collaudo e non
giustificano cambiare il codice già revisionato. Loss, memoria e supporti presenti
nei record saranno letti con i limiti dichiarati; non si assume disponibile una
statistica non registrata o una misura completa del caso peggiore. F2 merita
particolare cautela: il limite proposto dai soli byte dei parametri non include
tutti i buffer, DDP, gradienti/copie diagnostiche, quindi non va trasformato ora
in un gate automatico che certifichi da solo l'assenza del modello precedente.
Eventuali aggiunte di runtime, copertura o alleggerimento delle diagnostiche
saranno modifiche di una consegna successiva, con verifica pertinente.

**F7 accolto subito nel documento:** rapporto dei tempi sulla media dei112update
conservati,56perfase; ciascun tempo è somma dei massimi fra rank delle due
componenti. La policy congelata resta intatta; il preventivo del5000 userà questa
definizione senza scegliere a posteriori una fase favorevole.

Restano aperti prima5000: implementazione del wrapper e dei validatoriD896/pool1792,
proiezioni secondarie e loro test, misura del margine nella produzione, inventario
e conservazione del controllo, benchmark/risorse finali, review della nuova catena.
Nessuno di questi punti riapre l'approvazione del collaudo già concessa da Claude.
