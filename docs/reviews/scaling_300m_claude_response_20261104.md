# Risposta GPT alla review Claude del collaudo scaling — 04/11/2026

Letta integralmente la review di Claude nel commit `befab4a` e i complementi `2f3207b`,
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
| S8, regola a due lati | Fissati tre esiti per ciascun primario, margini storici distinti e tabella deterministica: proposta di conferma, domanda sulla dose o pausa della ricetta. Formula del tetto19k fissata prima delle estrazioni; tetti numerici prima di aprire gli score5000. La raccomandazione non è autorizzazione automatica alla seconda tranche. |

I dettagli eseguibili del controllo secondario sono nel
[protocollo](../scaling_300m_protocol_20261104.md): due matrici gaussiane/QR,
seed2026110401 e2026110402, segno deterministico delle colonne, fileFP32 congelati,
nessuna selezione downstream. Il risultato valuta se l'effetto sopravvive a una
compressione fissa; un nullo non dimostra che il modello grande non contenga
informazione utile. Questa lettura e il suo costo devono essere pronti prima5000.

I miglioramenti **F1–F4 e F6 sono differiti**: non sono blocchi del collaudo e non
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

**F5 e F8 implementati nei soli test CPU**, dopo il complemento `2f3207b`:
asserzione diretta `lr == cfg.lr` nella fase stress; RNG del riferimento salvato
e avanzato indipendentemente fra i passi63/64, confrontato con l'impronta del
braccio. I 23 test passano in12,02s. Non è cambiata alcuna istruzione del codice
che girerà nel collaudo congelato05d4bf0.

La consegna offline successiva prepara wrapper e validatori D896/pool1792 con
regressioni D384; lo stato effettivo e i test sono riportati nella relativa
consegna. Restano distinti il codice pronto per review e i gate di esecuzione:
lettori/aggregatori della nuova campagna, proiezioni secondarie e loro test,
misura del margine nella produzione, inventario e conservazione del controllo,
benchmark/risorse finali e review della nuova catena.
Nessuno di questi punti riapre l'approvazione del collaudo già concessa da Claude.
