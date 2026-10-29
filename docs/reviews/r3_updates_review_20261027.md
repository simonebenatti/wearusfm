# WearUsFM — lettura degli aggiornamenti R3 del 27 ottobre 2026

**Parere:** completare il pilot firmato con M = 280, trattando il margine come scelta operativa ancora incerta. Il lavoro nuovo rende la diagnosi più utile, soprattutto per la regressione. Servono due correzioni circoscritte alla valutazione E2 prima di interpretarne i risultati. La lettura causale del profilo dei bordi va ridimensionata.

## Perimetro e riferimenti

Letti il checkout Claude fino a d43dbcf (27/10, 11:43), registro, decisioni, protocollo della posa ed emendamento, report e JSON del pilot precedente, codice dell'ingest e della valutazione. Base del branch GPT: 4680617. La lettura integra due verifiche indipendenti con Astra, richiesto da Simone.

I percorsi riportati sotto sono relativi al checkout Claude: /Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine. Stato operativo dai documenti locali, senza interrogare il cluster. Nessuna modifica al codice di Claude, alle soglie firmate o ai job; questa è una review mirata, non una certificazione completa della consegna.

## 1. Gli aggiornamenti consolidano il pilot precedente

La campagna pilot-obiettivi-20261024 è chiusa: endpoint a una e due epoche, letture pool e decoder per canale, provenienza e consuntivo. Nessuna variante supera il criterio firmato. I descrittivi non mostrano un vantaggio generale nascosto a una epoca o nella lettura per canale. R2 ha un piccolo vantaggio P1-dec a una epoca (+0,49 punti), inferiore alla soglia pratica e non mantenuto a due epoche. R1 e R5 perdono su EPN-612 con entrambi i readout e a entrambe le epoche.

R5 cresce più rapidamente fra i due endpoint, ma due punti senza intervallo appaiato non stabiliscono una legge di apprendimento né giustificano un nuovo training più lungo. Le curve restano informative, non una promessa di sorpasso. La vicinanza R0-e1/C1 (18,32 contro 18,21) sostiene la coerenza del punto di partenza.

R3c ha completato tronco e ramo (44,5 GPU-ora); la sua valutazione anticipata è autorizzata e le quattro estrazioni sono registrate come sottomesse. È una scelta utile: il confronto con R0 chiarirà l'effetto di togliere JEPA prima dell'attribuzione del contributo della posa. R3 resta nella catena ingest, collaudo e training: non c'è ancora un risultato scientifico da giudicare.

Fonte: results/pilot_obiettivi_20261024/README.md, pendenza.json e confronti/; docs/claude_GPT_shared, voci 27/10. Piccolo errore testuale nel README: nel paragrafo dei descrittivi a una epoca l'IC di R5 pool P1 a due epoche è scritto [-1,09; +0,05]; il JSON del criterio e la tabella iniziale danno [-1,0735; -0,0166] punti. Il verdetto di non adozione non cambia.

## 2. M = 280: compromesso operativo, non misura dell'estensione dell'artefatto

Ho ricalcolato margin_from_collaudo sul rapporto 62409366: l'oggetto ottenuto è identico al JSON di lettura salvato. Il campione comprende 40 registrazioni di 10 utenti nuovi, 4.544.475 righe e 408 blocchi di zeri. Le righe nulle sono il 9,403%, contro il 17% del primo campione di due utenti.

| Scelta | Durata per lato | Tetto di esclusione aggiuntiva sul campione |
|---|---:|---:|
| M = 280, firmato da Simone | 140 ms | 5,028% |
| M = 2.800, restituito dalla v1 | 1.400 ms | 50,276% |

Sono tetti calcolati come 2 M × numero di blocchi / righe, prima di sovrapposizioni e bordi, oltre alle righe già nulle. Non sono la perdita effettiva dell'intero corpus né il conteggio finale delle patch che riceveranno supervisione.

La decisione di mantenere la riserva come descrittiva è corretta. Il profilo lungo smentisce l'ipotesi iniziale di un plateau persistente: la condizione R2 fallisce. Non identifica però separatamente movimento reale, artefatti di elaborazione ed errore della cinematica inversa. La differenza prima assoluta della posa aumenta sia per un movimento veloce sia per una misura inaccurata.

Le frasi di decisioni.md secondo cui «oltre i primi 20 ms l'eccesso non è artefatto», «la coda è comportamento» e una posa più scattosa «come bersaglio è rumore anche se non è artefatto» eccedono l'evidenza. Sono ipotesi, e l'ultima confonde dinamica reale con rumore: per la regressione della traiettoria quella dinamica può essere proprio il segnale utile.

Anche il controllo per altezza del salto va interpretato come associazione. L'altezza è stimata dalla norma della posa a 301–400 campioni dal bordo; i terzili possono differire per postura, utente, movimento e tracking. L'assenza di un rapporto alto/basso elevato non esclude artefatti comuni ai gruppi o indipendenti dall'ampiezza. Il riferimento lontano può inoltre selezionare periodi più calmi.

Nel JSON R5 supera la propria soglia in 104 bin, compresi intervalli ben oltre i primi 40 campioni, fino a 3.570. Questo non prova che siano tutti artefatti: mostra che l'indicatore non isola un unico transitorio iniziale. M5 = 270 è l'ultimo superamento ENTRO 300 campioni per costruzione; anche M3 è limitato a 300. La vicinanza M3/M5 e la sensibilità 260–300 rispetto alle soglie scelte non sono due conferme indipendenti o un intervallo d'incertezza.

M = 280 resta un compromesso ragionevole per procedere col pilot autorizzato. La firma è esplicita e la spesa di supervisione sul campione è contenuta. Il risultato riguarderà la posa con questa maschera di validità; un insuccesso non escluderà l'intera famiglia di obiettivi motori. La copertura effettiva dell'ingest e dei target validi aiuterà a quantificare il limite.

Fonti: docs/decisioni.md:4697; results/passo2/emg2pose_pose_ingest_collaudo_62409366.json; emg2pose_pose_margine_lettura_62409366.json; src/wearusfm/ingest/emg2pose_pose.py:151, :467, :577.

## 3. La nuova valutazione è utile anche per la regressione

E1 verifica la posa su utenti esclusi dal training; E3 conserva l'asse temporale e legge la posa dai latenti congelati con una ridge, scegliendo la regolarizzazione fra utenti del train. Questo risponde alla richiesta di valutare anche regressione, anziché ridurre tutto alla classificazione su medie temporali.

Il criterio di adozione P1/P2 resta quello firmato e riguarda la classificazione. Un beneficio robusto di R3 su E1/E3 sarebbe comunque un risultato scientifico utile per la regressione della posa su nuovi utenti, anche senza passaggio del criterio P1/P2. Non proverebbe ancora trasferimento della regressione a forza, altri montaggi o altri dataset.

Lettura quantitativa necessaria: se R3c e R0 sono già sotto 0,9 in E3 non segue che R3 aggiunga poco; occorre guardare quanto migliora rispetto a loro. Analogamente, superare soglie diverse non sostituisce un confronto diretto con incertezza.

E4 osserva norme e coseni su un microbatch locale del processo 0, come dichiarato in E9. Con intervallo 500 la finestra 500–1300 contiene normalmente due misure. È un utile allarme grossolano; la frase «la posa non muove l'encoder» sarebbe troppo forte senza considerare composizione del microbatch, altri processi e accumulo.

## 4. Due correzioni concrete alla valutazione E2

### Gap fra allineato e disallineato su supporti diversi

In src/wearusfm/harness/pose_eval.py:384–392 il disallineato usa solo le coppie conservate e l'intersezione delle maschere, mentre la MAE allineata sottratta viene da tutte le finestre dinamiche dell'utente. Questo può invertire il segno del gap.

Riproduzione con tre finestre dinamiche: le due appaiabili hanno MAE allineata 0 e disallineata 1; la terza, senza partner, ha MAE allineata 100. Il gap corretto sulle coppie è +1, quello prodotto dal report è -32,333. Correzione richiesta: calcolare entrambi gli errori sulle medesime finestre e sul medesimo supporto valido, prima dell'aggregazione per utente. Il difetto riguarda il gap descrittivo, non modifica direttamente il gate E2.

### Fallback fra mani opposte della stessa sessione

In pose_eval.py:280–297 il fallback richiede solo stesso utente/fase e registrazione diversa. Può accoppiare sinistra e destra della stessa sessione allo stesso istante: la riproduzione restituisce partner [1, 0]. Il protocollo parla di un'altra sessione; inoltre lo scambio può conservare movimento bimanuale simultaneo o cambiare mano sollecitata.

Se la standardizzazione per lato è attiva, il codice confronta anche coordinate standardizzate diverse: il target j è espresso con le statistiche di j, ma E2 applica la deviazione di i. La distanza può non coincidere con quella fisica degli angoli. Correzione richiesta: costruire coppie coerenti per lato/sessione e disallineamento, oppure convertire esplicitamente entrambi i target in uno spazio comune. Quantificare coppie escluse/fallback. La frequenza reale di questo caso e l'attivazione di per_side non sono state misurate.

Lo script repro_pose_eval_20261027.py accanto a questa nota esegue entrambi i casi sul codice reale del checkout passato come argomento. La correzione riguarda l'harness di valutazione; non cambia il training di R3/R3c.

Sul piano scientifico, E2 migliore del costante globale non dimostra automaticamente leakage: fase, postura e componente lenta possono essere informazione legittima. Il criterio congelato può segnalare che il controllo non discrimina abbastanza bene il contributo dell'allineamento; la causa va mantenuta aperta.

## 5. Prosecuzione consigliata e verifiche eseguite

Completare ingest e collaudo R3 secondo la firma, leggere R3c appena disponibile e portare a termine la prova prevista. Far correggere a Claude i due punti E2 prima di usare quel rapporto. Le letture E1/E3 e i confronti R3/R3c/R0 permetteranno poi di distinguere apprendimento, accessibilità e trasferimento, mantenendo esplicita l'incertezza sui target.

Verifiche indipendenti: ricalcolo integrale della lettura del margine identico al JSON; controllo dei conteggi e dei tetti; lettura dei JSON di criterio e pendenza; due riproduzioni sintetiche E2 sul modulo reale. La suite completa di 1.448 test è riportata da Claude e non è stata rieseguita in questa review. Nessuna nuova campagna o lettura della letteratura; restano esclusi TinyMyo e PhysioWave e inclusi gli approcci EEG, con attenzione distinta a classificazione e regressione.
