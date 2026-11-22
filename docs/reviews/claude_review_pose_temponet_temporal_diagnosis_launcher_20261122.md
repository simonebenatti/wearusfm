# Review Claude: seconda diagnosi di GPT sul controllo temporale TCN e seguito (22/11/2026)

Oggetto: la diagnosi della chat GPT «TCN», distinta da quella su cui avevamo converguto in mattinata.
- Ramo `codex/pose-temponet-launcher-20261120`, commit `e8c9824461b5e18494ff462f1be8abda71b76d5e`, base
  `d3e41d06afbc5d47cb988488784a4655f16584af`.
- Documenti: `docs/reviews/pose_temponet_temporal_diagnosis_20261122.md` e
  `docs/prompt_claude_tcn_temporal_diagnosis_20261122.md`.
- Prova: `results/infra/pose_temponet_execution_20261121_A1/analysis_20261122/`.

Richiesta: testo di GPT incollato da Simone il 22/11, con in più le parole di Simone: «GPT ha finito i token, quindi qua,
quando hai fatto la tua review e proposto una soluzione puoi lanciare tu direttamente e impostarti un timer per guardare ogni
30 minuti».

## Esito, in tre verdetti

1. **Diagnosi: confermata, nessun blocco.** Tutti i numeri tornano. Quattro audit indipendenti della catena
   bersaglio–normalizzazione–maschere–loss–selezione non hanno trovato difetti. Manca una cosa nella lettura: il riferimento
   `cleanLS` è stimato sul bersaglio pulito, quindi una testa allenata sul rumoroso parte sotto zero (sezione 3).
2. **Diagnostici e controllo sul bersaglio pulito: approvati, e lanciati da me** con le regole già concordate con l'altra chat
   di GPT (sezione 5). La versione di questa diagnosi aveva esiti non decidibili («scarso», «buono»): sono sostituiti dalla
   tabella a priorità. Regole, costi e ID sono congelati in `docs/decisioni.md` del ramo
   `claude/tcn-temporal-diag-20261122` prima di qualsiasi job.
3. **Candidato residuale: non approvato come passo successivo.** I numeri archiviati non lo motivano e una simulazione
   giocattolo non mostra guadagno (sezione 6). Nessuna variante parte ora: si sceglie dopo i diagnostici, con i numeri in mano.

Su «allungherei subito i training»: le curve archiviate dicono che non aiuta (sezione 4). Non so se la frase sia tua o di GPT,
quindi la tratto come una richiesta possibile e la misuro: un terzo job, D2, prosegue fino a 300 epoche la corsa archiviata.
È descrittivo: non cambia la selezione né lo stato di A1.

**Che cosa lancio:** D0 (rilettura degli stati archiviati), D1 (fit sul bersaglio pulito) e D2 (corsa archiviata portata a 300
epoche). In tutto 2,40 GPU-ore prenotate, consumo atteso 0,9-1,3.

## 1. Metodo

- Letti diagnosi, prompt, protocollo congelato, codice dei controlli e JSON archiviati di A1.
- Sette revisori indipendenti senza accesso al cluster, ciascuno con una lente diversa: catena del bersaglio; supporto,
  maschere e pannelli; ciclo di training; numeri e prova di rappresentabilità; candidato residuale; operazioni e costi; mappa
  del codice. Ogni difetto serio dichiarato è passato a uno scettico. Esiti, note e script in
  `claude_review_pose_temponet_temporal_diagnosis_launcher_20261122_evidence/`.
- Alcuni revisori hanno usato una **simulazione giocattolo** su CPU: latenti sintetici, ma generatore, testa, loss e ricetta
  veri. Mostra che un comportamento è possibile, non che valga sui latenti veri. Dove la cito lo dico.
- Letture sul cluster in sola lettura: coda, quota, prenotazioni e formato del file di stato della TCN.

## 2. Che cosa ho verificato

| Affermazione di GPT | Esito | Nota |
| --- | --- | --- |
| Sei controlli `COMPLETED 0:0`; sorgenti e contesto `gpu8` attesi | confermato | sei ricevute con `source_sha256 9aaf1c8b…` e `numeric_sha256 12cb2a0b…`, sei nodi diversi |
| I tre controlli lineari passano | confermato | recuperi fra 97,46% e 99,73% contro il 90% |
| Recupero TCN 24,17% / 18,88% / 21,71% | confermato | ricalcolato dagli accumulatori: 24,173053 / 18,875450 / 21,710110 |
| Pooled: `cleanLS` 0,728479, oracolo 0,670659, soglia 0,699569, TCN 0,715926 | confermato | |
| Miglior `E_dyn` di tutte le curve 0,715826, cioè 21,88% | confermato | LR 0,003, esposizione 27; nessuno dei 915 punti scende sotto la soglia |
| MLP 0,735130; la TCN migliora del 2,61% | confermato | |
| Fitting pooled `E_dyn` 0,685897 e MSE 1,698865; esclusi 0,715926 e 1,800145 | confermato | |
| `E_dyn` contro `y*` sugli esclusi: TCN 0,139236, MLP 0,196055 | confermato | L0 0,199032 |
| Curve complete: 300 epoche a LR 1e-4, 150 agli altri | confermato | scelta a LR 0,003, esposizione 31, update 899 / 930 |
| Prova di rappresentabilità: errore relativo 4,66e-7 | confermata | script rieseguito, JSON identico |
| «Nessun difetto di codice dimostrato» | confermato | sezione 3 |

Sulla prova di rappresentabilità: è corretta, ma vale solo in valutazione. Con il dropout attivo gli stessi pesi non sono un
punto fermo della ricetta: nel giocattolo la loss pulita attesa in modalità di training è fra 0,091 e 0,141, contro circa zero in
valutazione. Dice che la funzione sta nella classe, non che sia raggiungibile dall'allenamento.

## 3. Difetti cercati nella catena: nessuno

Cercato un errore concreto in generazione del bersaglio, LayerNorm, supporto e ritaglio, loss, selezione, uguaglianza dei
pannelli. Con prove su CPU fatte con le funzioni vere e con i JSON archiviati:

- **Stesso bersaglio nei tre luoghi.** Calibrazione, training delle teste e cancello usano lo stesso `y`. Varianza del rumore
  sugli esclusi 1,6918 e 1,6928 contro 1,69. Nessuna fuga del rumore degli esclusi nel training.
- **Stesso supporto.** Loss, metrica di selezione e cancello usano le stesse patch (6-153), gli stessi giunti e le stesse
  finestre. Nessuno sfasamento.
- **Stessi pannelli.** `cleanLS`, oracolo e TCN hanno gli stessi `window_count` (934, 920, 1.854), lo stesso denominatore
  dinamico (identico nel fold 0, a un'ultima cifra nel fold 1) e la stessa soglia.
- **Pooled.** È il rapporto delle somme, nel cancello e nella selezione: 21,71% e non 21,52%.
- **Ciclo di training.** Stato migliore copiato in profondità subito dopo la valutazione; dropout spento in valutazione; il
  gradiente raggiunge il ramo profondo dal secondo passo; ripresa identica alla corsa continua. Il clip a norma 1,0 nel
  giocattolo scatta in 5 passi su 1.740.

Note, nessuna delle quali spiega il cancello:
- Il generatore legge le patch di riempimento grezze, la testa le vede azzerate. Qui non conta: tutte le 1.907 finestre hanno
  160 patch vere.
- Selezione e cancello leggono gli stessi utenti esclusi. È il disegno, e favorisce la TCN.
- «Nessun `optimizer_alarm`» certifica poco: guarda solo la curva del learning rate vincente.
- Le statistiche del generatore sui latenti veri (unità attive, scale) stanno solo dentro `calibration.pt`. D0 le estrae.

**La precisazione che manca nella diagnosi.** `cleanLS` è stimato sul bersaglio pulito e valutato sul rumoroso. Le teste sono
allenate sul rumoroso. Sulla scala del cancello, dove 0% è `cleanLS` e 100% è l'oracolo:

| Lettore, allenato su `y` | fold 0 | fold 1 | pooled |
| --- | ---: | ---: | ---: |
| TCN | +24,17% | +18,88% | +21,71% |
| MLP | −12,24% | −10,65% | −11,50% |
| L0 neurale | −13,55% | −10,82% | −12,28% |
| L0 in forma chiusa | −19,10% | −16,38% | −17,84% |

Partendo da L0 neurale la TCN chiude il 30,27% della distanza dall'oracolo; il cancello ne chiede il 55,47%. Il cancello resta
valido: il protocollo dichiarava l'asimmetria. Cambia la lettura del 21,71%.

**Un sospetto nuovo, da misurare in D0.** Due revisori, in modo indipendente, hanno visto nel giocattolo che le ReLU del
penultimo strato del ramo profondo (64→32) si spengono quasi tutte nelle prime due epoche. Uno scettico l'ha riprodotto e
precisato:
- a LR 0,003 la frazione di attivazioni positive crolla da circa 0,5 a non più di 0,18 in 18 corse su 18, con sei costruzioni
  diverse dei latenti sintetici. Poi il ramo si riaccende: in un seme su cinque resta dormiente per circa 45 epoche;
- a LR 0,005 e 0,007 il ramo è spento a 10 epoche in 3 semi su 3; a LR 0,01 in 17 corse su 18, e resta spento per tutte le
  150 epoche in 3 corse lunghe su 3. Una TCN con il ramo spento si comporta come L0;
- nessun allarme del selettore lo vede.

Sui dati veri non è dimostrato. È però coerente con una cosa archiviata: la curva della TCN a LR 0,01 resta piatta al livello di
L0 per 150 epoche e non peggiora, mentre a LR 0,003 la TCN supera L0 e poi peggiora. Se è così, il bordo alto della griglia è
un L0 travestito e la ricerca del learning rate si ferma di fatto a 0,003, proprio dove il recupero massimo cresce ancora con
il learning rate (5,0%, 5,4%, 10,8%, 21,9%). D0 conta le unità attive su ogni stato salvato, compresi quelli a LR 0,01.

## 4. Allungare i training: le curve dicono di no

Recupero pooled lungo la curva archiviata del learning rate scelto, 0,003:

| Epoca | 10 | 31 (scelta) | 50 | 75 | 100 | 150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recupero | +3,07% | +21,71% | +13,43% | −1,15% | −11,06% | −22,62% |

- Tutti e cinque i learning rate hanno il minimo sugli esclusi ben prima della fine. Recupero a fine corsa: +0,57% (LR 1e-4,
  300 epoche), −5,93%, −39,81%, −22,62%, −15,79%.
- Fra le epoche 18 e 35 il recupero sta fra il 19% e il 22%: non è un picco stretto mancato dalla selezione.
- La ricetta congelata ha un tetto a 300 epoche; oltre servono sorgenti nuovi.

Con questa ricetta e questo bersaglio rumoroso, più epoche allontanano dalla soglia.

Lo misuro comunque, perché costa poco: **D2** riprende le due traiettorie dallo stato archiviato a 150 epoche e le porta a 300,
il tetto della ricetta congelata. Previsione scritta prima: nessuna epoca fra 151 e 300 scende sotto l'`E_dyn` del punto scelto
(0,715926). Se una scende, ho torto e allungare aiuta.

D1 mette alla prova la spiegazione: se allenando sul pulito il recupero sale e regge, il calo è rumore imparato, e il rimedio
non sono le epoche. In D1 l'orizzonte segue la regola congelata della ricerca: 150 epoche, 300 solo se il minimo cade nelle
ultime cinque.

## 5. Diagnostici: che cosa cambia rispetto alla proposta di questa diagnosi

La proposta va nella direzione giusta. La eseguo nella forma già concordata con l'altra chat di GPT (`8fac3a82`, nota
`4a48b814`), che è più precisa. Differenze, tutte dichiarate nel protocollo:

| Punto | In questa diagnosi | Che cosa lancio |
| --- | --- | --- |
| Diagnostici senza fit | «senza nuovi fit», costo non dichiarato | è un job su GPU (D0): gli stati stanno su Leonardo. 20 stati TCN più i 4 scelti di L0 e MLP, con parità a 1e-12 |
| Durata del controllo pulito | «massimo 150 epoche» | la regola congelata della ricerca: 150, e 300 solo se il minimo cade nelle epoche 146-150 |
| Letture | epoche 1, 10, 31, 75, 150 | le stesse, più il punto scelto; curve complete su fitting ed esclusi, contro `y` e `y*` |
| Esiti | «scarso», «buono», «intermedie inconclusive» | tabella a priorità con soglia 0,50, senza sovrapposizioni |
| Soglie | «non riusare meccanicamente le soglie del rumoroso» | il punteggio che decide resta contro `y`, come nel cancello: stessi denominatori, stessa soglia. Le letture contro `y*` sono descrittive |
| Parità del driver | non prevista | prima dei fit puliti, lo stesso driver su `y` deve ripercorrere la prima epoca archiviata |
| Scomposizione dell'errore | totale, centrato, livello | MSE ed `E_dyn` contro `y` e `y*`, più la scomposizione esatta in errore pulito, livello del rumore e allineamento al rumore |
| Rami affine e profondo, deriva della LayerNorm | chiesti | presenti: energia dei due rami, unità vive per strato, scarto della LayerNorm dall'identità |
| Prova di rappresentabilità sui dati veri | chiesta | presente: il generatore scritto nella testa è valutato dal valutatore congelato sui latenti veri |
| Training più lunghi sul rumoroso | non previsti | D2: la corsa archiviata a LR 0,003 portata da 150 a 300 epoche, solo descrittiva |

Protocollo completo: `docs/tcn_temporal_diag_protocol_20261122.md` sul ramo `claude/tcn-temporal-diag-20261122`.

## 6. Candidato residuale: perché no, per ora

Il candidato: LayerNorm fissa, base ridge stimata sul fitting con `alpha` dalla CV interna e poi congelata, TCN allenata sul
residuo.

1. **Lo scambio della base vale poco.** La base del candidato non è archiviata nel mondo temporale: la stima è circa −10,5%
   pooled sulla scala del cancello, contro −12,28% di L0 neurale. Cambiare la parte lineare vale quindi fra 2 e 7 punti pooled,
   secondo quanto vale oggi il ramo affine della TCN, che nessuno ha misurato. Ne mancano 28,29.
2. **Il resto deve venire dal ramo profondo, che vede lo stesso segnale.** Con quella base servirebbero circa 60 punti dal ramo
   temporale. Oggi l'intera TCN ne ottiene circa 32 sopra quella linea: servirebbe quasi il doppio, e il candidato non dà al
   ramo profondo niente di nuovo da cui impararlo. Lo scrive anche GPT: non toglie rumore e non aggiunge informazione.
3. **Nessun indizio che i due rami si disturbino.** Nel mondo lineare la TCN attuale recupera già il 98,27% pooled.
4. **Nel giocattolo non alza il livello.** Con il ramo profondo attivo: candidato 0,311 e 0,290, ricetta attuale 0,315 e 0,291.
   Nessuna variante provata arriva a 0,50. Il suo solo vantaggio misurato è evitare lo spegnimento iniziale del ramo a LR
   0,003; a LR 0,01 il ramo si spegne anche con la base congelata.
5. **Sono due fattori, non uno.** La LayerNorm fissa non ha effetto misurabile nel giocattolo e non è sostenuta da alcun dato.
6. **Costa una catena nuova.** Cambiare la testa cambia l'identità dei sorgenti: le ricevute di A1 decadono, i comparatori L0
   e MLP cambiano significato, e nel mondo lineare il controllo diventa quasi tautologico perché la base è la ridge di
   riferimento.

Due cose che avevo scritto nella prima stesura e che gli scettici hanno corretto:
- «la base parte a −17,84% e L0 neurale sta 5,5 punti sopra»: quel numero è il lettore in forma chiusa senza regolarizzazione,
  non la ridge del candidato. Nel mondo lineare, sui latenti veri, la ridge con `alpha` dalla CV è anzi migliore di L0 neurale
  di 1,64 punti del margine temporale;
- «l'errore della base diventa invisibile alla TCN»: l'algebra è vera, ma vale per la parte lineare di qualunque ricetta. Non
  è uno svantaggio del candidato.

**Che cosa sceglierei, a seconda dei risultati.** Sono ipotesi da fissare prima di provarle, una sola alla volta, con conferma su
un mondo nuovo:
- D0 trova il ramo spento a LR 0,01 e molte unità inattive a 0,003: il problema è l'avvio, e la griglia del learning rate è
  tagliata dallo spegnimento. Modifiche più piccole del candidato: un breve riscaldamento del learning rate, oppure il ramo
  affine inizializzato alla ridge e non congelato. Nel giocattolo evitano lo spegnimento a LR 0,003, non a 0,01.
- D0 trova il ramo sano e D1 passa: il limite è il rumore con questo campione. La leva è la regolarizzazione, oggi quasi
  assente (il weight decay contrae i pesi del 2,7% in 899 passi). Nel giocattolo un weight decay di 0,3 porta il recupero da
  0,31 a 0,42 in un seme e da 0,29 a 0,37 in un altro: è l'unica modifica singola che alza il livello, e non arriva a 0,50.
- D1 non passa: né il candidato né la regolarizzazione sono indicati. Va riletta la specifica del controllo.

## 7. Costi, identità e ricevute

- **Riusabile in sola lettura**, con verifica di hash: le due calibrazioni, i sei file di stato e di selezione, le ricevute
  della catena, la cache dei latenti, il piano dei controlli come tabella di pin, il pin numerico `gpu8`, i 247 sorgenti.
- **Nuovo:** riferimento di deploy, checkout di campagna, ID logici `TCND-*`, radice delle uscite, ricevute proprie. Il codice
  nuovo sta fuori dal manifest dei sorgenti: l'identità `9aaf1c8b…` non cambia.
- **Costo misurato** (benchmark 65016447 e sei controlli): un'epoca dei due fold 2,33 s; una valutazione degli esclusi 2,72 s.
  D0 circa 4 minuti; D1 circa 25 minuti, 45 se prosegue a 300 epoche; D2 circa 25 minuti. Consumo atteso 0,9-1,3 GPU-ore.
- **Prenotato:** D0 20 minuti, D1 75 minuti, D2 40 minuti: 2,40 GPU-ore con la convenzione di A1, cioè 19,2 ore locali.
  Sotto-budget della catena 4 GPU-ore. Il valore «0,2-0,4 GPUh» di GPT è coerente come consumo del solo controllo pulito a 150
  epoche.
- **Candidato, se un giorno si facesse:** catena intera da rifare. Tutto lo speso di A1 fino ai controlli è 3,681 GPU-ore e
  31,975 core-ore: il calcolo non è l'ostacolo, lo sono definizione e review.

## 8. Che cosa non ho verificato

- Niente è stato misurato sui latenti veri: lo faranno D0 e D1.
- Lo spegnimento delle ReLU è dimostrato solo nel giocattolo: latenti sintetici, un fold.
- Il livello della base ridge del candidato nel mondo temporale è una stima: non è archiviato.
- Gli effetti di weight decay, riscaldamento e inizializzazione alla ridge vengono dal giocattolo: uno o due semi, un fold,
  40 epoche.
- Il codice dei diagnostici è mio: la sua review è nel ramo `claude/tcn-temporal-diag-20261122`, fatta da revisori indipendenti.

## 9. Testo per GPT

```text
Review Claude della diagnosi TCN e8c9824461b5e18494ff462f1be8abda71b76d5e (ramo codex/pose-temponet-launcher-20261120).
Documento: docs/reviews/claude_review_pose_temponet_temporal_diagnosis_launcher_20261122.md, ramo claude/temponet-db8-20261109
(commit nel messaggio di Simone), con la cartella _evidence.

Tre verdetti.
1. Diagnosi: confermata, nessun blocco. Tutti i numeri tornano dagli accumulatori; la prova di rappresentabilità è corretta
   (rieseguita, JSON identico) ma vale solo in eval. Quattro audit indipendenti della catena bersaglio-LN-supporto-loss-selezione:
   nessun difetto. Manca una precisazione: cleanLS è stimato sul pulito, L0 neurale sta a -12,28% pooled, da lì la TCN chiude il
   30,27% e il cancello ne chiede il 55,47%.
2. Diagnostici e controllo pulito: approvati ed eseguiti da Claude su incarico di Simone («puoi lanciare tu direttamente»), nella
   forma già concordata con l'altra tua chat (8fac3a82, nota 4a48b814): D0 è un job con parità a 1e-12 su 20 stati TCN più 4 di
   L0/MLP; D1 ha parità del driver sulla prima epoca, selezione invariata su y, orizzonte secondo la regola congelata (150
   epoche, 300 solo se il minimo cade in 146-150) e tabella a priorità con soglia 0,50 al posto di «scarso/buono». Aggiunte le
   tue letture a 1, 10 e 75 epoche, energia dei rami affine e profondo, unità ReLU vive, deriva della LN e il generatore scritto
   nella testa valutato sui latenti veri. In più D2, solo descrittivo: la corsa archiviata a LR 0,003 portata a 300 epoche.
   Protocollo e regole congelate: docs/tcn_temporal_diag_protocol_20261122.md e docs/decisioni.md sul ramo
   claude/tcn-temporal-diag-20261122. A1 resta ferma.
3. Candidato residuale: non approvato come passo successivo. Lo scambio della base vale poco: la ridge con alpha da CV è
   stimata a circa -10,5% pooled contro -12,28% di L0 neurale (2-7 punti sui 28,29 mancanti); il resto, circa 60 punti sopra
   quella linea contro i circa 32 di oggi, dovrebbe venire dal ramo profondo, che vede lo stesso segnale rumoroso. Nel mondo
   lineare la TCN recupera già il 98,27%; in una simulazione giocattolo (latenti sintetici) il candidato non alza il livello
   (0,311 / 0,290 contro 0,315 / 0,291) e la LN fissa non ha effetto. Cambia inoltre il significato dei comparatori e fa
   decadere le ricevute.

Sospetto nuovo, da misurare in D0: nel giocattolo le ReLU del penultimo strato si spengono quasi tutte nelle prime due epoche
(a LR 0,003 in 18 corse su 18, poi il ramo si riaccende; a LR 0,01 resta spento per 150 epoche in 3 corse su 3). La curva vera
a LR 0,01 è piatta al livello di L0 per 150 epoche: il bordo alto della griglia potrebbe essere un L0 travestito.

Allungare i training: le curve dicono di no. Curva vera a LR 0,003: +21,71% a 31 epoche, +13,43% a 50, -1,15% a 75, -22,62%
a 150; tutti e cinque i learning rate hanno il minimo prima della fine. D2 lo misura fino a 300 epoche.

Dopo i risultati nessun passo parte da solo. Alternative più piccole del candidato, da fissare prima e una alla volta:
riscaldamento del learning rate o affine inizializzata alla ridge e non congelata, se D0 trova il ramo spento; regolarizzazione
più forte, se D0 è sano e D1 passa.
```
