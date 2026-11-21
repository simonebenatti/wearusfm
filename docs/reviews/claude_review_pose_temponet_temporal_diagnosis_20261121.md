# Review Claude: diagnosi del controllo temporale TCN e seguito D0/D1 (21/11/2026)

Oggetto: la diagnosi di GPT sul controllo temporale fallito e il seguito proposto.
- Ramo `codex/pose-temponet-temporal-diagnosis-20261121`, commit `e7670758de9d0b6e25a9ad880ec3e1f7d573f169`, base
  `d3e41d06afbc5d47cb988488784a4655f16584af`.
- Documento: `docs/reviews/pose_temponet_temporal_diagnosis_20261121.md`.
- Prove: `results/infra/pose_temponet_temporal_diagnosis_20261121/`.

Richiesta: messaggio di GPT incollato da Simone il 21/11. È una review scientifica, non un rilascio: una sola review mirata, con
gli eventuali blocchi in testa. Nessuna implementazione e nessun lancio.

## Esito

**Nessun blocco. La diagnosi regge; D0 e D1 vanno fatti. Quattro condizioni vanno scritte nella proposta prima del pacchetto
operativo.** In ordine di importanza:

1. **La regola di lettura di D1 oggi non è decidibile** (sezione 5). Lo stesso esito può cadere in tre righe diverse. Va fissata
   prima dei fit, con le letture a esposizione fissa dichiarate in anticipo.
2. **D0 e D1 non dicono se un rimedio dal lato della testa esiste** (sezione 6). Il successo di D1 è compatibile sia con «basta
   una testa diversa» sia con «a questo campione nessuna testa allenata così passa». Serve un riferimento sotto rumore, oppure va
   scritto che la variante sarà scelta senza quella prova.
3. **La variante non si sceglie e si conferma sullo stesso mondo sintetico** (sezione 7). Serve un mondo nuovo con semi dichiarati
   prima, un solo tentativo, e poi la linea torna a Simone.
4. **D0 e D1 partono solo dopo una decisione registrata di Simone** (sezione 8), con ID, costi e sotto-budget. I tetti di A1
   coprono la matrice prevista, non prove nuove.

Sul rimedio: concordo con la priorità a capacità e regolarizzazione rispetto a più epoche e al warm start, che i numeri escludono.
Non concordo che D1 basti a motivarla, e «più stretta» non è ovviamente il rimedio.

Che cosa non copre: nessun codice nuovo è stato rivisto, perché non c'è. Il pacchetto operativo avrà la sua review.

## 1. Metodo

- Letti la diagnosi, i tre README delle prove, il protocollo v3 §4.4, le voci di `docs/decisioni.md` e il codice del controllo.
- Rieseguiti i tre script di GPT su un checkout isolato di `e7670758`: uscite identiche a quelle committate.
- Ricalcolati da me, dai JSON grezzi dei controlli, il gate, le curve per learning rate, la selezione e i riferimenti in forma
  chiusa.
- Due revisori indipendenti senza accesso al cluster, uno sulla lettura statistica e uno sul disegno, e quattro controlli
  incrociati sui loro punti principali. Esiti, note, script e log in
  `claude_review_pose_temponet_temporal_diagnosis_20261121_evidence/`.
- I revisori hanno anche eseguito una **simulazione giocattolo** su CPU: latenti sintetici, ma generatore, testa, loss e ricetta
  veri; un fold, un seme, LR 0,003, orizzonti troncati. Mostra che un andamento è possibile, non che valga sui latenti veri.
- Nessun job, nessun accesso al cluster.

## 2. Che cosa ho verificato della diagnosi

| Riscontro di GPT | Esito | Nota |
| --- | --- | --- |
| Recupero 24,17% / 18,88% / 21,71% contro il 50% | confermato | ricalcolato da `dynamic_sse` e `dynamic_target_ss`, non dai valori arrotondati |
| Formula del gate | conforme | `(E_cleanLS − E_TCN) / (E_cleanLS − E_oracle)`, per fold e pooled, come nel protocollo §4.4 |
| Miglior punto di tutta la ricerca: 21,88% | confermato | è un punto della curva pooled (epoca 27, LR 0,003), non uno stato salvato; per fold esistono solo i valori del punto scelto |
| Cinque LR, 150-300 epoche, minimi interni | confermato | minimo MSE alle epoche 181, 64, 31, 31 e 93 per LR da 0,0001 a 0,01 |
| Proseguire peggiora | confermato | a 150 epoche del LR 0,003 il recupero pooled scende da +21,7% a −22,6% |
| La TCN impara parte del segnale | confermato | sul bersaglio pulito degli esclusi 0,1392 contro 0,1990 di L0 |
| Prova costruttiva di rappresentabilità | confermata | testa non modificata, LayerNorm nel percorso. Vale in valutazione: con il dropout attivo quei pesi non sono un punto stazionario della ricetta |
| Margine dell'oracolo circa 8% | confermato | 8,02% e 7,84% sugli esclusi outer |

## 3. Tre precisazioni alla diagnosi

### 3.1 Il riferimento è stimato sul pulito: una testa allenata sul rumoroso parte sotto zero

`E_cleanLS` è il lettore lineare stimato sul bersaglio **pulito** `y*` e valutato su `y`. Le teste sono allenate sul bersaglio
**rumoroso**. Sulla scala del gate, dove 0% è `cleanLS` e 100% è l'oracolo:

| Lettore, allenato su `y` | fold 0 | fold 1 | pooled |
| --- | ---: | ---: | ---: |
| TCN | +24,2% | +18,9% | +21,7% |
| L0 neurale | −13,6% | −10,8% | −12,3% |
| MLP | −12,2% | −10,7% | −11,5% |
| L0 in forma chiusa | −19,1% | −16,4% | −17,8% |
| L13 in forma chiusa | −85,6% | −69,5% | −78,1% |

- Alla pari, cioè partendo da L0 neurale invece che da `cleanLS`, la TCN chiude il 33,2% / 26,8% / 30,3% della distanza
  dall'oracolo. Per raggiungere il gate ne servirebbe il 55,5% pooled.
- L'asimmetria è reale ma non spiega da sola il fallimento: 24% e 30% sono due scale dello stesso esito.
- Il protocollo la dichiarava («possono essere più severi»), quindi il gate resta valido così com'è. La diagnosi però non riporta
  mai L0 e MLP su questa scala: vanno aggiunti.

### 3.2 Quanto ha imparato la TCN e quanto le costa il rumore

Il controllo lineare misura direttamente il prezzo del rumore per questa ricetta. Lì non c'è niente di temporale da imparare, e
la TCN è scelta quasi alla stessa esposizione (LR 0,003, epoca 28, contro l'epoca 31 del temporale). Finisce sopra l'oracolo di
una quantità pari al 19,7% e al 18,7% del margine temporale; la ridge del 12,5% e del 10,9%.

Lettura additiva, approssimata perché unisce due mondi con rumore simile ma non uguale (sigma 1,35 e 1,4 nel lineare, 1,3 nel
temporale):

| | fold 0 | fold 1 |
| --- | ---: | ---: |
| Margine temporale catturato, circa | 44% | 38% |
| Prezzo del rumore, circa | −20% | −19% |
| Recupero netto misurato | 24,2% | 18,9% |

Quindi all'epoca scelta mancano due cose insieme. La funzione è imparata per meno di metà, e il rumore toglie altri venti punti.
Anche senza prezzo del rumore il 50% non sarebbe raggiunto.

Allenare più a lungo non aiuta, perché la rete adatta il rumore più in fretta di quanto impari la funzione:

| LR | 0,0001 | 0,0003 | 0,001 | 0,003 | 0,01 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Miglior recupero pooled | +5,0% | +5,4% | +10,8% | +21,9% | −14,9% |
| Recupero a fine corsa | +0,6% | −5,9% | −39,8% | −22,6% | −15,8% |

A convergenza una testa di 72.904 parametri perde circa l'intero margine. Lo dicono tre cose: la caduta a −22,6% e a −39,8%; il
lettore lineare con lo stesso contesto di 13 patch, che stimato su `y` finisce 70-86 punti sotto il riferimento; e un conto sul
numero di parametri, che dà fra il 65% e il 181% del margine. Quest'ultimo è solo un ordine di grandezza e vale per la stima
portata fino in fondo, non per il checkpoint scelto.

Che cosa l'archivio **non** dice: quanta parte dell'errore al checkpoint scelto sia rumore e quanta funzione non imparata. La
tabella sopra è la stima migliore disponibile; D1 la misura.

### 3.3 Lo scarto fra fitting ed esclusi non indica un cattivo trasferimento fra utenti

La diagnosi lo chiama l'indizio principale e lo lega a «rumore e generalizzazione». I numeri archiviati separano già le due cose:

| Lettore | Variazione di `E_dyn` da fitting a esclusi, fold 0 / fold 1 |
| --- | ---: |
| Oracolo | −0,0009 / −0,0021 |
| `cleanLS`, stimato su `y*` | +0,0018 / +0,0003 |
| L0 in forma chiusa, stimato su `y` | +0,0156 / +0,0137 |
| TCN | +0,0348 / +0,0282 |
| L13 in forma chiusa, stimato su `y` | +0,0919 / +0,0822 |

Chi è stimato senza rumore si trasferisce agli altri 20 utenti senza perdita. Lo scarto della TCN sta fra quelli dei due lettori
lineari stimati su `y`: è della grandezza attesa dalla sola memorizzazione del rumore. Decide il dato che D0 aggiunge, cioè
l'errore pulito sul fitting.

## 4. D0: sì, è un job

Fattibile come lo descrive GPT, verificato sul codice e con un prototipo sulle fixture:
- esistono lo stato migliore e quello finale per tutti e cinque i LR e per i due fold, con i contatori dichiarati
  (`train.py:452-459`);
- `y`, `y*`, maschere, soglie e partizione si ricostruiscono bit per bit da `calibration.pt` con funzioni esistenti;
- la funzione di punteggio accetta già il bersaglio pulito su qualsiasi insieme di finestre.

Tutti gli input stanno solo su Leonardo: **D0 è un job Slurm**, con ID, tetto e decisione registrata. Non è una lettura dal Mac.

Da precisare nel pacchetto:
1. **Il numero che decide.** È il recupero pulito del checkpoint scelto sul fitting. Vale al più 86,1% / 69,3% se non c'è rumore
   memorizzato; se il meccanismo è quello del giocattolo vale circa 51% / 44%. Va dichiarato prima.
2. **Riferimenti diretti.** L'oracolo di fitting calcolato dagli array, non ricostruito dal margine di L13. Se a 150 epoche
   l'`E_dyn` sul fitting scende sotto l'oracolo di fitting, è la prova diretta del rumore memorizzato.
3. **Riproduzione.** Per fold solo per lo stato scelto; per l'epoca 150 esiste soltanto il valore pooled. Vanno dichiarati
   tolleranza e dispositivo, e l'uguaglianza del bundle ricostruito con quello archiviato prima di qualsiasi punteggio.
4. **Informazione gratuita.** I 20 stati salvati (cinque LR, migliore e finale, due fold) si rileggono in pochi secondi ciascuno, e
   ogni stato finale contiene la loss di training per epoca del suo fold. Sono le curve per fold che la diagnosi dà per mancanti.
5. **Autocontrollo.** Lo stesso punteggio pulito sul fitting per L0 e MLP, dove le forme chiuse danno la risposta attesa.

## 5. D1: sì, ma la regola di lettura va fissata prima

D1 è meccanicamente corretto: stessa inizializzazione, stessi batch e stesse maschere di dropout sono ottenibili esattamente, e
scegliere l'epoca sulla MSE rumorosa degli esclusi segue l'errore pulito entro 1-2 punti di recupero.

**Il problema è la tabella degli esiti.** Così com'è:
- un minimo al bordo delle 150 epoche è «inconclusivo» secondo il testo e «ottimizzazione da indagare» secondo la riga 3, ma la
  stessa corsa può soddisfare la riga 1 con un recupero sopra il 50%;
- «molto inferiore», «scarso» e «curve ancora in discesa» non hanno una quantità;
- nessuna lettura all'esposizione 31 è dichiarata: la ripartizione fra funzione non imparata e rumore verrebbe scelta dopo aver
  visto le curve.

Con un bersaglio senza rumore la curva tende a un altopiano e il minimo cade vicino al bordo per puro caso: nel giocattolo, a
150 epoche, in un seme su due. Lo stesso esito scientifico sarebbe «inconclusivo» per un seme e non per l'altro.

Regola proposta, tre frasi da scrivere prima dei fit:
1. **Si legge nell'ordine del protocollo.** Recupero su `y` al punto scelto, nei tre pannelli: se è almeno il 50% ovunque è la
   riga 1, e un minimo nelle epoche 146-150 si annota soltanto. Sotto il 50% con il minimo in 146-150 è «non risolto entro 150
   epoche», senza estensione. Sotto il 50% senza bordo si guarda il fitting: recupero sul fitting almeno al 50% vuol dire che la
   funzione non si trasferisce fra utenti, sotto il 50% vuol dire che non è imparata nemmeno lì.
2. **Letture a esposizione fissa, dichiarate prima:** 31, 150 e il punto scelto; per fold e pooled; fitting ed esclusi; contro
   `y` e `y*`. La differenza fra D1 e A1 all'esposizione 31 è l'effetto del rumore di training a parità di esposizione.
3. **«Curve ancora in discesa» esce dai criteri**, oppure va definita.

Altre tre cose da scrivere:
- Il recupero di D1 è alla pari con `cleanLS`, perché entrambi sono stimati sul pulito: non è confrontabile punto per punto con
  il 24,17% / 18,88% di A1.
- Un residuo sul fitting può venire da dropout e weight decay, non da un difetto di ottimizzazione. Accanto all'errore in
  valutazione va pubblicata la loss in modalità di training.
- Un fallimento al LR 0,003, scelto sotto rumore, resta ambiguo. Un successo vale a maggior ragione.

D1 chiede un driver nuovo: bersaglio di training e bersaglio di selezione oggi sono lo stesso campo, e la ricerca congelata non
sa fare un solo LR a orizzonte fisso. Va scritto fuori dall'albero congelato, importandolo senza modifiche. Prima dei fit puliti
serve un **cancello di parità**: lo stesso driver, con bersaglio `y`, riproduce i primi punti pubblicati del LR 0,003.

**Previsione, scritta prima di vedere D1.** Mi aspetto che passi: nel giocattolo la stessa testa allenata sul pulito arriva
all'80% all'epoca 28 e a circa il 91% a 150. Vale come esclusione: se non passasse, il lavoro cambierebbe direzione.

## 6. Il riferimento che manca

D0 e D1 separano bene tre ipotesi: difetto di ottimizzazione, mancato trasferimento fra utenti, effetto del rumore. Non separano
le due forme dell'ultima:

- a questo campione il rumore si può battere con una testa diversa o meglio regolarizzata;
- a questo campione nessuna testa allenata con questa ricetta arriva al 50%.

Le due danno lo stesso esito di D0 e D1. La diagnosi lo riconosce quando scrive che la calibrazione non garantisce che metà del
margine sia stimabile, ma la tabella non ha una riga per questo caso. E nessun braccio allena su `y` qualcosa di diverso dalla
candidata già fallita.

Nel giocattolo le due cose convivono davvero:

| Testa, giocattolo | Bersaglio | Recupero sugli esclusi |
| --- | --- | ---: |
| Candidata | `y*` | 80% all'epoca 28, in salita |
| Candidata | `y` | 31%, con 92% sul fitting (dato vero: 24% / 19%, con 86% / 69% sul fitting) |
| Famiglia del generatore, larga 8, inizio casuale | `y` | 27% |
| Famiglia del generatore, larga 8, inizio casuale | `y*` | 44% |
| Famiglia del generatore dai pesi veri | `y` | 67% e stabile fino all'epoca 94 con il dropout della ricetta; 80% e in calo all'epoca 28 senza dropout |
| Candidata dai pesi veri, canali in più liberi | `y` | da 100% a 55% in 23 epoche, in calo |

Letture, con i limiti di un giocattolo:
- il successo sul pulito convive con ogni testa a inizio casuale ferma al 31% o sotto sul rumoroso;
- dai pesi veri la famiglia giusta resta sopra il 50% fin dove è stata seguita: la soglia non risulta irraggiungibile in senso
  statistico, ma le corse sono troncate;
- una testa stretta quanto il generatore **non** è il rimedio ovvio: da un inizio casuale impara male anche sul pulito, in parte
  per il dropout sugli strati larghi 8.

**Riferimento proposto, nello stesso pacchetto di D1.** La famiglia del generatore inizializzata ai pesi veri e allenata su `y`,
LR 0,003, 150 epoche, per fold.
- Non è una candidata e non è ammissibile: misura che cosa resta di una funzione giusta dopo l'allenamento sul rumoroso.
- Si legge alle esposizioni 31 e 150, mai all'epoca scelta sugli esclusi, perché all'esposizione 0 il recupero è 100% per
  costruzione.
- Va fatta senza dropout, oppure in coppia con lo stesso braccio su `y*`, leggendo la sola parte dovuta al rumore. Con il dropout
  della ricetta circa 14 dei punti persi nel giocattolo vengono dal dropout e non dal rumore.
- Non serve una classe nuova: l'incorporamento a zero già scritto da GPT nella testa di produzione dà gradiente esattamente
  nullo ai canali in più, quindi è un allenamento largo 8 mascherato.
- Regola dichiarata prima: se la parte dovuta al rumore lascia meno del 50% a 150 epoche in un pannello, a decisione registrata
  va la specifica del controllo, non la testa. Se lascia almeno il 50%, un rimedio dal lato della testa può esistere.

Se GPT non vuole il braccio, la riga 1 va riscritta: il successo di D1 giustifica la priorità di capacità e regolarizzazione
rispetto a epoche e warm start, non la premessa che un rimedio dal lato della testa esista. E va scritto che cosa succede se la
variante fallisce.

## 7. Il rimedio e come sceglierlo

Concordo con GPT su che cosa scartare:
- **Warm start dalla ridge: no.** Il ramo affine non è il collo di bottiglia. L0 neurale, a −13,6% e −10,8%, fa già meglio della
  forma chiusa stimata su `y`, a −19,1% e −16,4%.
- **Più epoche: no.** I minimi sono interni e dopo il minimo il recupero crolla.

Sul resto:
- **La regolarizzazione esplicita oggi è quasi assente.** Con AdamW, weight decay 0,01 e LR 0,003 i pesi si contraggono meno del
  3% negli 899 update fino all'epoca 31. Resta il dropout 0,1. Di fatto regolarizza solo l'arresto anticipato.
- **«Più parsimoniosa» non vuol dire per forza più stretta.** Il giocattolo dice che restringere rallenta l'apprendimento della
  funzione. Frenare l'adattamento al rumore di una testa larga è l'altra strada.
- **Il campione resta una leva.** Le 1.907 finestre di train sono 10 finestre da 4 secondi per registrazione. Più finestre per
  utente ridurrebbero il prezzo del rumore per tutte le teste, anche nel confronto reale. Costa una nuova estrazione e cambia gli
  input: non è il primo passo.

**La condizione di metodo.** Cambiare la candidata dopo un gate fallito e ripetere il gate è una modifica della specifica fatta
dopo aver visto questo mondo. I semi del generatore e del rumore sono fissi nel codice, i fold sono gli stessi, e le stesse metà
escluse hanno già scelto LR ed epoca in A1, sarebbero rilette da D0 e D1 e poi confermerebbero la variante. Va registrato, prima
che esistano i risultati di D1:
- la variante è valutata sul mondo originale, per confronto con A1, e su **un mondo nuovo** con semi dichiarati prima e la sua
  calibrazione;
- il 50% deve valere sul mondo nuovo; nessuna riestrazione di mondi;
- un solo tentativo; se fallisce la linea torna a Simone.

Cambiare la costante del seme tocca il runtime congelato: chiede la sua identità dei sorgenti e la sua review.

## 8. Conformità al protocollo e costi

- Lo stato registrato è `recipe_not_admitted`, con «arresto scientifico prescritto». Il protocollo chiede una decisione
  registrata per una nuova diagnosi e per ogni modifica della specifica, e vieta la ricerca finché si vince.
- La proposta non cambia soglia, selezione o controllo, e rimanda ID e budget al pacchetto. Va però scritto in chiaro che D0 e D1
  partono solo dopo una decisione di Simone registrata in `docs/decisioni.md`, con ID, tempi, risorse, costo in GPU-ore e in ore
  locali, sotto-budget addebitato e residuo. La convergenza di questa review non è quella decisione.
- I tetti di A1 coprono la matrice prevista: 3,681111 GPU-ore usate su 56,10. Il fattore 1,5 non si estende a prove nuove.
- Costo: secondo il revisore del disegno 0,205 GPU-ore è circa 1,6 volte troppo basso per il disegno descritto, perché D1
  raddoppia le valutazioni e una valutazione costa più di un'epoca. È una stima, non una misura. Per D0, cancello di parità, D1 e il riferimento l'ordine è di 1-2 GPU-ore, da misurare.

## 9. Che cosa ho corretto della mia prima lettura

Nel messaggio provvisorio a Simone avevo scritto due cose che i revisori hanno ridimensionato:
- «il rumore costa fra il 65% e il 181% del margine»: vale a convergenza, non al checkpoint scelto, dove il prezzo misurato è
  circa 20 punti e la parte più grande è la funzione non ancora imparata;
- «una testa piccola come il generatore misura che cosa è recuperabile»: da un inizio casuale nel giocattolo fallisce per
  l'ottimizzazione, non per il rumore. Il riferimento utile parte dai pesi veri.

## 10. Prompt per GPT

```text
Review Claude della diagnosi del controllo temporale TCN (commit e7670758de9d0b6e25a9ad880ec3e1f7d573f169, base d3e41d06):
nessun blocco. La diagnosi regge, D0 e D1 vanno fatti. Quattro condizioni da scrivere nella proposta prima del pacchetto operativo.
Review: docs/reviews/claude_review_pose_temponet_temporal_diagnosis_20261121.md, ramo claude/temponet-db8-20261109 (commit nel
messaggio di Simone), con la cartella _evidence (due revisori indipendenti, quattro controlli incrociati, script e log).

Verificato: i tre script rieseguiti danno uscite identiche; gate, curve per LR, selezione e riferimenti ricalcolati dai JSON
grezzi; formula conforme al protocollo 4.4. Tutti i riscontri confermati.

Condizioni, in ordine:
1. Regola di lettura di D1 decidibile, scritta prima dei fit. Oggi un minimo al bordo è «inconclusivo» nel testo, «ottimizzazione»
   nella riga 3 e può stare nella riga 1. Proposta: si legge nell'ordine del protocollo (recupero su y >= 50% nei tre pannelli =
   riga 1, il bordo 146-150 si annota; sotto il 50% con bordo = non risolto entro 150; sotto senza bordo si guarda il fitting);
   letture a esposizione fissa 31, 150 e punto scelto, per fold e pooled, fitting ed esclusi, contro y e y*; «curve ancora in
   discesa» fuori dai criteri o definita.
2. D0 e D1 non separano «un rimedio dal lato della testa esiste» da «a questo campione nessuna testa allenata così arriva al
   50%». Riferimento proposto nello stesso pacchetto: famiglia del generatore dai pesi veri, allenata su y, LR 0,003, 150 epoche,
   per fold; letta a 31 e 150, mai all'epoca scelta; senza dropout o in coppia con lo stesso braccio su y*. Si ottiene dal tuo
   incorporamento a zero nella testa di produzione (gradiente nullo sui canali in più). Sotto il 50% a 150 va a decisione la
   specifica del controllo, non la testa. Se non lo vuoi, riscrivi la riga 1 e di' che cosa succede se la variante fallisce.
3. La variante non si conferma sullo stesso mondo: mondo originale per confronto, più un mondo nuovo con semi dichiarati prima e
   la sua calibrazione; il 50% vale sul mondo nuovo; un solo tentativo, poi torna a Simone.
4. D0 e D1 partono solo dopo una decisione di Simone registrata, con ID, costi e sotto-budget. I tetti di A1 coprono la matrice
   prevista, non prove nuove. D0 è un job: i suoi input stanno solo su Leonardo.

Tre precisazioni da aggiungere alla diagnosi:
- sulla scala del gate L0 neurale sta a -13,6 / -10,8 / -12,3%, MLP a -12,2 / -10,7 / -11,5%, L0 in forma chiusa su y a -19,1 /
  -16,4 / -17,8%; alla pari contro L0 neurale la TCN chiude 33,2 / 26,8 / 30,3%, ne servirebbe il 55,5% pooled;
- nel controllo lineare la stessa ricetta, scelta all'epoca 28, finisce il 19,7 / 18,7% del margine temporale sopra l'oracolo:
  lettura additiva, catturato circa 44 / 38%, prezzo del rumore circa 20 / 19%;
- lo scarto fitting/esclusi (+0,0348 / +0,0282) sta fra quelli di L0 e L13 in forma chiusa su y (+0,0156 e +0,0919 nel fold 0),
  mentre cleanLS si trasferisce senza perdita (+0,0018 / +0,0003): è compatibile con la sola memorizzazione del rumore.

Per i pacchetti: D0 con riproduzione per fold del solo stato scelto e pooled per l'epoca 150, tolleranza e dispositivo dichiarati,
oracolo di fitting calcolato dagli array, i 20 stati salvati e le loss per epoca; D1 con driver esterno all'albero congelato e
cancello di parità sul bersaglio y. Costo dell'ordine di 1-2 GPU-ore, da misurare: 0,205 è circa 1,6 volte troppo basso.

Sul rimedio: d'accordo che warm start e più epoche sono esclusi dai numeri. «Più stretta» non è ovviamente il rimedio: in una
simulazione giocattolo la testa larga 8 da inizio casuale arriva al 27% su y e al 44% su y*. Nessuna implementazione o lancio da
parte mia.
```
