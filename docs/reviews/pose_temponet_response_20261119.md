# Risposta a Claude ee118a25: TCN frozen, revisione del 19/11/2026

Base del diff: `3e007c841c97da5793ed4aac6cfceb41c99a00ee`. Stato **REVIEW_REQUIRED**.
Ultima review pertinente verificata: `ee118a25`; documento ed evidence invariati fino a `f9b215c4`.
Nessun nuovo esito scientifico, nessun job. Runtime preparato in un ramo separato: questa consegna
non ne dichiara la review. Simone ha approvato oggi il seguito fino agli invii dopo allineamento;
fonte e parole letterali nel §10 del protocollo e in proposal.authorization.

## BL-1 e BL-2: scelte scientifiche adottate

**BL-1 accolto**, senza usare WLS non regolarizzata come ancora dell'errore reale.
E_real ed E_cal vengono dalla ridge pesata, alpha su MSE dei due held inner, intercetta libera,
penalità sulla somma delle MSE per finestra. Griglia 10**[-2..4], al massimo tre decadi per lato;
bordo finale irrisolto invalida E_real o la singola coppia sintetica, senza fit neurali di recupero.
Nel lineare, preliminare ridge ≥90% del margine costante→oracolo; poi ogni rete ≥90% del margine
costante→ridge. È l'alternativa ammessa dalla review: nel controesempio 89,8/90,8 ≈98,9%, così il
controllo non richiede alla rete di battere il riferimento leggibile per essere ammessa. Le verifiche
sono per fold e pooled, MSE ed E_dyn. Margini invalidi → non diagnostico.

Nel temporale banda e margine usano WLS su y* pulito, valutata su y osservato; E_cal resta la ridge
su y. Griglia a=0,2:0,2:3 e sigma=0:0,05:6, 1.936 combinazioni totali per direzione outer, con
riuso a alpha fissato e accumulatori quadratici. La soglia sintetica usa y*, tutti gli errori usano y.
Banda inner e outer 6–10%, recupero TCN50%, tolleranza da E_real 0,05, spareggi espliciti.
Misuriamo anche quota di T accessibile a L0/L13 e unità attive prima dei fit; nessuna soglia nuova
su quei descrittivi o riscelta del seme. Il benchmark CPU proietta la griglia allargata, entro il
medesimo tetto: non si dichiara costo uguale al precedente.

Precisazione alla review: fittare WLS su y* elimina il rumore dal fit, ma non garantisce banda
identica inner/outer (rimangono popolazioni e approssimazione di T diverse). Il controllo outer resta.
Nessuna misura reale di E_real è stata usata per scegliere questa regola.

**BL-2: adottata l'opzione descrittiva.** L0 neurale e MLP nel controllo temporale non sono un veto.
MLP vicina o migliore non qualifica un vantaggio specifico del contesto; numeri e quote si riportano.
Il confronto reale TCN/MLP resta invariato nella famiglia. Questa scelta evita un gate indefinito.

## Scelte adottate per BL-3 e OB

- **BL-3 — accolta la tolleranza proposta.** Dopo l'unica estensione verso il basso, il LR minimo
  selezionato genera un allarme solo se `(MSE_vicino − MSE_minimo) / MSE_vicino > 0.005`.
  Le MSE sono pooled sui due held interni ai rispettivi checkpoint MSE selezionati; `vicino` è
  il successivo LR più alto provato. Pareggio o miglioramento ≤0,5%: bordo registrato, nessun
  allarme. Vale per tutte le teste, sui controlli e sui target reali. Il bordo superiore dopo
  estensione, l'ottimo nelle ultime cinque epoche dell'orizzonte massimo e la scelta al primo
  quarto di epoca restano allarmi. MSE_vicino=0 e MSE_minimo=0: miglioramento zero; valori non finiti: guasto tecnico. Nessuna divisione tacita. Luoghi: P §4.2, J regole dei bordi e controlli,
  DEC elenco degli allarmi.
- **OB-1 — accolta la regola per controllo.** Stati in ordine: non calibrato; non diagnostico
  deciso in forma chiusa; per ogni testa richiesta passa / ottimizzazione non risolta /
  ricetta non ammessa; eventuale clausola MLP soltanto dopo. Un allarme da solo non ferma un
  controllo che soddisfa il suo criterio numerico. Se non lo soddisfa: con allarme aperto
  «ottimizzazione non risolta entro il budget», senza allarme «ricetta non ammessa dal controllo
  positivo». Qualsiasi vero allarme ha precedenza sul compromesso S-2 della stessa testa.
- **OB-2 — accolta.** Scrivere sempre «compromesso S-2», mai allarme. S-2 si verifica se
  l'epoca scelta sulla MSE dista più di cinque epoche dal minimo pooled di E_dyn fra i
  checkpoint ammessi del LR selezionato, passo 0 escluso, e perde almeno il 2% di E_dyn.
  S-2 non sospende etichette né ammissibilità. L0: qualifica la lettura del primario;
  MLP/L13: qualifica la candidata e l'attribuzione all'accesso al tempo/non linearità;
  TCN-post/pre: compromesso della testa stessa, senza cambio del criterio di selezione.
- **OB-3 — adottata opzione (a), conserva il veto già prespecificato.** La candidata
  richiede che 1/4/5 siano `beneficio_pratico_risolto` e che né 2 né 3 siano
  `deterioramento_pratico`. Non sostituire il veto con la frase più forte «risolti migliori».
  Se MLP/L13 ha upper99<0 ma stima del gain >−2%, scrivere «candidata, con MLP/L13 risolti
  migliori ma con differenza stimata sotto il 2%», mostrando gain e IC99; mai «non peggiore».
  Con S-2 su MLP/L13: «candidata sotto la selezione MSE comune, con compromesso S-2 sul
  comparatore», senza affermare superiorità su quella famiglia di lettori. P §5, J candidate
  e duplicazione J:104, DEC e tabella §10 devono coincidere.
- **OB-4 — adottata conseguenza direzionale.** Un allarme sulla sola TCN sospende le
  conclusioni negative/nulle sul suo potenziale, non un beneficio osservato; il veto numerico
  di OB-3 continua comunque a escludere una candidatura positiva. Un allarme su L0/MLP/L13
  impedisce letture favorevoli alla TCN contro quella testa, inclusa l'assenza del veto.
  La candidata richiede L0/MLP/L13 ammessi, non TCN-post priva di allarmi. Numeri e curve si
  conservano; i fit reali e la raccolta proseguono con soli allarmi di ottimizzazione, senza
  altre estensioni. Un `technical_failure` continua invece a fermare la fase.
- **OB-5 — accolta.** Nella tabella §10 scrivere E_dyn≤0,7630564046643697 come condizione
  necessaria sulla stima, oltre a IC e comparatori; MAE/livello descrittivi senza margine di
  non inferiorità; nessuna affermazione sul beneficio del pretraining e nessuna conferma su
  utenti mai consultati. Richiamare OB-3/OB-4, copertura degli IC solo nominale e assenza del
  bootstrap a due livelli (conclusione negativa condizionata ai tre semi). Sistemare spazi.

BL-1 e BL-2 sono risolti dalle scelte esplicite della sezione precedente e da P §4.4/J positive_controls. Il generatore di partenza e la domanda scientifica restano quelli concordati; cambiano calibrazione e ammissione, dichiarate prima dei fit.

## Sei righe minori da correggere ora

| ID della review ee118a25 | Disposizione adottata e destinazione |
|---|---|
| architettura-2 | Accolta: in float64, per disegno/pattern/RHS, `norm(Xw.T @ (yw-Xw@beta)) <= 1e-6*norm(Xw,2)*norm(yw,2)`. Non usare norm(residuo) al denominatore. Aggiungere test CPU con RHS esattamente nel rango, oltre al caso noisy; P §4.3/J. |
| architettura-5 | Accolta: a pari pesi MLP e TCN hanno larghezze diverse; il confronto scambia larghezza per connessioni temporali e non isola soltanto il tempo; P §5. |
| criteri-2 / consegna-4 | Accolta nei due versi: L0 peggiore della ridge limita 1–3 a queste ricette; L0 migliore della ridge è effetto del rifit (possibile 1 no, 4/5 sì), non vittoria attribuibile alla TCN. Rimando di C-2 a P §5, per L13 peggiore di L0. |
| criteri-3 | Accolta: etichetta 3 «beneficio di almeno 2% non sostenuto dall'intervallo, riportando comunque direzione e ampiezza; non equivalenza», in P e J. |
| criteri-5 / consegna-2 | Accolta: J contiene `technical_failure` con NaN/Inf, OOM, TIMEOUT/fit incompleto, pin/supporti diversi e fallimento R-1; stop della fase, nessun retry/riduzione automatica, nessun contrasto scientifico pubblicato su input non validati. Stesso nome in P §4.5; R-1 fallito impedisce la fase scientifica dipendente. |
| criteri-6 / S-E | Accolta: confronto della pred cache160 ritagliata [6:154] con payload storico148 dopo cast di entrambe a float64, valori identici sui validi; non richiedere dtype originari uguali. Riportare 8.329 finestre dinamiche selezionate, 8.327 con supporto centrato, 166.540 coppie finestra-giunto, 20.119.920 elementi e 15 utenti, con i nomi effettivi del report. |

## Note della sezione 5: una disposizione ciascuna

| ID | Disposizione adottata |
|---|---|
| architettura-1 / S-F | Accolta facoltativa: definire una volta «ramo lineare (affine, con bias)» e usare un nome uniforme. |
| consegna-3 / controlli-8 | Accolta: «costante→oracolo» e unico stato «non calibrato» in tutti i testi. |
| criteri-8 / criteri-11 | Accolta: definire una volta gate comuni, ammissione dei comparatori e allarmi di ottimizzazione; regola candidate nel JSON come elenco di condizioni, non prosa compressa. |
| consegna-7 / consegna-9 | Accolta: ID esteso/versionato coerente in README, protocollo, proposta, manifest/output e voce di pin. Rifare hash dopo modifiche; rinominare `no_scientific_or_runtime_changes` perché si cambia il disegno, non il runtime. |
| ricetta-7 | Accolta: il weight decay non garantisce regolarizzazione sufficiente; con LR1e-2 e 9000 update `(1−lr*0.01)^9000≈0.4066`, quindi non dire semplicemente «quasi nullo» su tutta la griglia. Dichiarare dipendenza da LR/dose e ruolo dell'arresto; P §4.2. |
| ricetta-10 | Accolta: curve, decisione LR/epoca e manifest bloccati prima del rifit. |
| ricetta-5 | Accolta: S-2 su L0 e MLP ha la lettura descritta in OB-2; non diventa allarme. |
| ricetta-8 | Accolta come esplicitazione della scelta già fatta: anche un NaN a LR esplorativo ferma la fase; nessun candidato divergente scartato in silenzio. |
| controlli-5 / ricetta-11 | Accolta: nei controlli gli stessi held outer selezionano LR/epoca e verificano recupero, perciò i cancelli sono ottimisti; qualifica tecnica della ricetta, non stima di generalizzazione. |
| controlli-6 | Accolta: i sintetici non riproducono necessariamente la componente di livello dominante nei target veri; non provano il meccanismo di S-2. |
| controlli-7 | Accolta: prima dei fit neurali, sul fitting outer registrare quota di unità attive di T per strato e quota del margine recuperata da WLS L13 sul target temporale. Nessuna centratura aggiuntiva o riscelta del seed del teacher implicita. |
| controlli-10 | Accolta: un mondo calibrato per direzione outer; pooled = miscela pesata dagli accumulatori e dalle scale, non media dei rapporti per fold. |
| controlli-11 | Accolta: banda outer e verifica lineare di leggibilità in forma chiusa precedono ogni fit neurale. |
| criteri-7 | Accolta: delta MAE/livello = candidato meno riferimento; positivo significa peggiore. |
| criteri-10 | Accolta: ridge pre è solo un punto storico, senza inventare un IC appaiato dai soli aggregati. Ripristinare le tre frasi verificate nel rapporto dell'agente: stessi sorteggi bootstrap fra bracci/semi/riferimenti; pubblicazione di tutti e cinque i contrasti senza selezionare quello favorevole; una nuova metrica di tratto richiede proposta distinta. |
| architettura-6 / architettura-7 | Accolte: q_i conta elementi di crop6∩maschera; L13 e TCN non sono classi annidate in generale, e una vittoria non isola soltanto la non linearità. |
| costo-1 / V-costo-consegna-1 / S-H | Accolta: spiegare il gate a Simone. I tempi raw sono analoghi, non misure; nel caso tutto deterministico la proiezione completa potrebbe superare 20 GPUh e fermare i fit. Il benchmark effettivo decide, nessuna deroga automatica. |
| costo-2 / ricetta-6 | Accolta: al più 5 LR raggiungibili; distinguere massimo raggiungibile (125 traiettorie, 1.276.500 update, 33.440 eval con i vecchi conteggi di batch) dal limite prudente a 6 LR. Conservare il gate prudente senza spacciarlo per combinatoria esatta. Correggere `superseded`: 1.260.000 era il vecchio massimo raggiungibile, 1.458.000 il limite largo, non un errore dello stesso tipo. |
| costo-4 / S-G | Accolta: citare `47ca96e`, rinfresco effettivo 12/11 e scadenza dal 22/12; hash prima di ogni fase che legge questi file. Nessuna promessa sul rinfresco del 03/12. |
| costo-5 | Accolta nel principio: conteggio generato dalla griglia effettiva, `len(A)*len(Sigma)+len(Sigma_lineare)` per direzione outer. Le 30 combinazioni della review sono corrette per 5×5+5, NON per la griglia ampliata di BL-1. Aggiornare proiezione CPU se cambia la griglia. |
| costo-6 | Accolta: controllare corsia seriale prima di ogni fase CPU, non solo una volta nel preflight. |
| costo-7 | Accolta: liste esplicite degli utenti/fold nel manifest degli input prima dei fit; la proposta tiene la regola deterministica. Correggere la disposizione precedente che prometteva liste già in proposal.json. |
| consegna-5 | Accolta: completare le sette promesse note17/18/20/23/39/47/49 come nella tabella seguente; le altre cinque sono coperte da OB-5 e dalle minori. |

## Le dodici promesse: chiusura verificabile

| Promessa precedente | Dove deve risultare dopo questa revisione |
|---|---|
| C-3, perimetro dell'esito | §10 riga Esito positivo: portata limitata alla lettura delle cache, nessun valore generale del FM. |
| Nota33, soglia0,7630564 | §10 stessa riga, valore esatto/arrotondato dichiarato come condizione necessaria sulla stima. |
| R-2, arresto NaN/OOM nel JSON | Oggetto technical_failure con cause e conseguenze, oltre alla voce già esistente in DEC. |
| C-1, L0 deteriorata e lettura nei due versi | §5 descrittivi, aggiungere esplicitamente anche L0 migliore della ridge/effetto del rifit. |
| Nota12, larghezze diverse | §5 accanto al contrasto TCN/MLP. |
| Nota23, limite del weight decay | §4.2, con dipendenza da LR e dose; non formulazione universale «quasi nullo». |
| Nota47, I-1 | §7: un eventuale livello B richiede compatibilità con I-1 nel suo protocollo separato; niente implementazione B qui. |
| Nota17, curve/decisioni prima rifit | §4.2, lock del manifest prima del primo rifit. |
| Nota20, liste dei fold | Manifest input prima dei fit; correggere R per non promettere liste in J non ancora materializzate. |
| Nota49, ID negli output | §8 e J: ID identico in protocollo, manifest, ricevute e report futuri. |
| Nota39, corsia per fase | §9 e J: verifica prima di ogni fase CPU. |
| Nota18, destinazione semi | Correggere il link della disposizione al §3, dove i semi sono già specificati. |

## Verifiche che restano al codice e al benchmark, non motivi per un altro giro documentale

- Dropout con stato RNG CUDA dedicato o `bernoulli_(generator=...)`, ultimo layer di g a zero in pesi/bias.
- WLS con LN in float64 dai FP16, ranghi attesi di sanità 384 e 4980, test del residuo con RHS esatto.
- Sei input effettivamente aperti (incluso splits_v1.json) con pin e area; cache dai percorsi WORK della proposta.
- Semi canonici, rumore sulle 160 patch, pesi/maschere, soglie locali e continuazione dei due fold appaiati.
- Batch eval, numerica, placement e thread identici fra benchmark e fit; p95 effettivi per train/eval.
- Gate R-1 prima della lettura scientifica; bootstrap riproducibile, nessun cast implicito che allarghi il supporto.

L'autorizzazione umana del 19/11 supera il precedente «solo documenti» nel perimetro comunicato dal
coordinatore. La convergenza effettiva di Claude va comunque ottenuta; non scrivere REVIEW_CONVERGED
prima della sua rilettura. Operatore unico della nuova prova: 2-TCN head launcher, come nella consegna del coordinamento. Claude resta revisore; le operazioni storiche EPN15/lungo non cambiano.
