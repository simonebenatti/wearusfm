# Loss attuale e alternative motivate — 28/10/2026

**Aggiornamento 29/10:** B19618 ha migliorato anche la dinamica nei latenti mediati. La [valutazione aggiornata su PSD e loss](psd_loss_dopo_b19618_20261029.md) riprende la discussione dopo i risultati; le osservazioni sotto descrivono lo stato precedente a B19618.

Risposta alla domanda di Simone dopo l'integrazione `78268d6`. Analisi e proposta, nessuna modifica al training
o sottomissione. B19618 mantiene il protocollo congelato; le future sottomissioni spettano a semplicione.

**Decisione successiva di Simone, dopo la nota di Claude:** «forse possiamo guardarci ma un po piu avanti.
finirei quello che stiam facendo ora». Rinviati sia i confronti fra loss sia il controllo dei latenti non mediati.
La priorita' resta completare e leggere B19618 con le valutazioni gia' sottomesse. Nessuna nuova estrazione,
sonda o variante di training da questa discussione. Le alternative sotto rimangono ipotesi, non lavori avviati.

## Obiettivo effettivo

Il config del checkpoint B e il preflight B19618 confermano:

`L = 0.2 * (L_logRMS + L_bande + L_inviluppo + L_ms_fast + L_ms_slow) + 1.0 * L_posa`.

Sono errori quadratici mascherati. La posa usa 20 angoli standardizzati con statistiche train, media sugli elementi
validi dentro ogni finestra, poi media sulle finestre con posa valida. Non viene diluita dalle finestre degli altri
dataset senza posa. Le ancore sono statistiche dell'EMG, comprese le due scale a 50 e 500 ms. I bersagli spettrali
delle ancore restano attivi: sono distinti dai rami PSD aggiuntivi esclusi dal default. JEPA ha peso zero;
RVQ, keep e rami spettrali non contribuiscono. La ricetta A/R0 storica aveva anche JEPA con peso uno.

Fonti: `src/wearusfm/training/jepa.py` (`pose_terms`, composizione delle loss), `src/wearusfm/model/anchors.py`,
`results/infra/pilot_posa/b19618_20261028/preflight/preflight_receipt.json`, `docs/continuazione_b19618_20261028.md`.

## Perche' esplorare, e che cosa non e' dimostrato

B5000 ha un beneficio dinamico held-out del 7,252%, ma varianza predetta/target del 6,324%. E3 lineare migliora
il livello medio, non la dinamica. Questo motiva una diversa enfasi dell'obiettivo; non dimostra che la MSE sia
la causa. Su ciascuna finestra/giunto e sul medesimo supporto valido, MSE = errore della media al quadrato + MSE
degli scarti centrati: la loss attuale gia' penalizza il movimento. L'ottimo quadratico e' una media condizionata;
informazione insufficiente o ambiguita' temporale possono produrre traiettorie attenuate anche con ottimizzazione corretta.

I dieci campioni E4 mostrano gradiente di posa apprezzabile, ma sono locali al rank/micro-batch: non provano il
bilanciamento globale ottimale e non giustificano aumentare alla cieca il peso della posa. MAE in radianti usata
per E1 e MSE standardizzata usata nel training hanno inoltre pesi e sensibilita' differenti.

## Priorita' proposta

1. Leggere prima A/B/C della tranche B19618. Nessuna modifica in corsa o nuova campagna da questa discussione.
2. Se resta il limite dinamico, piccolo confronto controllato con ricetta invariata salvo la loss: MSE attuale
   contro MSE + lambda * MSE centrata per finestra/giunto. Cosi' il peso relativo sulle variazioni diventa 1+lambda,
   mantenendo il vincolo sulla posa assoluta. Centrare usa i target solo durante il training; inferenza invariata.
   Stessi supporti/maschere, almeno due istanti validi per giunto per il termine centrato, nessuna divisione per
   la minuscola varianza delle finestre quasi statiche. Una loss centrata da sola non vincolerebbe il livello medio.
3. Come alternativa successiva: errore fra incrementi predetti e reali a intervalli temporali prefissati.
   Confrontare il movimento corretto, non penalizzare il movimento in se'. Scala, maschere, intervalli validi
   e sensibilita' a rumore/ritardi vanno fissati sul train. La derivata immediata a 25 ms non e' automaticamente
   la scelta migliore. Correlazione da sola non fissa ampiezza/offset; sola regolarizzazione di smoothness
   rischia di aggravare l'attenuazione; non sono le prime prove consigliate.
4. Huber o MAE: controlli semplici se emerge un problema di errori estremi o di metrica, ma non rimedio specifico
   alla dinamica. Anche una previsione mediana puo' essere piatta. La loss cinematica sulle dita e' pertinente
   alla posa, ma piu' specifica del task e da tenere distinta dall'utilita' FM. Una loss contrastiva multimodale
   cambierebbe il disegno del pretraining e le coppie/negativi: direzione separata, non prima sostituzione.

Un confronto preliminare dei lettori su cache train congelate puo' costare poco e verificare se cambiare enfasi
aiuta il lettore; non verifica l'effetto sul backbone. Un esito negativo non esclude il training end-to-end.
Prima di un lancio servono protocollo, pesi/scale fissati sul train, confronto a pari dose e preventivo.
La val gia' consultata non diventa il banco di tuning. Misurare insieme errore assoluto, dinamica, finestre poco
dinamiche e trasferimento P1/P2: varianza piu' alta o loss nuova piu' bassa non bastano a dimostrare utilita'.
I vecchi R1/R2/R5 cambiavano target/aggregazione JEPA o frontend; non esauriscono queste loss supervisionate di posa.

## Riscontro in letteratura

- Salter et al., emg2pose (NeurIPS 2024), training setup: L1 degli angoli con peso 1 e distanza euclidea delle
  dita ottenuta mediante cinematica con peso 0,01. Dimostra una scelta diversa plausibile per il task; non isola
  un vantaggio della loss rispetto alla nostra architettura e al nostro protocollo.
  https://arxiv.org/html/2412.02725v1
- Hadidi et al. (preprint, marzo 2026), re-evaluation di position/velocity decoding: conserva MAE + distanza
  delle dita e attribuisce cambiamenti rilevanti anche a scala del decoder, ottimizzazione e training congiunto
  Tracking/Regression. Ulteriore motivo per non attribuire automaticamente la scarsa dinamica alla MSE.
  Il loro decoder autoregressivo e' diverso dalla nostra testa; non trasferire direttamente la correzione di scala.
  https://arxiv.org/html/2603.08212v1

Consultati i testi primari il 28/10/2026. Review indipendente del subagent concorde su formula, riduzioni, priorita'
e limiti; nessun test di codice necessario per questa nota. Nessuna alternativa dichiarata superiore senza prova.

## Nota sulla discussione con Claude, rinviata nel merito sperimentale

Convergenza sull'ordine: finire la tranche corrente e poi scegliere eventuali prove. E3 non dimostra un limite
informativo dell'intero encoder: legge una trasformazione della media dei K latenti, con una classe lineare,
un subset train e alpha selezionato dalla validazione interna, non un ottimo globale garantito. Anche un'eventuale
ridge negativa sui latenti non mediati non dimostrerebbe assenza di informazione non lineare o temporale.
La classificazione su finestre P1/P2 puo' dipendere anche da dinamica e coattivazione: la predominanza causale
della postura media non e' stabilita dai nostri risultati. Il costo del probe non mediato, compresa la parte
CPU/memoria e la regolarizzazione nell'alta dimensionalita', andra' preventivato prima di riprenderlo; 0,1 GPU-ora
non e' un preventivo verificato. Nessuno di questi punti cambia il protocollo o il lancio B19618 gia' avvenuto.
