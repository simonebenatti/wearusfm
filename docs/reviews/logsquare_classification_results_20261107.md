# T100-LQ: verifica indipendente della classificazione — 07/11/2026

**Esito: analisi confermata.** Il ramo log-quadratico migliora EPN-612 rispetto ai due lineari. Non mostra un vantaggio rispetto a GELU su EPN, né un miglioramento aggregato dimostrato su P1/P2. Il risultato è specifico del compito e del checkpoint a 19.618, con un solo seme di pretraining.

## Evidenza e controlli

Report del confronto 63775886, `COMPLETED 0:0`, codice congelato `61f0021753fedf170eb1a7d40fde3fa40a638861`. SHA-256 del report originale: `3517fc272ca5a258953d23d71e9b3aa894527755cbbcd153b93864b9503e7bfc`. Gli originali sono archiviati in `results/infra/logsquare_evaluation_20261107/final/`; hash e ricevute sono accanto a ciascun report.

Lo [script indipendente](../../results/infra/logsquare_evaluation_20261107/final/analyze_classification_independent.py) non importa i moduli di analisi o scoring di produzione e non esegue fit. Ricostruisce dalle due OOF del candidato i conteggi per soggetto e classe con `bincount`, controlla le predizioni held-out dei quattro fold e ricalcola tutti i punti e gli IC usando una matrice di molteplicità dei soggetti. I conteggi storici coincidono con il precedente confronto tokenizer. I risultati ricalcolati coincidono entro 3×10⁻¹² punti percentuali; [sintesi JSON](../../results/infra/logsquare_evaluation_20261107/final/classification_independent_analysis.json).

Verificati gli audit di **560 fit: 140 nuovi LQ e 420 storici**, tutti convergenti secondo il criterio configurato; nessun warning di convergenza, eccezione o raggiungimento del limite di 12.000 iterazioni. Il massimo LQ è 4.150; lo storico massimo è 5.916. Griglia C, selezione sul validation fold, refit, identità dei fold e separazione fra train/val/held sono coerenti. Non si tratta di 560 nuovi fit. Le OOF storiche non sono state scaricate nuovamente in questo controllo: si riusano conteggi, audit e ricevute già archiviate, mentre il confronto sul cluster ha ricontrollato gli hash delle OOF originali.

Bootstrap appaiato per soggetto: 2.000 ricampionamenti, seed 20261102; un generatore attraversa i dataset in ordine alfabetico come congelato. P1 pesa DB2/DB3/DB6 rispettivamente 8/13, 3/13, 2/13; P2 pesa EPN/UCI 1/2 ciascuno. Si riportano balanced accuracy in percentuale e differenze in punti percentuali.

## Punti osservati

| Dataset / aggregato | Gruppi-soggetto | T25-L | T100-L | T100-G | T100-LQ |
|---|---:|---:|---:|---:|---:|
| epn612 | 314 | 81,601 | 81,102 | 82,577 | 82,633 |
| ninapro_db2 | 32 | 18,559 | 18,679 | 18,224 | 18,631 |
| ninapro_db3 | 8 | 3,152 | 2,833 | 3,380 | 2,428 |
| ninapro_db6 | 8 | 34,816 | 34,136 | 33,748 | 33,199 |
| uci_emg | 19 | 77,438 | 76,342 | 78,457 | 76,882 |
| p1 | — | 17,505 | 17,400 | 17,187 | 17,133 |
| p2 | — | 79,520 | 78,722 | 80,517 | 79,757 |

## Tutti i contrasti predefiniti

| Dataset / aggregato | LQ − T100-L, primario | LQ − T25-L, secondario | LQ − T100-G, secondario |
|---|---:|---:|---:|
| epn612 | +1,531 [+1,013; +2,015] | +1,032 [+0,512; +1,569] | +0,055 [-0,512; +0,603] |
| ninapro_db2 | -0,049 [-0,818; +0,728] | +0,071 [-0,522; +0,642] | +0,407 [-0,378; +1,091] |
| ninapro_db3 | -0,405 [-1,220; +0,363] | -0,725 [-1,484; -0,123] | -0,952 [-1,798; -0,225] |
| ninapro_db6 | -0,937 [-2,730; +0,790] | -1,617 [-4,234; +1,156] | -0,549 [-2,586; +1,362] |
| uci_emg | +0,540 [-2,297; +3,522] | -0,556 [-2,693; +2,068] | -1,575 [-5,309; +2,400] |
| p1 | -0,267 [-0,845; +0,312] | -0,372 [-0,912; +0,209] | -0,054 [-0,656; +0,503] |
| p2 | +1,036 [-0,397; +2,495] | +0,238 [-0,894; +1,604] | -0,760 [-2,584; +1,183] |

EPN è l’endpoint principale della classificazione; gli altri singoli dataset sono secondari. Il confronto primario rimane T100-L: non si cambia riferimento dopo aver letto GELU. Il confronto G, aggiunto dopo i suoi risultati precedenti, resta esplorativo.

## Lettura scientifica

1. **EPN: beneficio specifico credibile entro questo esperimento.** LQ raggiunge 82,633% contro 81,102% del lineare 100 ms: +1,531 pp, IC95 [1,013; 2,015]. Anche rispetto a T25-L il contrasto è positivo (+1,032 pp). La rappresentazione congelata rende più leggibili i gesti di questo dataset. Il confronto +0,055 pp con GELU ha IC [−0,512; 0,603]: non dimostra superiorità, ma neppure equivalenza fra i due tokenizzatori.

2. **P1/P2: nessun miglioramento aggregato dimostrato.** P1−L è −0,267 pp [−0,845; 0,312]; P2−L è +1,036 pp [−0,397; 2,495]. Il margine di non inferiorità di 1 pp è sostenuto per entrambi contro L e T25-L; ciò esclude una perdita maggiore di quel margine nel senso del criterio congelato, non prova beneficio o identità. Contro G la NI è sostenuta per P1, irrisolta per P2 (IC [−2,584; 1,183]).

3. **Non nascondere i risultati sfavorevoli.** Su DB3 LQ è sotto G di 0,952 pp e sotto T25-L di 0,725 pp, con IC secondari sotto zero. La balanced accuracy resta bassa in tutti i bracci; DB3 ha 8 gruppi-soggetto e 49 classi. DB2/DB6/UCI non mostrano un vantaggio LQ dimostrato. EPN spiega gran parte del segnale favorevole di P2; l’incertezza UCI rende l’aggregato meno risolutivo. Non è evidenza di un vantaggio generale su tutti i segnali a 200 Hz.

4. **Implicazione per il seguito.** Il ramo è una seconda trasformazione che migliora EPN rispetto al lineare; non dà, da questa sola classificazione, una ragione per sostituire GELU o preferire automaticamente LQ per il prossimo training. Un’estensione a 49.046 risponderebbe a una domanda di dose ancora aperta, non consoliderebbe un vantaggio già dimostrato su GELU. La decisione deve includere la posa e confronti a pari dose; nessun nuovo job è disposto da questa review.

Un seme, checkpoint fissi e sonde già adattate: gli IC descrivono l’incertezza del ricampionamento dei soggetti condizionata a questi modelli, non la variabilità fra training indipendenti. Gli stessi dati di sviluppo sono stati letti ripetutamente, senza correzione per confronti multipli; non è un test finale intatto. L’intervento comprende il ramo e la sua traiettoria di ottimizzazione, incluso l’effetto dichiarato sul clipping. Il risultato non attribuisce il vantaggio alla sola frequenza di campionamento, non prova un’energia di banda e non risolve l’utilità generale del FM o il trasferimento a una regressione diversa dalla posa.

**Nessun fit, training, invio, deploy o monitoraggio aggiunto.** Questa consegna comprende solo ricalcolo offline e lettura dei risultati completi.
