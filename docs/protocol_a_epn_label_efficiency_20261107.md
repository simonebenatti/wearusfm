# Protocollo A — efficienza dei dati etichettati su EPN

07/11/2026. **Preparazione affidata a GPT con la disposizione di Simone «A GPT». Stato: `awaiting_claude_design_review`.** Questo è un disegno proposto, non un piano eseguibile: scelta della dose e budget restano da approvare. Nessun lancio; Claude resta l'operatore della linea 2.

## 1. Domanda e perimetro

Il teacher T100-G 30M a 49.046 passi aiuta rispetto a scratch quando diminuiscono i soggetti di train, mantenendo uguale la dose di ottimizzazione? Il controllo congelato usa una sonda logistica sulle feature pool. Non si cambia il riferimento a 19.618 del primario di classificazione già firmato: scegliere qui il teacher 49.046 definisce una nuova domanda, non una promozione retroattiva.

Si propone EPN di sviluppo con gli stessi quattro fold di EPN15. Nessun nuovo pretraining, architettura, loss, seme di pretraining o test ufficiale. Il confronto con pochi soggetti è distinto da poche finestre per soggetto e dalla calibrazione di un singolo nuovo utente. Un seguito su altri dataset resta un protocollo distinto; DB3 non viene rimosso dai primari precedenti.

## 2. Popolazione, sottoinsiemi e budget di etichette

EPN: 314 soggetti, sei classi, 25 finestre per soggetto e classe, 47.100 finestre. Si conserva l'esclusione dei test ufficiali P2, senza ricostruire nuovi split.

| Fold | Soggetti train al 100% | Validation fissa | Held fisso | Aggiornamenti raccomandati per candidato |
|---|---:|---:|---:|---:|
| 0, 1 | 211 | 24 | 79 | 59.355 |
| 2, 3 | 212 | 24 | 78 | 59.625 |

| Frazione nominale del train | Soggetti train | Finestre train | Soggetti etichettati train + validation | Aggiornamenti nell'alternativa a 15 epoche |
|---|---:|---:|---:|---:|
| 5% | 11 | 1.650 | 35 | 3.105 |
| 10% | 22 | 3.300 | 46 | 6.195 |
| 25% | 53 | 7.950 | 77 | 14.910 |
| 50% | 106 | 15.900 | 130 | 29.820 |
| 100% | 211 / 212 | 31.650 / 31.800 | 235 / 236 | 59.355 / 59.625 |

Selezione annidata: ordinare i soli soggetti train originali per SHA256 della stringa UTF-8 `20261107|fold={fold}|subject={id_completo}`; pareggio per ID completo lessicografico. Prendere prefissi di lunghezza `ceil(frazione * N_train)`. Gli ID mantengono `trainingJSON/` o `testingJSON/`. Dopo la selezione si mantiene l'ordine originale delle righe. Il valore 20261107 identifica l'unica selezione proposta; seed del modello, testa e dataloader **0**, distinto e riavviato per ogni candidato. Stessi soggetti fra FT, scratch, frozen e candidati LR; nessun warm start tra frazioni.

Il 5% significa 11 soggetti train **più 24 validation**: circa il 15% del budget totale di etichette del pieno, non il 5%. Pubblicare entrambi i conteggi. La validation esterna fissa è più grande del train al primo punto. È una curva simulata su dati di sviluppo già esaminati, non una misura prospettica su un dataset intatto.

Manifest da produrre prima dei fit: popolazione originale, train selezionato/escluso, val/held, righe originali, classi, hash e frazioni realizzate. I soggetti esclusi non entrano in gradienti, vocabolario, pesi di classe o scaler. Si conserva la calibrazione non supervisionata storica delle prime dieci finestre di ciascun soggetto, anche val/held; non si pretende assenza di calibrazione del nuovo soggetto.

## 3. Modelli, dose e selezione

**Raccomandazione da discutere: aggiornamenti costanti entro fold**, pari alle 15 epoche del train completo nella tabella. Ripetere solo il train selezionato fino a quel numero. Batch 8, AdamW con weight decay 0,01, clipping 1, BF16, stessa testa LN + Linear e preprocessing EPN15. Warmup 100 update, coseno sull'intero orizzonte del fold. A ogni passaggio del subset conservare lo shuffle storico e l’ultimo batch parziale, senza `drop_last` e senza riavviare optimizer, schedule o RNG. Registrare esempi elaborati: uguali update non garantiscono identiche esposizioni.

**Quindici letture della validation**, ai passi `j * U_fold / 15`, j=1…15, indipendenti dalle piccole epoche reali. Validare ogni epoca darebbe circa 287 occasioni al 5% contro 15 al 100%. Tutti i candidati completano la dose; checkpoint con migliore balanced accuracy di validation, pareggio al primo passo. Nessuna estensione di dose o early stopping guidati dai risultati. Le curve train/validation descrivono l'apprendimento, non certificano un optimum.

| Regime | Candidato 0: LR encoder / testa | Candidato 1: LR encoder / testa |
|---|---|---|
| FT dal teacher 49.046 | 1e−5 / 1e−3 | 1e−4 / 1e−2 |
| Scratch | 1e−4 / 1e−3 | 1e−3 / 1e−2 |

Sono coppie legate, non una griglia cartesiana. Stessa griglia della testa e due occasioni di tuning per regime; il contrasto è fra pipeline ottimizzate su due candidati, non a identico LR dell'encoder né fra ottimi della griglia. Nessuna estensione automatica se vince un estremo. Selezione del candidato sulla sola validation, pareggio al candidato 0. Held predetto una volta per nuova cella, dopo la chiusura dei due candidati; il candidato 0 non produce held autonomamente.

**Alternativa meno costosa: 15 epoche per frazione.** FT e scratch restano appaiati, ma le piccole frazioni hanno meno aggiornamenti. Misura una ricetta con meno dati e meno ottimizzazione; non attribuire l'intero gap al risparmio di etichette. Stesse 15 letture validation, una per epoca, coseno sul rispettivo orizzonte e warmup 100. Va scelta prima di nuovi risultati, non come ripiego dopo un esito.

## 4. Controllo congelato e riuso

Frozen: teacher 49.046, **pool 768**, StandardScaler addestrato solo sul train selezionato, logistica L2/lbfgs, class_weight balanced, `C={0.001,0.01,0.1,1,10,100}`, `max_iter=12000`, `tol=1e-4`, otto thread. Selezione C sulla stessa validation, pareggio al C minore; refit solo sul train selezionato, mai train+validation. Tutti i candidati devono convergere; una non-convergenza blocca la cella e non permette di scartare il C problematico. È un controllo pratico forte, con lettore e tuning diversi dal trainer GPU, non un'ablazione del solo sblocco encoder.

La cache 49.046 esiste, è stata preservata su WORK e non richiede estrazione GPU. I percorsi e SHA256 di teacher e cache sono in [proposal.json](../results/infra/epn_label_efficiency_20261107/proposal.json). La mappatura SCRATCH→WORK conserva provenienza e hash originali, senza riscrivere i piani congelati.

Riuso condizionato, da dimostrare prima del piano eseguibile:

- **Scratch 100%:** otto candidati EPN15 e quattro OOF selezionate. Parità di input, ruoli, ordine, calibrazione, modello casuale, RNG, testa, pesi di classe, optimizer, LR, batch, schedule, update, 15 letture validation e tie-break. Il teacher nominato non fornisce pesi a scratch, ma la configurazione deve essere identica. Il nuovo loop al 100% deve riprodurre le operazioni storiche: uguale conteggio di update da solo non basta. Un bridge cita ricevute e hash originali senza rinominarli.
- **Frozen 100%:** quattro fold delle OOF LONG 49.046/pool, soltanto con la stessa ricetta, input e selezione. Alle frazioni inferiori scaler e logistica vanno rifittati sul train ridotto.

Se il riuso non è dimostrato, la preparazione del piano eseguibile si ferma: niente otto fit scratch aggiuntivi impliciti o altro tuning. Il nuovo conteggio e preventivo passano dalla decisione di Simone.

## 5. Contrasti e regola di lettura proposta

**Un primario:** vantaggio medio FT−scratch integrato sulla curva a train ridotto, con n=(11,22,53,106), x=ln(n), S balanced accuracy in punti percentuali:

`L_r = sum_i [(S_r(n_i)+S_r(n_(i+1)))/2] * [ln(n_(i+1))-ln(n_i)] / ln(106/11)`

`Delta_L = L_FT49k - L_scratch`.

Il 100% è un riferimento secondario. L'area normalizzata è in punti percentuali, non AUROC né percentuale di etichette risparmiate. Scala, integrazione e primario sono proposti prima dei risultati nuovi; mostrare sempre tutti i punti, perché un'area positiva può nascondere svantaggi locali.

Lettura a due lati: IC interamente positivo favorisce FT, interamente negativo scratch, comprendente zero irrisolto. **Soglia pratica proposta nuova: 1 pp**, non ereditata da V-R1. Beneficio pratico se stima ≥1 pp e IC positivo; deterioramento pratico se stima ≤−1 pp e IC negativo. Effetti risolti sotto soglia si riportano come piccoli, non nulli. Queste etichette non affermano che l'intero IC superi la soglia pratica.

Secondari descrittivi: FT−scratch a ciascuna delle cinque frazioni; frozen e FT−frozen; FT49k 100%−FT19k 100% storico, solo se rigorosamente comparabile. Non c'è un FT19k a train ridotto: la curva non identifica l'effetto della continuazione 19k→49k. Niente interpolazione selettiva per annunciare una percentuale esatta di etichette risparmiate, né sostituzione del primario con la frazione migliore.

Bootstrap: **2.000 repliche appaiate per soggetto held, seed 20261102**, stessi sorteggi in tutte le frazioni/regimi e confronti storici pertinenti; IC percentile al 95%. In ogni replica calcolare balanced accuracy pooled delle OOF e poi area e contrasti. Pairing esatto di row/target/subject/fold. Non usare la media dei quattro fold come nuova metrica. Gli IC condizionano su modelli, split e unico seme di subset/training: non coprono la variabilità del piccolo train scelto né quella tra pretraining. Secondari senza correzione per molteplicità, dichiarati descrittivi. Held di sviluppo già consultato; nessuna conclusione generale di FM o conferma su test ufficiale.

## 6. Matrice e stime preliminari, non budget

| Componente | Celle nuove | Candidati per cella | Fit nuovi | Riuso condizionato |
|---|---:|---:|---:|---|
| FT 49.046, cinque frazioni × quattro fold | 20 GPU | 2 LR | 40 GPU | nessuno |
| Scratch, quattro frazioni ridotte × quattro fold | 16 GPU | 2 LR | 32 GPU | 8 candidati / 4 celle al 100% |
| Frozen, quattro frazioni ridotte × quattro fold | 16 CPU | 6 C | 96 selezioni + 16 refit CPU | 4 celle al 100% |

72 fit GPU nuovi; 20 celle logiche frozen, di cui quattro riusabili. Non sono 72+16 allocazioni già definite: benchmark, confezionamento CPU e aggregazione restano da dimensionare.

Dal consuntivo EPN15: FT 10,644 GPUh per candidato, scratch 10,640. La media 8,21 includeva il frozen più veloce e non va usata per questi due regimi. [Stima riproducibile](../results/infra/epn_label_efficiency_20261107/resource_estimate.json):

| Politica | Soli 72 fit GPU, stima | Cosa resta fuori |
|---|---:|---|
| Update costanti, proposta raccomandata | **766,22 GPUh** | benchmark, frozen CPU, aggregazione, checkout e margini |
| 15 epoche per frazione, alternativa | **239,79 GPUh** | stesse esclusioni; extrapolazione lineare più incerta sui subset piccoli |

I costi sono proiezioni, non tetti prenotati né consumi osservati della nuova ricetta. Il preprocessing e le 15 validation non scalano con il train: soprattutto la seconda stima può sottovalutarli. I valori illustrativi con un solo LR nel JSON dimezzano aritmeticamente i conti, ma non sono un braccio proposto: eliminare un candidato cambierebbe il tuning e richiederebbe definire il riuso coerente.

## 7. Consegna successiva, controlli e arresti

Prima di qualsiasi invio: review di Claude del disegno; decisione di Simone sulla politica e relativo budget; implementazione e test; pin dei manifest e dei riusi; benchmark train-only; preventivo completo e review del codice. Nessun benchmark di questa proposta è stato lanciato. Il disegno non eredita il tetto EPN15 o il residuo della valutazione lunga.

Il benchmark deve separare preparazione, update e validation, con warmup CUDA escluso dal tempo, misurare anche il subset minimo e prevedere tutti i candidati/frazioni. Il gate globale deve ammettere tutti i fit prima del primo invio; se una cella supera tempo/memoria/costo, niente matrice parziale. CPU frozen con contesto numerico e convergenza controllati. Quote, retention e copie dei risultati si scrivono nel futuro piano operativo, non si assumono quelle delle campagne chiuse.

Divergenze numeriche: conservare causa e traccia, nessuna predizione da checkpoint non finito; il candidato è non selezionabile. Se l'altro completa, si può selezionarlo, dichiarando che la ricerca non ha due candidati validi. Se nessuno completa la cella non è valutabile. OOM, TIMEOUT e guasti sono fallimenti operativi distinti e fermano la fase; nessun retry o aumento di limite automatico. Nessun primario su matrice incompleta: si conservano gli artefatti disponibili senza cambiare fold, frazioni o supporto.

Test necessari: annidamento e identità dei ruoli; indipendenza dalle righe train escluse; parità al 100% e RNG; 15 opportunità validation e dose rispettate; selezione solo validation; held una volta; bridge che rifiuti ogni differenza sostanziale; bootstrap appaiato e area controllati con un calcolo indipendente. Il runtime non è ancora implementato. Non modificare retroattivamente EPN15 o LONG; nuovo controller e namespace.

[Nota implementativa](reviews/protocol_a_implementation_20261107.md), [conteggi verificati](../results/infra/epn_label_efficiency_20261107/population_counts.json), [proposta strutturata](../results/infra/epn_label_efficiency_20261107/proposal.json).
