# T100-LQ: risultato scientifico finale — 07/11/2026

**Campagna completata fino a 19.618 passi. Il ramo log-quadratico migliora EPN-612 rispetto al lineare, ma non migliora la dinamica della posa e non dimostra un vantaggio generale nella classificazione.** Su EPN il risultato è vicino a quello della GELU; non emerge una ragione scientifica per sostituire subito la ricetta GELU già in consolidamento o per prolungare LQ a 49.046 soltanto nella speranza di recuperare la posa.

Questa conclusione riguarda la linea T100-LQ: non dichiara concluse le altre campagne del progetto. Tutti i sedici job CPU scientifici e i due checkout sono COMPLETED 0:0; gli ultimi confronti posa, 63786157 e 63786158, sono finiti alle 16:36:15 e 16:36:51 CET. Nessun nuovo job o monitor è stato avviato per questa lettura.

## Domanda e confronto

La modifica affianca alle 64 risposte lineari del banco le 64 risposte `log(0,001 + x²)`, con proiezione aggiuntiva inizializzata a zero. Finestra 100 ms, passo 25 ms, overlap 75 ms, anti-aliasing invariato. Si conservano le risposte con segno: non è una sostituzione con GELU, né una misura esplicita dell'energia di banda. I filtri e le proiezioni sono apprendibili; logaritmo ed epsilon restano fissi.

Il confronto primario congelato è **T100-LQ contro T100-L**, separatamente per errore dinamico di posa ed EPN-612. T25-L e T100-G sono secondari dichiarati. Un training nella taglia 30M, seme 0, stessa inizializzazione condivisa non addestrata, dati, loss e schedule: ancore 0,2, posa 1, JEPA spento; WSD totale 49.046, endpoint osservati 5.000 e 19.618. Il training a 19.618 non include ancora il decay finale dello schedule.

Valutazione dal teacher congelato: ridge istantanea sulla posa alle due dosi; logistica convergente sulle cinque classificazioni a 19.618. Non è fine-tuning completo del FM. Le cache, i supporti, le ricette dei lettori e i riferimenti storici sono stati verificati prima dei confronti. Tutti i 140 fit nuovi di classificazione convergono senza warning o raggiungimento del limite; anche i 420 fit storici usati sono risolti. Le otto celle di posa hanno selezione ridge risolta.

## Quadro dell'endpoint a 19.618

| Misura | T25-L | T100-L, primario | T100-G, secondario | T100-LQ |
| --- | ---: | ---: | ---: | ---: |
| Errore dinamico E_dyn, minore è meglio | 0,851975 | 0,843363 | 0,838933 | **0,844088** |
| Correlazione temporale centrata, posa | 0,3847 | 0,3958 | 0,4014 | **0,3949** |
| Varianza predetta / target, posa | 14,78% | 15,61% | 16,88% | **16,19%** |
| MAE posa sulle finestre dinamiche, rad | 0,304224 | 0,300370 | 0,300068 | **0,300370** |
| EPN-612, balanced accuracy | 81,601% | 81,102% | 82,577% | **82,633%** |
| P1, aggregato congelato | 17,505% | 17,400% | 17,187% | **17,133%** |
| P2, aggregato congelato | 79,520% | 78,722% | 80,517% | **79,757%** |

E_dyn è la somma degli errori quadratici delle traiettorie centrate nella finestra divisa per la somma della varianza dei bersagli sullo stesso supporto. Togliere il livello medio impedisce di confondere postura e inseguimento del movimento. I numeri della posa riguardano 15 utenti esclusi dal training, già impiegati come sviluppo, e 8.329 finestre dinamiche. Le varianze non sono ampiezze: il 16,19% di varianza corrisponde a circa il 40% in deviazione standard aggregata, non a un'ampiezza del 16%.

## Posa: nessun guadagno aggiuntivo dal log-quadratico

| Dose | T100-L | T100-LQ | Delta E_dyn LQ−L e IC95 appaiato | Riduzione relativa dell'errore |
| --- | ---: | ---: | --- | ---: |
| 5.000 | 0,910612 | 0,912850 | +0,002238 [−0,000242; +0,004880] | −0,246% |
| 19.618 | 0,843363 | 0,844088 | +0,000726 [−0,001190; +0,002724] | −0,086% |

Il candidato impara con la dose: E_dyn scende del 7,53% fra i due checkpoint. Anche il controllo scende del 7,39%. Questo andamento non identifica un beneficio della trasformazione aggiunta. A 19.618 migliorano rispetto a T100-L soltanto 6 utenti su 15; correlazione e MAE sono quasi identiche al controllo, e la variazione temporale resta attenuata. Non c'è evidenza di un recupero del movimento nascosto da una sola metrica aggregata.

La soglia pratica prestabilita richiedeva una riduzione relativa almeno del 2% e un IC del delta interamente sotto zero. Non è raggiunta in nessun contrasto dinamico LQ alle due dosi. **Non è una dimostrazione di equivalenza esatta**, ma manca il beneficio cercato, e l'intervallo osservato contro L è stretto rispetto alla riduzione pratica prevista.

| Controllo secondario | Delta E_dyn a 5.000 [IC95] | Delta E_dyn a 19.618 [IC95] | Lettura |
| --- | --- | --- | --- |
| T25-L | −0,008607 [−0,011759; −0,005579] | −0,007887 [−0,010246; −0,005484] | LQ migliore di circa 0,93% relativo, sotto il 2%; non isola il ramo log dalla finestra. |
| T100-G | +0,013035 [+0,007875; +0,018795] | +0,005156 [+0,003040; +0,007297] | LQ peggiore; a 19.618 circa +0,61% relativo. |

La scomposizione dell'MSE standardizzato sulle dinamiche conferma il quadro. A 19.618 LQ ha errore del livello medio 0,554861 e delle variazioni 0,443437, totale 0,998299; il lineare 0,552159 + 0,443056 = 0,995215. A 5.000 il lieve vantaggio LQ nell'MSE totale veniva dal livello medio, mentre l'errore dinamico era maggiore. Non trasformiamo quel vantaggio iniziale di postura in evidenza sul movimento.

I gruppi non primari sono pubblicati nella verifica indipendente. Sulle finestre statiche a 5.000 LQ riduce l'errore centrato del 2,44% rispetto a L, delta −0,026132 [−0,032046; −0,020293], con entrambi i rapporti sopra 1. Questo vantaggio non persiste a 19.618: delta +0,004429 [+0,000934; +0,008211]. Non viene usato per promuovere il candidato dopo il risultato del primario dinamico.

Precisazione dopo la review Claude 975f6a3: nello stesso gruppo statico a 5.000 anche T25-L e T100-G migliorano su T100-L del 2,25% e 2,33%. È T100-L a essere più sfavorevole in quel gruppo/dose; la bandiera numerica del report non identifica un segnale specifico LQ. Originali immutati, interpretazione esplicitata.

## Classificazione: beneficio specifico su EPN, nessuna superiorità sulla GELU

| Misura | LQ−T100-L, pp [IC95] | LQ−T25-L, pp [IC95] | LQ−T100-G, pp [IC95] |
| --- | --- | --- | --- |
| EPN-612, primario | **+1,531 [+1,013; +2,015]** | +1,032 [+0,512; +1,569] | +0,055 [−0,512; +0,603] |
| P1 | −0,267 [−0,845; +0,312] | −0,372 [−0,912; +0,209] | −0,054 [−0,656; +0,503] |
| P2 | +1,036 [−0,397; +2,495] | +0,238 [−0,894; +1,604] | −0,760 [−2,584; +1,183] |

Il beneficio EPN rispetto al controllo primario è presente e utile come risultato di sviluppo. La vicinanza a GELU non prova equivalenza fra i metodi, ma **non sostiene un vantaggio del log-quadratico sulla GELU**. Due trasformazioni non lineari differenti migliorano EPN rispetto al lineare a 100 ms: è un indizio a favore dell'utilità di strutturare diversamente il token per questo compito. Non dimostra che il meccanismo siano energie di banda o la sola frequenza di campionamento a 200 Hz.

Per P1 e P2 gli IC dei miglioramenti sul lineare includono zero. La non inferiorità entro il margine prestabilito di 1 pp è sostenuta per entrambi contro T100-L e T25-L. Contro T100-G è sostenuta per P1, irrisolta per P2. Non inferiorità non significa miglioramento, né garantisce ogni singolo dataset.

| Dataset | T100-L | T100-G | T100-LQ | LQ−L, pp [IC95] |
| --- | ---: | ---: | ---: | --- |
| NinaPro DB2 | 18,679% | 18,224% | 18,631% | −0,049 [−0,818; +0,728] |
| NinaPro DB3 | 2,833% | 3,380% | 2,428% | −0,405 [−1,220; +0,363] |
| NinaPro DB6 | 34,136% | 33,748% | 33,199% | −0,937 [−2,730; +0,790] |
| EPN-612 | 81,102% | 82,577% | 82,633% | +1,531 [+1,013; +2,015] |
| UCI EMG | 76,342% | 78,457% | 76,882% | +0,540 [−2,297; +3,522] |

Il vantaggio EPN non si ripete in modo dimostrato negli altri quattro dataset. DB3 resta debole in valore assoluto e peggiora nei contrasti secondari con T25-L, −0,725 pp [−1,484; −0,123], e con GELU, −0,952 pp [−1,798; −0,225]. Tutti i contrasti per dataset, incluse le incertezze non mostrate nella tabella compatta, sono conservati nel report originale e nella verifica indipendente.

## Il ramo è stato appreso, ma questo non basta a renderlo migliore

Le norme medie delle colonne di W_log del teacher crescono fra 5.000 e 19.618: Fourier 0,1383→0,2678, spline 0,0655→0,1949, MLP 0,1069→0,3258. Il candidato non è rimasto nella configurazione iniziale W_log=0. Anche le colonne dei kernel MLP, oggetto del controllo preliminare su epsilon, hanno pesi appresi non nulli.

Sono descrittivi, non un criterio di successo: norme dei pesi senza la distribuzione degli ingressi non misurano il contributo all'uscita e non attribuiscono il risultato a una famiglia. Non cambiamo epsilon dopo aver visto i risultati.

L'intervento include 24.576 pesi aggiuntivi e cambia anche il clipping globale. Nel collaudo, non nell'intero training, entrambi i bracci erano clippati in 64/64 passi; il rapporto dei fattori LQ/L aveva mediana 0,995062 e la quota g_log²/g_LQ² mediana 1,6681%. A parità di funzione iniziale non avevamo promesso identità degli aggiornamenti condivisi. Il risultato misura la ricetta LQ completa, non l'effetto puro del logaritmo isolato dall'ottimizzazione.

## Valutazione e seguito proposto

1. **Chiudere questa prova a 19.618 come esito misto:** positivo su EPN rispetto al lineare, nullo per il beneficio dinamico aggiuntivo cercato, nessun miglioramento aggregato P1/P2 dimostrato. LQ non è una soluzione generale migliore dei candidati precedenti.
2. **Non dare priorità a LQ49k sulla base di questi risultati.** Il miglioramento a 49k di un altro braccio non dimostra che LQ acquisisca un vantaggio relativo. Se si vorrà misurare quell'interazione con la dose, servirà un confronto pari dose con T100-L; GELU49k è un confronto pratico secondario, non sostituisce il controllo primario. Questa lettura non avvia una continuazione.
3. **Mantenere distinta la linea di consolidamento GELU e fine-tuning già discussa.** Qui GELU resta competitiva su EPN e leggermente migliore sulla posa; questa consegna non verifica né aggiorna lo stato operativo delle altre linee.
4. **Nel fork esplorativo, il seguito concordato rimane una proposta contrastiva su una base fissata**, con positivi, negativi, augmentazioni e peso definiti e revisionati prima di addestrare. Non prendere LQ come nuova base per automatismo e non introdurre contemporaneamente una nuova loss dinamica o un altro tokenizer. La scelta della base va esplicitata nel protocollo, usando questa evidenza come sviluppo.

Un modulo energetico strutturato resta una domanda diversa. Questo esperimento non esclude LEAF o altre stime di potenza, e non dimostra che slot interi, lettori più ricchi o fine-tuning non possano usare informazione diversa. Le sonde misurano l'accessibilità nella vista congelata provata, non tutto ciò che un FM adattato potrebbe imparare.

## Validità, risorse e riproducibilità

IC95 da 2.000 bootstrap appaiati per utente, seme 20261102; stessi supporti e target. Gli intervalli sono condizionati ai checkpoint e alle sonde già fittate: non includono variabilità fra pretraining indipendenti, né un nuovo fit a ogni ricampionamento. Un solo seme, dati di sviluppo riutilizzati, molti confronti e GELU secondaria aggiunta dopo i suoi risultati limitano la forza confermativa. Nessun verdetto di utilità generale del FM o di trasferimento a una regressione indipendente dalla posa supervisionata.

| Tranche | Consuntivo | Tetto / ambito |
| --- | --- | --- |
| Training, incluso il tentativo fallito prima degli update | 17,065556 GPU-ore; 136,524444 core-ore | 22 GPU-ore; 176 core-ore |
| Quattro estrazioni, due posa e P1/P2 | 1,739444 GPU-ore; 13,915556 core-ore | Accounting delle estrazioni; checkout esclusi |
| Letture CPU finali, 16 job più due checkout | 0 GPU-ore; 13,143333 core-ore | 51 core-ore |

Non è un totale dell'intera campagna: collaudo, diagnostica dell'inizializzazione, benchmark d'inferenza e altri checkout hanno consuntivi separati già pubblicati. Nessun costo è stato cancellato per rendere favorevole il confronto.

Evidenze: [archivio finale](../results/infra/logsquare_evaluation_20261107/final/), [posa, verifica indipendente](reviews/logsquare_pose_results_20261107.md), [classificazione, verifica indipendente](reviews/logsquare_classification_results_20261107.md), [review finale](reviews/logsquare_final_results_critic_20261107.md), [esecuzione CPU](logsquare_cpu_execution_20261107.md). Gli archivi gzip preservano esattamente i byte degli originali, verificati mediante SHA256 e dimensione; le predizioni OOF del candidato sono archiviate, quelle di posa e i coefficienti remoti sono verificati per hash senza scaricare le grandi cache dei segnali.
