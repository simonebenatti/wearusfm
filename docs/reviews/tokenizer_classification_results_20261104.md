# Tokenizer: confronto finale delle sonde e lettura complessiva — 04/11/2026

**La GELU sulla finestra da 100 ms dà un beneficio misurabile su EPN-612; la
sola finestra lunga non mostra un vantaggio in classificazione.** T100-G arriva
a 82,58% di balanced accuracy su EPN, contro 81,10% di T100-L e 81,60% di T25-L.
Il vantaggio della GELU è +1,48 punti percentuali, IC95% [0,83; 2,12]. Non emerge
un miglioramento generale su tutti i dataset. Sulla posa la riduzione combinata
dell'errore dinamico resta 1,53%, sotto il criterio pratico del 2%, e la varianza
predetta raggiunge soltanto il 16,88% di quella dei bersagli.

La matrice autorizzata è conclusa: tre training a 19.618, dodici estrazioni,
lettori posa a 5k/19618 e sonde convergenti P1/P2 a 19618. Nessun job residuo
di questa campagna, nessun nuovo esperimento lanciato. Monitor automatico spento.

## Provenienza e controlli

Confronto finale **63519077 COMPLETED 0:0**, 20 secondi su 2 CPU, terminato il
04/11 alle 15:34:55 CET. Report originale:
`results/infra/tokenizer_evaluation_20261103/probe_cpu_19618/final/comparison.json`.
SHA256 `20e899455889b37fe12747acbbba688f204bdcb29bf480a270f8fc748a1128e4`.
Runtime congelato `b315dadfedc09c522deeb56af829d34ebc04f4b4`, piano file SHA256
`dac83e0d3206da9f0c9c9cd3290cc1e7430fa40d496ba73bb01915cc0b573010`.

Quindici shard, sei aggregati e confronto finale conclusi senza errore. Hash
dei sei aggregati coerenti fra remoto, copie locali e riferimenti nel report;
identità di dose, teacher, cache, piano, ricetta e audit verificate. Le predizioni
OOF sono state appaiate per righe, etichette native, soggetti e fold. I file OOF
e i segnali restano fuori Git.

**420/420 fit convergenti**, nessun warning di convergenza; massimo 5.916 su
12.000 iterazioni. Restano 36 warning di scoring DB3, per classi predette assenti
nei target di alcuni fold, già registrati: non sono fallimenti del solver. La
convenzione congelata calcola la macro-recall sulle classi presenti nei target.
Nessuna replica bootstrap finale perde classi; nessun draw viene scartato.

La [review indipendente](tokenizer_classification_critic_20261104.md) ricalcola
punti, contrasti, IC, hash dei draw e criteri di non inferiorità dai conteggi per
soggetto, senza importare funzioni numeriche del progetto. Il verificatore e il
suo esito sono conservati accanto al report.

## Classificazione: numeri completi

Teacher congelato, pool768 FP32 senza PCA; StandardScaler stimato solo sul train
di ciascun fold. Logistica L2/lbfgs bilanciata, griglia C comune, quattro fold per
soggetto, selezione sulla validation interna e rifit sul solo train. Tre modelli
con lo stesso seme, dati, dose e obiettivo; cambiano finestra e attivazione.

Tutti i valori della tabella sono **balanced accuracy in percentuale**. P1 è la
media pesata DB2/DB3/DB6 (8/13, 3/13, 2/13); P2 è la media EPN/UCI (1/2, 1/2).

| Dataset / aggregato | T25-L | T100-L | T100-G |
|---|---:|---:|---:|
| **EPN-612, primario classificazione** | **81,601** | **81,102** | **82,577** |
| UCI EMG, secondario | 77,438 | 76,342 | 78,457 |
| NinaPro DB2 | 18,559 | 18,679 | 18,224 |
| NinaPro DB3 | 3,152 | 2,833 | 3,380 |
| NinaPro DB6 | 34,816 | 34,136 | 33,748 |
| P1 | 17,505 | 17,400 | 17,187 |
| P2 | 79,520 | 78,722 | 80,517 |

| Contrasto | EPN, delta pp [IC95%] | P1, delta pp [IC95%] | P2, delta pp [IC95%] |
|---|---:|---:|---:|
| T100-L − T25-L: finestra | −0,499 [−1,074; +0,113] | −0,105 [−0,673; +0,430] | −0,798 [−2,430; +1,061] |
| T100-G − T100-L: GELU | **+1,476 [+0,828; +2,121]** | −0,214 [−0,729; +0,340] | +1,796 [−0,352; +3,644] |
| T100-G − T25-L: combinato, secondario | **+0,977 [+0,399; +1,546]** | −0,318 [−0,802; +0,185] | +0,998 [−0,986; +3,111] |

Gli IC di tutti i contrasti sui singoli NinaPro e su UCI includono zero. Per UCI,
GELU rispetto a T100-L è +2,116 pp [−2,071; +5,856]: direzione favorevole, ma
incertezza ampia. Il guadagno aggregato P2 non ha un IC interamente positivo.
I numeri molto bassi di DB3 restano visibili e non sono nascosti dall'aggregato.

## Regole congelate e interpretazione

EPN è il primario della classificazione, distinto dal primario posa. L'IC del
contrasto GELU−lineare a 100 ms è sopra zero: segnale favorevole nella ricetta
provata. Anche il confronto combinato con T25-L è positivo, ma resta secondario.
La sola finestra lunga ha stime puntuali peggiori su EPN/P2 e IC che includono
zero: non abbiamo dimostrato che passare da 25 a 100 ms basti a migliorare la
classificazione, né che 25 ms rendessero inutilizzabile il modello.

La regola di non inferiorità con margine 1 pp si applica **solo a P1/P2**:

| Contrasto | P1 | P2 |
|---|---|---|
| T100-L − T25-L | Sostenuta | Irrisolta |
| T100-G − T100-L | Sostenuta | Sostenuta |
| T100-G − T25-L | Sostenuta | Sostenuta, al margine |

L'ultimo limite inferiore P2 è −0,985938 pp, soltanto **0,014062 pp** sopra −1:
la regola congelata è formalmente soddisfatta, ma non è una rassicurazione
robusta sul trasferimento generale. Non inferiorità non significa miglioramento,
equivalenza o garanzia per tutti i dataset; non arrotondare quel limite a −1 e
cambiare retroattivamente il verdetto. `decision_valid=true` indica che i fit
consentono la lettura prevista, non l'adozione automatica del tokenizer.

EPN/UCI erano esclusi dal pretraining. I soggetti P1 appartengono invece al
sottoinsieme di pretraining; i fold escludono soggetti dall'addestramento della
sonda, non dalla storia dell'encoder. Gli IC sono bootstrap appaiati per soggetto
(2.000 draw, seed 20261102), condizionati a un solo pretraining e alle sonde
fittate, su dati di sviluppo già osservati. Nessuna correzione per confronti
multipli o stima della variabilità tra semi. EPN è a 200 Hz, ma un effetto su
questo dataset non isola causalmente la frequenza di campionamento dagli altri
fattori che lo distinguono dai dataset a 2 kHz.

Il disegno contiene T25-L, T100-L e T100-G; manca T25-G. Il contrasto della
GELU è quindi identificato a 100 ms, senza stabilire se richieda la finestra
lunga o quale sia l'interazione completa fra finestra e attivazione.

## Lettura insieme alla posa

La [posa teacher a 19.618](tokenizer_pose_19618_results_20261104.md) misura una
ridge sulla sequenza media dei 64 slot, su 15 utenti esclusi dal pretraining
ma sullo stesso compito di posa supervisionato. Non è un trasferimento a una
regressione nuova, né la testa student del pannello di train.

| Misura posa, dinamiche | T25-L | T100-L | T100-G |
|---|---:|---:|---:|
| Errore dinamico normalizzato, meno è meglio | 0,851975 | 0,843363 | 0,838933 |
| Correlazione centrata | 0,384740 | 0,395775 | 0,401443 |
| Varianza predetta / target | 14,7801% | 15,6104% | 16,8815% |

Finestra: −1,01% relativo; GELU: ulteriore −0,53%; combinato: −1,53%. IC appaiati
favorevoli, ma tutte le riduzioni puntuali sotto il 2% fissato prima dei risultati.
La GELU peggiora leggermente alcuni secondari statici rispetto a T100-L.
**T100-G è un candidato promettente su EPN e moderatamente migliore nella lettura
dinamica; non è una soluzione al problema generale del FM o del movimento.**

## Seguito proposto, senza nuovi lanci

Il confronto a tre bracci è chiuso. Conservare T100-G come riferimento empirico
promettente su EPN e T100-L come controllo necessario per il prossimo candidato
già discusso: risposte log-quadratiche affiancate alle lineari sulla finestra
100 ms, inizializzazione non addestrata condivisa e nuovi pesi a zero. Il suo
confronto primario rimane con T100-L; quello con il migliore dei lineari è
secondario. Non spostare la base su GELU dopo aver visto il risultato.

L'eventuale preparazione comprende verifica delle scale sul train anche a
200 Hz, maschere, epsilon e preventivo separato training/estrazioni/CPU. Non è
ancora la potenza di banda esplicita, né una nuova loss. Conferma a dose completa
e più semi, fine-tuning con riferimento da zero e confronto controllato 30M/300M
restano possibilità distinte da ordinare con Simone, non conseguenze automatiche
di un IC positivo. Non occorre esaurire tutte le varianti a 30M per considerare
la scala, ma questi risultati non garantiscono che 300M migliori.

## Consuntivo della fase sonde

15 shard: **41,822222 core-ore**; 6 aggregati: **0,038333**; confronto finale:
**0,011111**. Totale **41,871667 core-ore su un tetto 101**, zero GPU aggiuntive.
Risorse e accounting in `probe_cpu_19618/final/resources.json`; il checkout
comune endpoint è contabilizzato separatamente una sola volta. Nessun retry dei
fit o cambiamento del protocollo. Training ed estrazioni hanno consuntivi GPU
separati, già pubblicati; questo numero non è il costo totale della campagna.
