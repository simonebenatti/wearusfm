# Review indipendente — classificazione tokenizer a 19.618 passi, 04/11/2026

**Esito: analisi verificata, nessun impedimento tecnico alla pubblicazione.** T100-G mostra un beneficio su EPN-612 rispetto a entrambi i lineari. La sola finestra lunga non dimostra un beneficio di classificazione. La dinamica della posa migliora poco e resta sotto la grandezza pratica prefissata. Non è una promozione generale del FM.

## Materiale e verifica

- Runtime congelato: `b315dadfedc09c522deeb56af829d34ebc04f4b4`.
- Piano sonde: SHA256 `dac83e0d3206da9f0c9c9cd3290cc1e7430fa40d496ba73bb01915cc0b573010`.
- [Rapporto finale](../../results/infra/tokenizer_evaluation_20261103/probe_cpu_19618/final/comparison.json): SHA256 `20e899455889b37fe12747acbbba688f204bdcb29bf480a270f8fc748a1128e4`.
- [Script indipendente](../../results/infra/tokenizer_evaluation_20261103/probe_cpu_19618/final/independent_critic_verify.py) e [esito riproducibile](../../results/infra/tokenizer_evaluation_20261103/probe_cpu_19618/final/independent_critic_verification.json).

Lo script usa soltanto JSON e NumPy, senza importare funzioni numeriche del progetto. Ricostruisce balanced accuracy e bootstrap dai conteggi per soggetto con matrici delle molteplicità di campionamento; verifica i draw SHA256, i 2.000 campioni condivisi fra bracci con seed 20261102, tutti i punti, IC 95%, contrasti, pesi P1/P2 e regole di non inferiorità. Differenza numerica massima rispetto al report: **8,9 × 10^-16**. Nessuna replica perde classi o è scartata in questi dati.

Verificati inoltre i 15 audit integrali locali contro gli hash nei rispettivi shard: **420 fit**, copertura di tutti i sei candidati C e del refit in ciascun fold, scelta C per score interno con pareggio al minore, identità del runtime/piano, pool768 FP32 e massimo 12.000 iterazioni. **420/420 convergenti**, massimo osservato **5.916 iterazioni**, zero warning di convergenza e zero eccezioni. I **36 record con warning di scoring** non sono warning del solver e restano conservati; non dichiarare «zero warning» senza qualifica.

Le verifiche OOF originali del comparatore includono parità esatta di righe, target, soggetti e fold; gli aggregati verificano le identità contro le cache congelate. Questa review ricalcola le statistiche dai conteggi e verifica gli audit, ma non scarica nuovamente le grandi matrici OOF. Le ricevute della raccolta e gli hash remoti restano l'evidenza del loro trasporto/integrità.

## Risultati di classificazione

Balanced accuracy percentuale; delta in punti percentuali. EPN è la domanda principale di classificazione già dichiarata, distinta dalla posa. UCI e i dataset NinaPro sono secondari; P1/P2 hanno i pesi congelati.

| Misura | T25-L | T100-L | T100-G |
|---|---:|---:|---:|
| EPN-612 | 81,601 | 81,102 | **82,577** |
| UCI | 77,438 | 76,342 | 78,457 |
| NinaPro DB2 | 18,559 | 18,679 | 18,224 |
| NinaPro DB3 | 3,152 | 2,833 | 3,380 |
| NinaPro DB6 | 34,816 | 34,136 | 33,748 |
| P1 | 17,505 | 17,400 | 17,187 |
| P2 | 79,520 | 78,722 | 80,517 |

| Contrasto | EPN delta [IC 95%] | P1 delta [IC 95%] | P2 delta [IC 95%] |
|---|---|---|---|
| T100-L − T25-L | −0,499 [−1,074; +0,113] | −0,105 [−0,673; +0,430] | −0,798 [−2,430; +1,061] |
| T100-G − T100-L | **+1,476 [+0,828; +2,121]** | −0,214 [−0,729; +0,340] | +1,796 [−0,352; +3,644] |
| T100-G − T25-L | **+0,977 [+0,399; +1,546]** | −0,318 [−0,802; +0,185] | +0,998 [−0,986; +3,111] |

Lettura conforme ai criteri:

1. **GELU aiuta EPN nella ricetta da 100 ms osservata.** Il contrasto principale G−L e quello pratico secondario G−T25 sono entrambi favorevoli, con IC sopra zero. È un risultato positivo specifico di questo task/dataset, non la prova causale della frequenza a 200 Hz.
2. **100 ms lineare da solo non migliora EPN:** la stima è negativa ma l'IC comprende zero. Non dimostra nemmeno danno certo o equivalenza. Gli IC di tutti i contrasti sui singoli NinaPro e UCI comprendono zero.
3. **P1/P2 conservati entro 1 pp per T100-G:** i limiti inferiori sono sopra −1 nei confronti con entrambi i controlli. Per G−T25, P2 passa però di soli **0,01406 pp**: limite esatto **−0,9859384761 pp**. Applicare la regola congelata, dichiarando che la conclusione è molto vicina al margine; non trasformarla in una conservazione robusta né aumentare i bootstrap dopo aver visto il risultato per cercare conferma.
4. **Il miglioramento complessivo di P2 resta irrisolto:** i suoi IC comprendono zero. Questo è compatibile con un beneficio EPN e con l'incertezza maggiore su UCI. Non inferiorità non significa miglioramento.
5. Per T100-L la non inferiorità P1 è sostenuta, quella P2 resta irrisolta. Non trasferire il margine di 1 pp ai singoli dataset, per i quali non è stata fissata una nuova regola.

## Rapporto con la posa

La [ridge della posa già conclusa](../../results/infra/tokenizer_evaluation_20261103/cpu_19618/final/report.json) resta un endpoint distinto. E_dyn sulle finestre dinamiche vale **0,851975 / 0,843363 / 0,838933** per T25-L / T100-L / T100-G. Riduzioni relative: finestra **1,011%**, GELU aggiuntiva **0,525%**, combinazione **1,531%**. Gli IC dei delta sono favorevoli, ma tutti gli effetti puntuali sono sotto il criterio pratico **2%**.

Con T100-G la correlazione centrata è circa **0,401** e la varianza predetta circa **16,88%** di quella dei target. Non chiamare risolto l'inseguimento del movimento e non usare il vecchio −7,3% del pannello di training come effetto validato tra tokenizer. Il beneficio EPN è comunque interpretabile perché era una domanda principale distinta, non selezionata dopo il risultato della posa.

## Limiti e conseguenze

- Un solo seme di pretraining, checkpoint a dose identica e sonde già fissate; IC condizionati a questi modelli/lettori, senza variabilità tra training e senza correzione per molteplicità. Validazione di sviluppo già riutilizzata, non test finale intatto.
- P1 usa soggetti del sottoinsieme pretraining, separati tra fit/val/held della sonda: non sono tutti utenti mai visti dall'encoder. EPN/UCI sono esclusi dal pretraining. I diversi contesti di generalizzazione vanno nominati.
- Il disegno non contiene T25-G: dimostra l'effetto dell'aggiunta di GELU alla ricetta T100-L, ma non se GELU richieda 100 ms né quale sia l'interazione completa finestra × attivazione.
- Le sonde lineari su pool768 misurano accessibilità con questo lettore, non tutta l'utilità possibile dopo fine-tuning. I punteggi P1 assoluti restano modesti e il beneficio EPN non basta a proclamare un FM generalmente efficace.
- **T100-G è un candidato sensato per la prossima verifica**, non un vincitore universale. Il risultato sostiene continuare il lavoro sulle trasformazioni del tokenizer; non prova che il ramo log-quadratico o le energie esplicite funzioneranno. Il confronto log-quadratico già discusso mantiene T100-L come controllo primario e il migliore dei lineari come confronto secondario. T100-G può restare un riferimento descrittivo aggiuntivo, senza sostituire il confronto secondario già concordato né cambiarne la ricetta dopo i punteggi. Nuova dose, scala o fine-tuning richiedono il rispettivo protocollo, non seguono automaticamente dall'etichetta `decision_valid`.

Nessuna modifica al codice scientifico, alle soglie o alle predizioni; nessun nuovo job o commit da parte del revisore.
