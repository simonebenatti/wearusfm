# Proposta H5: schedule del learning rate per TEMPONet su NinaPro DB8

Claude, 10/11/2026. Alla domanda «Preparo una proposta per H5 (dose e schedule del training)?» Simone ha risposto «Sì, preparala».
**E' una proposta: nessun job lanciato.** Soglie e regole diventano vincolanti solo dopo l'approvazione, con una voce in `docs/decisioni.md`
committata prima del primo job.
**Stato, 10/11 sera:** approvata («Approvo»), passo A eseguito (job 64062336), nessuna variante sopra la soglia, passo B non fatto. Vedi «Esito».

## Punto di partenza

Diagnosi conclusa (`docs/temponet_db8_diagnosis_plan.md`, «Esiti»):
- Con V13 (bersaglio standardizzato e padding simmetrico) e il modello `full` (sessioni 1 e 2 unite) la MAE e' **7,585 gradi** contro 6,89 ± 0,30:
  **non replicata**.
- La catena e' coerente col paper: la SVR torna a 7,37 contro 7,28.
- Restano 0,70 gradi propri del training della TCN. Con la stessa catena, l'MLP su Waveform Length fa 6,82.

## Cosa e' ancora libero: perche' la proposta riguarda lo schedule e non la dose

[Z21] fissa il training: loss L1, Adam con learning rate **«initial»** 1e-4, batch 64, 19 epoche fp32, poi un'epoca di
quantization-aware training (QAT) int8; ingresso grezzo. Quindi:

- **Dose (epoche, batch).** E' dichiarata: aumentarla non replicherebbe piu' il paper. L'unica leva di dose non dichiarata, il passo delle
  finestre di training, e' gia' quasi satura: da 8 a 4 campioni la MAE di S1 scende solo di 0,07.
- **Grandezza del learning rate e normalizzazione dell'ingresso.** Contraddirebbero affermazioni esplicite del paper (1e-4 iniziale,
  ingresso grezzo).
- **Restano due dettagli non dichiarati:** lo schedule del learning rate («initial» dice che il valore cambia) e il learning rate dell'epoca
  finale di QAT.

## Cosa dicono le curve del passo 3

Le curve vengono dai 72 modelli di fold di V13 nel job 64010753. Sono letture descrittive.

- La validazione grezza media scende a ~10,15 gradi entro l'epoca 5 e poi resta piatta fino alla 19 (fra 10,10 e 10,19), mentre la loss di
  training continua a scendere (da 0,404 a 0,348 nel bersaglio standardizzato). Con V0 la validazione scendeva ancora, lentamente, fino
  all'epoca 19: da 10,70 all'epoca 10 a 10,55.
- Il singolo modello oscilla da un'epoca all'altra. Fra le epoche 10 e 19 la mediana dello scarto fra massimo e minimo e' 0,51 gradi, e
  l'ultima epoca sta 0,25 sopra la migliore.
- Interpretazione: con learning rate costante i pesi restano a oscillare sul plateau, e un annealing di solito ne toglie una parte.

**Rumore della misura di scelta.** V13 col seme 0 e' stata allenata due volte, con validazione 9,590 (screening) e 9,666 (conferma): 0,076
gradi di differenza dovuti al solo cuDNN non deterministico. Fra i tre semi della conferma la deviazione standard e' 0,045. La soglia di 0,20
vale circa tre volte questo rumore.

## Varianti

Tutte partono da V13, con Adam 1e-4 iniziale, batch 64 e passo di training 8:

- **W1**: annealing a coseno del learning rate da 1e-4 a 0 sulle 19 epoche, aggiornato a ogni passo. E' la lettura di «initial learning rate».
- **W2**: le 19 epoche a 1e-4 costante, come oggi, piu' un'epoca a 1e-5. E' un sostituto in fp32 dell'epoca finale di QAT int8, se quella era
  fatta a learning rate ridotto. Si tratta di un'ipotesi: il paper non dichiara il learning rate della QAT.

Nessun'altra variante: piu' epoche, un learning rate diverso o la normalizzazione dell'ingresso contraddirebbero il paper (sezione sopra).

## Passo A — screening in validazione (GPU)

Fold1 e fold2, seme 0, 12 soggetti, solo validazione: il log e la sintesi non mostrano numeri di test.

**Regola di scelta da congelare.** Si sceglie la variante con la MAE di validazione piu' bassa (fold, EMA, media su 12 soggetti x 2 fold),
solo se migliora la base V13 di almeno **0,20 gradi**. Base: **9,628**, media delle due esecuzioni di V13 col seme 0 (9,590 e 9,666).
Altrimenti la replica si chiude a 7,585 e il passo B non si fa.

Costo: 48 modelli di fold (~93 s quelli di W1, ~98 s quelli di W2 con l'epoca in piu'), un job da 1 GPU con limite 1 h 45 min: al massimo
**1,75 GPU-ore**.

## Passo B — conferma sul test (GPU, solo se il passo A sceglie una variante)

Variante scelta, fold e `full` x 3 semi su 12 soggetti, `full` grandezza principale come al passo 3. Criterio: riferimento 6,89, tolleranza
±0,30, stesse etichette. Normodotati e amputati separati, descrittivi. E' una terza analisi, dichiarata come tale: i verdetti precedenti
restano.

Costo: ~3,7 ore con W1, ~3,9 ore con W2; limite 4 h 30 min, al massimo **4,5 GPU-ore**.

## Cosa aspettarsi

Per rientrare nella tolleranza il `full` deve scendere di almeno **0,40 gradi**. Dal passo 2 al passo 3, un guadagno di 0,32 in validazione e'
diventato −0,29 sul fold di test e solo −0,13 sul `full` a parita' di seme. Una variante dovrebbe quindi guadagnare circa 0,6-1,0 gradi in
validazione per avere una possibilita'. Le oscillazioni da togliere valgono 0,25-0,5 gradi prima dell'EMA. L'esito piu' probabile, se
l'annealing aiuta, e' una baseline un po' piu' forte, non la replica. Il valore della proposta e' chiudere l'ultimo dettaglio non dichiarato
compatibile col paper.

## Costi e tetto

| Passo | Risorse | Tetto |
|---|---|---|
| A | 1 GPU, ≤ 1 h 45 min | 1,75 GPU-ore |
| B | 1 GPU, ≤ 4 h 30 min (solo se A sceglie) | 4,5 GPU-ore |
| checkout (due) | 2 CPU, pochi minuti | ~0,05 core-ore |

Il tetto attuale dell'attivita' e' di 12 GPU-ore: ne sono state spese 8,445, quindi ne restano 3,555. Il passo A ci sta; per il passo B serve
un nuovo tetto, **15 GPU-ore in tutto** (8,445 + 1,75 + 4,5 = 14,7). Ogni job resta sotto le 5 GPU-ore. Le core-ore restano nel tetto di 40.

## Codice da scrivere dopo l'approvazione

- `scripts/temponet_db8.py`:
  - opzioni `--lr-schedule {constant, cosine}`, `--final-epochs` e `--final-lr`;
  - con uno schedule, il learning rate e' un tensore CUDA che Adam (`capturable=True`) legge a ogni replay del CUDA graph. Il PyTorch di
    Leonardo lo consente: `lr: Union[float, Tensor]` in `torch/optim/adam.py`, letto il 10/11;
  - col learning rate costante resta il percorso di oggi, a numero float, cosi' le configurazioni esistenti non cambiano.
- Il controllo eager contro graph copre anche lo schedule e gira all'inizio di ogni job dei passi A e B (`--check-graph`): se fallisce, il job
  si ferma prima di allenare.
- Launcher: un modo `screen_h5` per W1 e W2, e un modo di conferma che accetta W1 e W2.
- `scripts/temponet_db8_screen.py`: regola H5, con base 9,628, varianti W1 e W2 e soglia 0,20.
- Test CPU:
  - valori dello schedule (primo passo 1e-4, ultimo ~0, epoca finale a 1e-5);
  - uguaglianza con il codice di oggi quando lo schedule e' costante;
  - regola di scelta.
- Per ogni passo: commit, deploy, checkout congelato, job dal submitter comune, ricevute in `results/`, voce di registro; nessun rilancio
  senza diagnosi.

## Cosa non si fa

Nessuna scelta sul test, nessuna variante oltre W1 e W2, nessuna QAT int8, nessuna riscrittura dei verdetti precedenti. Una TEMPONet allenata
oltre il setup del paper (piu' epoche, learning rate diverso), utile come baseline dell'harness, sarebbe un altro obiettivo, da proporre a parte.

## Esito (10/11/2026)

Simone ha approvato la proposta («Approvo», alla domanda «Approvi la proposta H5 con il nuovo tetto di 15 GPU-ore?»); regola congelata in
`docs/decisioni.md` (028616c) prima del primo job.

**Passo A**: job 64062336 (`TEMPONET-DB8-h5screen-20261110`), codice f830b9f (checkout `temponet-db8-f830b9f`, job 64062227), `COMPLETED 0:0`
in 1 h 32 min 18 s; 24 modelli di fold per variante, 12 soggetti su 12. Prima del training, controllo eager/graph con lo schedule: differenza
massima 0 in entrambe le varianti (W1 con 60 valori del learning rate, da 1e-4 a 6,85e-8; W2 con 1e-4 e 1e-5). Ricevute di sola validazione in
`results/temponet_db8/h5_screen_20261110/`: sintesi, `run.json`, controlli e curve per epoca. Il log non contiene numeri di test e il test dei 48
modelli resta non letto.

| Variante | validazione (fold, EMA) | fold1 | fold2 | contro la base 9,628 |
|---|---|---|---|---|
| W1 (coseno) | 9,566 | 9,385 | 9,746 | −0,062 |
| W2 (epoca finale a 1e-5) | 9,531 | 9,397 | 9,665 | −0,097 |

**Regola congelata: resta V13.** La variante migliore, W2, guadagna 0,097, sotto la soglia di 0,20. Il passo B non si fa e la replica si chiude
a **7,585 gradi** (V13, `full`, tre semi; passo 3 della diagnosi): **non replicata**.

Letture descrittive, sola validazione, seme 0. La base per soggetto e' la media delle due esecuzioni di V13.
- Soggetti migliori/peggiori della base: W1 7/5, W2 8/4. Normodotati −0,02 (W1) e −0,06 (W2); amputati −0,28 e −0,30 (due soggetti).
- Lo schedule agisce come previsto sul rumore fra epoche. La mediana dello scarto fra massimo e minimo della validazione grezza, fra le epoche
  10 e 19, scende da 0,55 (V13, seme 0) a 0,30 con W1; W2, costante fino all'epoca 19, resta a 0,52.
- Validazione grezza a fine training: V13 10,223, W1 10,105, W2 10,097. L'epoca a 1e-5 di W2 migliora la sua epoca 19 di 0,067 in media, in
  15 modelli su 24.
- Dopo l'EMA il vantaggio scende a 0,06-0,10: l'EMA (alpha ~0,93, invariato) toglieva gia' gran parte di quel rumore. E' la stessa misura della
  differenza fra due esecuzioni di V13 con lo stesso seme (0,076).
- Lettura: lo schedule del learning rate non spiega la distanza residua di circa 0,70 gradi.

**Consuntivo H5**: 1,538 GPU-ore (12,3 ore locali con billing 8) e 0,03 core-ore (due checkout da 2 CPU, 36 e 13 secondi; quello di 58132b4
non e' stato usato). Attivita' intera: **9,983 GPU-ore sul tetto di 15** (79,9 ore locali) e **8,07 core-ore sul tetto di 40**. Spazio: 38 MB
su `$WORK`.

Restano ipotesi non verificate e non ricostruibili dal testo del paper, per esempio la QAT int8 vera o differenze nella definizione della
metrica e degli istanti di test. Il piano H5 e' esaurito: nuove varianti solo con una nuova proposta.
