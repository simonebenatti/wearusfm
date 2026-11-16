# Review della proposta Rank-N-Contrast di Claude — 16/11/2026

**Verdetto: idea pertinente per la regressione, nucleo numerico verificato; da correggere e completare prima di un training.**
La consegna è un prototipo in chat, non una patch integrata e collaudata. La seconda versione sostituisce la prima:
questa review valuta quella finale, compresi gli innesti suggeriti in `training/jepa.py` e `training/run.py`.
Non è l'implementazione della proposta InfoNCE del 07/11: cambia la domanda scientifica.

## 1. Perimetro ed evidenze

- Allegato di Simone: SHA256 `e4289c44da8f556443adfe2261608aa1744d23d52d5dc37bcdc6735757bdfdcc`, 42.953 byte.
- Modulo finale, righe 392–548 dell'allegato: SHA256 `8883c510f505534bcce1483ef6891cb2630d94fac4d73f8006f6eb1c502b8cc4`.
- Test finali, righe 637–748: SHA256 `8abac7f0f50a70a6884371d12fa541609ad58f472fa0579396a31e8690379dad`.
- Riferimento del repository: `df362727f3b1555c975d9ab2a2c5a96fd3391781`; `jepa.py` e `run.py` uguali a quelli di `origin/main` esaminato. Nessun commit di implementazione RnC fornito; nessun `rank_contrast.py` sul ramo Claude remoto esaminato.
- [Estratto, test, controesempi, hash e log](../../results/infra/rnc_review_20261116/). Sono materiale di review: il modulo è caricato in isolamento, **non installato in `src/`**.
- Tre verifiche indipendenti: matematica/numerica, integrazione nel trainer e disegno scientifico; controesempi rieseguiti dal revisore principale.

**Verifiche locali:** 12 test forniti da Claude e 21 test indipendenti passati; un test indipendente in `xfail` esplicito riproduce il difetto delle temperature estreme descritto sotto. I 21 test comprendono 128 combinazioni forward/backward contro un oracolo indipendente, differenze finite, maschere e pareggi, aggregazione per finestra/token, autocast CPU bf16 e conteggi int64. Rieseguiti anche i due controesempi separati sulle coppie `cross` e sulla partizione del batch.

Non eseguiti training completo, DDP, CUDA, NCCL o benchmark; nessun job o monitor. I test del modulo non certificano gli innesti nel trainer, che nell'allegato sono ancora frammenti con parti indicate da `...`.

## 2. Che cosa propone effettivamente

La loss legge la media dei 64 slot **a ogni istante**, sullo stesso encode mascherato dello student usato per la posa.
Ordina le distanze fra i bersagli angolari standardizzati e chiede una geometria coerente delle rappresentazioni.
È supervisione aggiuntiva dagli stessi angoli, non nuova informazione disponibile nei dati.

| Aspetto | Proposta del 07/11 | Prototipo di Claude |
| --- | --- | --- |
| Domanda | Migliorare la rappresentazione generale misurata da P1/P2, preservando la posa | Organizzare la rappresentazione temporale secondo la distanza fra pose |
| Tipo | InfoNCE su istanze, senza etichette per definire coppie | Rank-N-Contrast supervisionata dagli angoli |
| Dati che contribuiscono | Mix train, secondo ammissibilità dei negativi | Soltanto esempi con posa valida |
| Unità | Finestra aggregata | Istante sottocampionato, non intera traiettoria |
| Candidati | Batch globale e memoria teacher per dataset | Solo micro-batch del singolo processo |
| Teacher nel nuovo termine | Sì, positivo e memoria | No: rappresentazioni student con gradiente da entrambi i lati |
| Parametri nuovi | Proiettore | Nessuno |

Il principio è sensato per target continui: non occorre trasformare gli angoli in classi artificiali.
Il [lavoro RnC originale](https://papers.neurips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html)
motiva l'ordinamento delle rappresentazioni rispetto ai target di regressione. La
[loss ufficiale](https://raw.githubusercontent.com/kaiwenzha/Rank-N-Contrast/main/loss.py)
usa distanza L1 sulle etichette, similarità −L2, due viste e ogni candidato anche come positivo nel suo termine.
Coseno dopo LayerNorm, RMS standardizzata, normalizzazione al valore uniforme, sottocampionamento e filtro `cross`
sono adattamenti di Claude: plausibili in parte, ma non automaticamente coperti dai risultati del paper.

## 3. Rilievi da risolvere

### R1 — Alta priorità: `cross` può separare pose identiche

Nell'estratto, righe 63–64, `cand` contiene anche gli altri istanti della stessa finestra; `pos` li elimina.
Questi istanti partecipano ai denominatori ammissibili, senza mai essere la coppia positiva.
Non è una semplice esclusione delle coppie ridondanti: modifica l'obiettivo.

Controesempio eseguito: quattro pose identiche, due per finestra, logit uniformi. Per una query:

| Candidato | Derivata della loss normalizzata rispetto al logit | Direzione imposta dalla discesa |
| --- | ---: | --- |
| Posa identica, stessa finestra | +0,303413 | Diminuire la similarità |
| Posa identica, altra finestra, primo istante | −0,151707 | Aumentare la similarità |
| Posa identica, altra finestra, secondo istante | −0,151707 | Aumentare la similarità |

Il segno è sul logit, non una misura della traiettoria di un training; dimostra comunque la pressione indesiderata.
Con `all`, o escludendo la stessa finestra **sia dai candidati sia dai positivi**, le derivate in questo caso sono nulle.
Prova: `review_cross_semantics.py`, risultato `cross_semantics.json`.

**Richiesta:** non usare la modalità `cross` attuale come scelta predefinita del candidato. Partire da `all`, oppure definire un filtro motivato che rimuova le coppie da entrambi i ruoli. La seconda scelta richiede di verificare che restino abbastanza confronti. Non cambiare silenziosamente modalità durante un run.

Anche la frase «due pose quasi uguali non diventano mai negativi» va corretta: nella RnC un punto vicino può comparire nel denominatore del confronto con un punto ancora più vicino. Il vantaggio è il rango relativo, non l'assenza di ogni competizione fra pose vicine.

### R2 — Alta priorità: il bacino locale cambia l'obiettivo quando si divide il batch

`LossTerms` può sommare correttamente numeratori e conteggi locali, ma non ricostruisce coppie che attraversano GPU o micro-batch.
Il test «batch intero = micro-batch» suggerito nella consegna è incompatibile con questa definizione.

Prova sugli stessi quattro esempi, sei patch ciascuno, stride 2, `cross`:

| Partizione | Loss | Finestre contate | Norma del gradiente |
| --- | ---: | ---: | ---: |
| Un batch da 4 | 1,348515 | 4 | 0,209572 |
| Due micro-batch da 2 | 1,480332 | 4 | 0,312029 |
| Quattro micro-batch da 1 | 0 | 0 | 0 |

Fra le prime due righe il massimo scarto del gradiente è 0,06049233, pur contando le stesse quattro finestre.
Prova: `check_partition.py`, risultato `partition.json`. Non è una prova DDP eseguita: è un controesempio matematico al requisito di equivalenza.

**Richiesta:** scegliere prima dell'integrazione fra loss locale con partizione congelata e candidati globali.
La prima è una scelta scientifica lecita, ma batch locale, GPU, accumulo e ordinamento dei dati fanno parte della ricetta.
La seconda richiede gestione del gradiente anche dal lato candidato: raccogliere semplicemente gli student detached non replica l'RnC globale.
Il divieto di collettive dentro `LossTerms` non impedisce in assoluto di orchestrare i candidati nel trainer.
Una memoria teacher introduce invece un'altra variante, da dichiarare.

### R3 — Copertura e diversità dei confronti non ancora misurate

Con otto finestre locali e quota posa circa 23%, la media è 1,84 finestre di posa per rank. Come **solo esempio IID**, la probabilità di averne zero o una è 41,89%; non è una misura del loader reale, che usa blocchi correlati.
`cross` è inattivo con una sola finestra, mentre `all` può funzionare ma sfrutta soprattutto punti della stessa traiettoria.
Molte coppie temporali non equivalgono a molti esempi indipendenti.

Prima del collaudo misurare sui batch train reali: finestre con posa, registrazioni e soggetti distinti, ancore valide, update con termine attivo, duplicati e sovrapposizioni.
L'indice di finestra non è un'identità fisica: due indici diversi possono provenire dallo stesso intervallo o dalla stessa registrazione.
Non cambiare il mix dei dati solo nel candidato per far funzionare la nuova loss.

### R4 — Integrazione ancora incompleta, dichiarata correttamente da Claude

Gli innesti finali sono coerenti nei punti verificabili: `pose_time_valid` comprende patch nascoste e slab; il pre-conteggio e il forward usano `anchor_support`; chiavi e diagnostiche sono presenti anche senza posa; lo zero resta derivabile; i checkpoint precedenti mantengono peso zero.
Non emerge un mismatch deterministico di chiavi/conteggi dalle porzioni esaminate.

Restano patch effettive, CLI/config, planner, controllo della composizione in `scripts/arms_collaudo.py`, test del trainer con peso zero/attivo e senza posa, ripresa e distribuzione. Il pannello fisso fino a 32 finestre e il training locale da 8 non misurano la stessa loss contrastiva; Claude lo segnala giustamente.
Il test di invariabilità a peso zero deve coprire anche RNG, dati e stato iniziale, non soltanto la costruzione del modulo.

### N1 — Difetto numerico circoscritto alle temperature estreme ammesse

`_MASKED_LOGIT = -1e4` entra nella somma cumulativa. Se i logit veri sono ancora più negativi, il valore mascherato non è trascurabile.
Tre embedding con coseno reciproco −0,5 producono loss 2,251629 a temperatura 0,00005 e 96.179,6694 a 0,00001, mentre l'oracolo dà 1.
La temperatura proposta 0,1 non presenta questo problema. Restringere il dominio ammesso oppure implementare una somma mascherata sicura; non basta motivare la scelta con `exp(-1e4) == 0`.
Il test `xfail` conserva la regressione aperta, senza contare come verifica superata.

## 4. Lettura scientifica e raccomandazione

**Terrei RnC come candidata per migliorare la posa.** È più direttamente mirata alla regressione della discriminazione di istanze proposta in precedenza. Non la presenterei come equivalente o automaticamente superiore alla InfoNCE per tutto il FM: agisce solo sui dati etichettati e forza la geometria verso quegli angoli. Posture uguali possono accompagnarsi a contrazioni diverse utili per altri task; l'effetto sul trasferimento va misurato.

Inoltre non è una loss sulla velocità: ordina pose istantanee. Può aiutare senza recuperare l'ampiezza o la dinamica temporale; la validazione deve separare livello medio e movimento come abbiamo già fatto.

Se scegliamo questa domanda, la base che consiglio resta **T100-G 30M**, con controllo nuovo appaiato `0,2 ancore + posa` e candidato identico `0,2 ancore + posa + λ RnC`. Stessi pesi iniziali non addestrati, flusso di dati, maschere, partizione e dose; niente aggiunta contemporanea di JEPA, InfoNCE o LQ. Questa review non modifica lo stato della proposta del 07/11 né approva una nuova ricetta.

Ordine consigliato:

1. Fissare la domanda sulla posa, rimuovere la pressione `cross` dimostrata e scegliere candidati locali o globali.
2. Audit train della disponibilità e diversità delle coppie, poi completare integrazione e test pertinenti.
3. Collaudo tecnico con misura di costo, gradienti globali e clipping. Il valore della loss vicino a 1 non determina il suo peso. La testa posa parte a zero: inizialmente il suo gradiente sull'encoder è nullo, mentre RnC agirebbe subito. La diagnostica E4 locale dell'ultimo micro-batch non sostituisce una calibrazione globale.
4. Congelare una sola ricetta e il confronto: posa dinamica come endpoint centrale, P1/P2 come lettura di trasferimento e possibile deterioramento. Soglie, temperatura, stride, risoluzione e peso vanno fissati prima dei risultati; non ereditare senza discussione i primari della diversa domanda InfoNCE.

Non serve scartare il lavoro di Claude: buona parte del nucleo è riutilizzabile. Serve correggere la definizione delle coppie e decidere cosa deve dimostrare l'esperimento prima di spendere un training.

## 5. Riproduzione locale

Dal repository, con `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src` e il venv `wearusfm-tok`:

```text
python -m pytest -q -p no:cacheprovider results/infra/rnc_review_20261116/test_supplied.py results/infra/rnc_review_20261116/test_independent.py
python results/infra/rnc_review_20261116/review_cross_semantics.py
python results/infra/rnc_review_20261116/check_partition.py
```

`verification.json` riporta comandi esatti, codici di uscita e hash; `environment.log` riporta interprete, versione Torch e assenza CUDA.
Tutti gli esempi sono sintetici. Nessun segnale wearable o dato di partecipanti è stato aggiunto al repository.
