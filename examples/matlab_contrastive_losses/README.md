# Le due loss contrastive, in MATLAB

Versioni leggibili delle formule implementate nel commit scientifico `db5ac41`, preparate per Simone il 16/11/2026. Usano solo MATLAB base e matrici numeriche; non richiedono Deep Learning Toolbox. Il codice Python del training e il protocollo in review rimangono invariati.

## Aprire ed eseguire

Da MATLAB, aprire questa cartella e lanciare:

```matlab
results = demo_contrastive_losses;       % esempi, numeri e figure
report  = verify_matlab_losses;          % confronto con il forward Python
edit wearusfm_info_nce
edit wearusfm_rank_contrast
```

Il demo restituisce anche le matrici intermedie in `results.nceDetails` e `results.rncDetails`, utili con un breakpoint dentro i cicli.

| File | Contenuto |
| --- | --- |
| `wearusfm_info_nce.m` | Query student e chiavi teacher: positivo della stessa finestra e negativi ammissibili |
| `wearusfm_rank_contrast.m` | Rappresentazioni per istante e angoli: ordine delle similarità guidato dalla distanza fra pose |
| `demo_contrastive_losses.m` | Due esempi sintetici con rappresentazioni corrette, disordinate e uniformi |
| `verify_matlab_losses.m` | Undici confronti numerici con le funzioni PyTorch effettive |

## 1. InfoNCE

Le righe di `Q`, `Kpositive` e `Knegative` devono essere già normalizzate L2. Per una matrice `X` con righe non nulle:

```matlab
X = X ./ sqrt(sum(X.^2, 2));
```

Per ogni query `i` si calcola:

```text
positivo = dot(Q(i,:), Kpositive(i,:)) / temperatura
negativi = dot(Q(i,:), Knegative(j,:)) / temperatura, solo j ammissibili
loss_i   = log(exp(positivo) + somma exp(negativi)) - positivo
```

Nel codice il logaritmo della somma degli esponenziali è calcolato in modo stabile. Il positivo compare una sola volta. Meno di 32 negativi ammissibili, o una query senza supporto, danno contributo zero; si divide comunque per il numero totale delle query. **Loss zero per assenza di supporto non significa un buon modello**: controllare `details.eligible`.

Nel demo il minimo è esplicitamente abbassato a 2, per avere tre righe ispezionabili. La ricetta scientifica conserva 32. Il caso uniforme vale `log(1 + numero_negativi)`; la scala di InfoNCE non è quindi direttamente confrontabile con quella di RnC.

## 2. Rank-N-Contrast

`H` ha una riga per istante selezionato, dopo la media dei 64 slot. `Y` contiene gli angoli standardizzati e `valid` i giunti validi; `windowId` indica da quale finestra arriva ciascun istante.

La selezione temporale è a monte: stride Python 8 e fase 4 corrispondono ai patch MATLAB **5, 13, 21, ...**, tenendo solo quelli fisici con posa valida. La funzione MATLAB riceve direttamente queste righe. La similarità è il coseno dopo LayerNorm senza parametri; la distanza della posa è l'RMS sui giunti validi in comune.

Per ciascun istante `i` e candidato `j`:

```text
denominatore = candidati k la cui posa dista da i almeno quanto la posa di j
termine_ij  = log(somma_k exp(similarita(i,k)/temperatura))
              - similarita(i,j)/temperatura
```

Si includono tutti i pareggi. Ogni candidato ammissibile compare come positivo nel proprio termine. `pairAllowed` deve essere simmetrica e toglie una coppia da entrambi i ruoli; la diagonale è sempre esclusa. La funzione non costruisce le identità fisiche: il chiamante deve escludere in questa matrice le copie dello stesso istante, come fa il runtime Python.

Per ciascuna ancora si divide la somma dei termini per il riferimento a similarità uniformi (`sum(log(numero_candidati_nel_denominatore))`). Servono almeno due candidati. Si media prima sulle ancore della stessa finestra, poi sulle finestre con supporto: una finestra lunga non pesa automaticamente più di una corta. A similarità tutte uguali la RnC normalizzata vale **1**, se esiste supporto. Il valore non è una probabilità né è limitato a 1.

Tutte le righe candidate devono appartenere allo stesso batch globale: dividere RnC in piccoli batch e mediare le loss modifica il problema.

## Che cosa è incluso e che cosa resta nel trainer

Questi file calcolano il **forward numerico**, per leggere le formule e provare esempi. Non implementano `dlarray`, autodiff o l'allenamento di un FM. Nel training Python il gradiente InfoNCE attraversa le query student ma non le chiavi teacher; nella RnC attraversa tutte le rappresentazioni, anche quando compaiono come candidati. I bersagli di posa non ricevono gradienti.

Pool temporale, proiettore, FIFO/EMA, filtri di dataset e tempo, gather distribuito, maschere fisiche, calibrazione dei pesi e schedule restano nel trainer. Qui `negativeMask`, `valid` e `pairAllowed` ne rappresentano gli ingressi già preparati. La versione RnC usa `resolution=0`, come la proposta.

Le combinazioni del confronto restano:

```text
C0: 0,2 * loss_ancore + loss_posa
CN: C0 + lambda_N(t) * InfoNCE
CR: C0 + lambda_R(t) * RnC
```

I lambda non vengono scelti da questi esempi.

## Verifica eseguita

Eseguito MATLAB R2026b sul Mac: **5 casi InfoNCE + 6 casi RnC**, scarto assoluto massimo dal Python float64 **9,09e−13** (tolleranza 5e−10). Inclusi maschere, negativi insufficienti/assenti, pose con NaN mascherati, pareggi, esclusione simmetrica di duplicati, finestre con numeri differenti di ancore, temperatura piccola. Anche le asserzioni del demo sono passate.

Risultati del demo: InfoNCE allineata 0,013386, disallineata 5,013386, uniforme 1,098612; RnC ordinata 0,456209, mescolata 1,261304, uniforme 1. Sono dati sintetici, non risultati scientifici di WearUsFM. Questi controlli riguardano il forward; non sostituiscono i test dei gradienti Python o quelli distribuiti ancora in review.

`python_reference.mat` è generato da `generate_reference.py` usando direttamente le due funzioni Python del repository. `reference_manifest.json` ne registra gli SHA; `verification.json` riporta l'esecuzione MATLAB. Tutti i dati di esempio sono sintetici.
