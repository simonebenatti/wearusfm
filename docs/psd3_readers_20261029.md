# PSD-3: accessibilità delle rappresentazioni a encoder congelato

Preparazione GPT del 29/10/2026, base `b1c1bda`. Simone autorizza la preparazione
in parallelo al controllo delle sonde affidato a Claude. **Nessun esperimento o job
Leonardo sottomesso da questa consegna.** Il codice non modifica trainer, loss del
FM, checkpoint o sonde storiche. Si addestrano soltanto lettori delle cache.

## Domanda e limiti

La classificazione interna di B19618 non mostra un beneficio rispetto a R3c allo
stesso passo. Il controllo di convergenza di Claude deve verificare prima
l'affidabilità numerica di quel confronto. PSD-3 chiede poi se una lettura più
flessibile o una vista che conserva i canali renda accessibile informazione che
la sonda lineare sul pooling non usa.

Questo primo passo riguarda **classificazione P1/P2 interna**, non regressione o
inseguimento della posa. Le cache `channel_readout` hanno già mediato il tempo.
`pool` concatena medie delle rappresentazioni locali e del backbone (768 valori);
`dec` conserva i canali del decoder (C × 384 valori) ma media le patch valide.
Cambiano quindi anche profondità e natura della rappresentazione: **dec−pool non
è un'ablazione causale della sola media dei canali**. Non basta per decidere di
togliere/ripristinare il pooling nel FM.

Per la dinamica il passo successivo, distinto, usa feature con asse temporale.
Le cache della posa già presenti conservano T × 384 ma mediano i 64 latenti per
patch: possono servire per un lettore temporale, non per una prova sui latenti
interi. Se servono questi ultimi basta una nuova inferenza dal checkpoint
congelato; non occorre riaddestrare il FM per estrarli. Qualunque modifica del FM
resta successiva a queste diagnosi.

## Inventario e provenienza

Piano macchina: `results/infra/psd3_preparation_20261029/plan.json`, costruito dai
rapporti e sidecar archiviati di B19618. Dieci cache, **5.431.115.084 byte (5,058
GiB)** complessivi, due modelli con stessi ingressi, soggetti e fold:

| Dataset | Finestre per modello | Soggetti interni | Canali | pool | dec |
| --- | ---: | ---: | ---: | ---: | ---: |
| NinaPro DB2 | 38.544 | 32 | 12 | 768 | 4.608 |
| NinaPro DB3 | 9.834 | 8 | 12 | 768 | 4.608 |
| NinaPro DB6 | 33.961 | 8 | 16 | 768 | 6.144 |
| EPN612 | 47.100 | 314 | 8 | 768 | 3.072 |
| UCI EMG | 364 | 19 | 8 | 768 | 3.072 |

L'inventario qui è **documentale**, da estrazioni già archiviate; non attesta una
nuova lettura degli array su Leonardo. La richiesta di inventario header a
semplicione è stata inviata senza autorizzare job; durante la preparazione il
canale dell'app non ha restituito lo stato della chat. Nessuna sostituzione con
un lancio diretto. Prima dell'esecuzione si verificano header e SHA effettivi.

Il piano congela SHA256 delle cache/checkpoint/ingressi/split, ricetta di
estrazione, codice di estrazione e ruoli train/val/held di ogni fold. B checkpoint
`96f87035520e291e1d96920722ae89bd440b4e5c5cbb335b7a04d65e75bfc229`;
R3c `ee78d2a96264322258f73b2b3b249c800b76276fb557f5023c5d76830b243e29`.
Sono feature **teacher**, coerenti col confronto storico di classificazione;
non confonderle con lo student della valutazione della posa.

## Disegno fissato prima dei risultati

Matrice **B/R3c × pool/dec × logistica/MLP**, su cinque dataset e quattro fold.
Per ogni coppia modello/dataset/vista la trasformazione è stimata una sola volta
per fold e condivisa dai due lettori. Nessuna vista `local` aggiuntiva o
concatenazione post hoc, nessuna ricerca su dimensione/architettura/epoche.

1. Stessi ruoli di soggetti della calibrazione storica, fissati nel piano. Train,
   validation e held sono disgiunti; ogni soggetto è held una volta. Classi
   assenti dal train del fold escluse da validation/held con la stessa regola
   storica, conteggiata e controllata contro l'impronta delle righe OOF originali.
2. StandardScaler e PCA **stimati soltanto sul train**, aritmetica float64;
   coordinate float32 identiche per i due lettori. Dimensione comune **128**,
   PCA randomized, seme20261029, iterated_power7, oversamples10, senza whitening
   né riscalatura delle componenti. Nessun cap automatico: se train≤128 si ferma.
   Il train più piccolo atteso ha223 finestre. Si registra varianza spiegata e
   impronta della trasformazione. Il confronto storico pool768 non è sostituito.
3. Logistica bilanciata L2/lbfgs, C∈{0,001;0,01;0,1;1;10;100}, max_iter12000,
   tol1e−4, seme0. C scelto dalla BAcc di validation; parità→C minore. Refit sul
   solo train, mai train+validation. Sette fit per fold; warning/n_iter/tempo
   attribuiti a ogni candidato e refit. Un fit al limite resta segnalato e non
   consente di dichiarare risolto un limite della rappresentazione.
4. MLP CPU **128→32 ReLU→classi**, senza dropout o batch normalization. Adam,
   lr1e−3, batch512,60 epoche fisse, semi0/1. Obiettivo del lettore:
   mean[w_classe × CE] + λ/2 × somma dei quadrati dei due pesi, bias esclusi;
   w_classe=N/(K×N_classe), solo train, optimizer weight_decay0.
   λ∈{0;0,001;0,01}. Stessa inizializzazione e ordine minibatch per ogni λ a seme
   fissato. λ scelto sulla media BAcc validation dei due semi; parità→λ maggiore.
   Si conservano i due candidati selezionati: nessun refit/ensemble. Prima
   inferenza held dopo la selezione. Le60 epoche sono un budget, **non una prova
   di convergenza**; si conservano curve train, punteggi train/validation,
   gradienti non finiti come errore e tutti i parametri del fit.

I semi del lettore misurano sensibilità dell'ottimizzazione, non sono due
pretraining indipendenti. La dimensione comune controlla la capacità d'ingresso
ma la PCA può eliminare direzioni discriminanti a bassa varianza. Un risultato
negativo vale per queste viste e questi lettori/dose, non prova assenza di
informazione nell'intero encoder.

## Lettura statistica

Balanced accuracy OOF per dataset. MLP: media delle due BAcc, non BAcc delle
predizioni in ensemble. P1 con pesi DB2=8/13, DB3=3/13, DB6=2/13; P2 pesi1/2.
Bootstrap di2.000 campioni di soggetti, seme20261029, stratificato per dataset:
stesso campione per tutte le celle e tutti i semi. IC95 percentile, condizionali
all'unico pretraining e ai lettori selezionati; non includono la variabilità
dell'intera procedura di addestramento/selezione. Nei draw con classi mancanti
la BAcc usa quelle presenti; si contano esplicitamente i draw con una sola classe.

Contrasti fissati, differenze in punti percentuali:

- dec−pool, per modello e lettore;
- MLP−logistica, per modello e vista;
- B−R3c, per vista e lettore;
- interazione (MLP−logistica in dec)−(MLP−logistica in pool), per modello.

Lettura descrittiva, non una famiglia di test con significatività corretta per
molteplicità. Nessuna soglia di adozione, selezione di un vincitore o via a R3
completo. Esito favorevole alla MLP: indicazione di accessibilità non lineare
nelle viste compresse. Esito favorevole a dec: motivazione a una successiva
ablazione che isoli il pooling. Nessun guadagno: restringe queste due ipotesi,
con i limiti di PCA, dose e convergenza dichiarati. La regressione resta aperta.

## Costi e ordine operativo

**Zero GPU-ore**, nessuna estrazione nuova. Campagna completa:20 shard
(modello×dataset×vista),80 trasformazioni,560 fit logistici e480 fit MLP.
Il secondo lettore riusa la trasformazione del primo. Nessun job per singolo
iperparametro. Memoria iniziale proposta32GiB/shard: DB6dec float64 vale circa
1,67GB prima di copie/workspace; verificare il picco effettivo nel collaudo CPU.

Preventivo di allocazione **provvisorio**, non stima di durata già misurata:

| Fase | Risorse proposte | Tetto core-ore |
| --- | --- | ---: |
| Checkout/verifiche | 1CPU ×1h | 1 |
| Timing train-only, B DB6/EPN ×pool/dec | 4job ×4CPU ×15min | 4 |
|20 shard completi | 20job ×4CPU ×1h | 80 |
| Rapporto completo | 4CPU ×15min | 1 |
| **Totale candidato** | **0GPU** | **86** |

I tempi dei20 shard **non sono ancora dimostrati compatibili con1h**. La prova
train-only misura PCA completa, due fit C=1 e un MLP da5 epoche sul primo fold;
non legge score validation/held. Proiezione per fold:
T_PCA +3,5×T_due_logistiche +72×T_MLP5. Non è un limite superiore: C grandi,
altri fold e modello possono essere più lenti. Pubblicare risorse e tetto finale
dopo il timing; se incompatibile con86core-ore, rivedere solo il piano di risorse
prima dei risultati, senza ridurre selettivamente celle o budget di training.
Massimo serial4h; nessun timeout seguito da retry cieco.

Residuo di riferimento prima dei controlli nuovi432,14core-ore CPU-only e
64,448056GPU-ore. Il controllo Claude ha un preventivo separato; PSD-3 non deve
occupare risorse riservate o essere contabilizzata come già eseguita. Anche il
preventivo86 non è una prenotazione nel ledger.

Sequenza: (a) preparazione/test/review ora; (b) leggere l'audit Claude e risolvere
eventuali difetti di provenienza/riproducibilità; (c) semplicione verifica cache,
esegue timing e fissa il piano effettivo tramite submitter/ledger; (d)20 shard e
rapporto. Il passaggio dipende dalla validità tecnica dell'audit, **non** dal
segno di B−R3c. Nessuna richiesta di un secondo consenso per un test già approvato.

Aggiornamento alla pubblicazione: Claude ha consegnato codice `22bf2cd`, piano
`b98ece0` e prompt di review `e074659` su origin. Questa preparazione non ne
costituisce la review e non attesta che il controllo sia già stato eseguito.

## Entrypoint e guardie

`scripts/psd3_readout.py` offre `plan`, `inventory`, `benchmark`, `run`, `compare`.
Il piano è generabile senza Leonardo dagli archivi. `inventory` legge solo gli
header NPZ e i piccoli metadati; non ricalcola SHA o carica matrici. `run` e
`benchmark` verificano hash intero/byte, pins, identità e finitezza prima dei fit.
`inventory --verify-sha` aggiunge la verifica SHA completa; l'inventario senza
questa opzione è marcato header-only, con confronto di byte/metadati/forme/dtype
ai pin. Non dichiara verificata la finitezza del payload. Si registra l'identità
del sorgente prima e dopo ogni shard, con NumPy/SciPy/sklearn/torch e backend
BLAS/thread; cambiamenti durante il calcolo invalidano il risultato.
Ogni `run` prende `--plan`, `--model`, `--dataset`, `--view`, `--out`; il CLI non
espone override scientifici. I file esistenti, anche progressi falliti, non si
sovrascrivono. Errori interrompono lo shard e lasciano audit, non un risultato
completo. Il confronto richiede `--plan`, quattro fold e tutti gli audit dei fit
completi con configurazioni congelate; controlla anche che ogni predizione sia
una classe del train del proprio fold, senza esigere che compaia nei target OOF.
Richiede tutti20 shard con codice/librerie/ricetta uguali
e finestre/etichette/soggetti appaiati; nessuna intersezione opportunistica.

Esempio per il futuro esecutore, **non lanciato** (da inserire nel submitter con
SHA, risorse e dipendenze già congelati; non eseguire sul login):

```bash
python scripts/psd3_readout.py run --plan results/infra/psd3_preparation_20261029/plan.json --model B --dataset ninapro_db6 --view dec --out "$OUTPUT_DIR/B_ninapro_db6_dec.json"
```

Moduli nuovi `psd3_inputs`, `psd3_readers`, `psd3_compare`; nessuna modifica ai
file della sonda che Claude sta strumentando. L2/CE qui appartengono al lettore
di classificazione: non avviano l'esplorazione delle loss del FM discussa prima.
