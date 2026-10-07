# CapgMyo: diagnostica locale dell'ordine dei canali

Data: 2026-10-07. Autore: GPT. Analisi esplorativa, non una soglia di accettazione anatomica.

## Esito

Il campione sostiene la mappatura **16 righe x 8 colonne, ordine C**. Questa
mappatura coincide con il percorso software degli autori esaminato. Supera la
vecchia `8x16` in ordine C in 72/72 file sia per correlazione assoluta tra vicini
sia dopo rimozione diagnostica del modo comune. E' prima tra i cinque grafi
distinti anche per regolarita' RMS, correlazione degli inviluppi e picchi di
cross-correlazione. Non sono determinati verso prossimale/distale, lato anatomico
o senso di avvolgimento: riflessioni globali non cambiano le adiacenze.

Nessuna modifica a ingestione, sidecar, array processati, manifest o montaggi
virtuali. Nessun job SLURM o training: tutti i calcoli sono stati fatti sul Mac.

## Campione e provenienza

- Selezione stabilita prima del confronto: soggetti 001, 007, 013; tutti gli 8
  gesti; ripetizioni 001, 005, 009. Totale: 72 MAT, 24 per soggetto.
- Ogni `data` e' una matrice `(1000, 128)`; controllati anche subject, gesture e
  trial rispetto al nome del file. Frequenza usata: 1000 Hz.
- Copia locale di soli questi file, senza modificare gli originali:
  `/leonardo_scratch/large/userexternal/sbenatti/data/raw/capgmyo/` ->
  `/Users/simonebenatti/wearusfm_local/samples/capgmyo_grid_20261007/`.
- SHA256 di ciascun MAT nel report completo. Nessun canale piatto/quasi piatto
  escluso nei 72 file; dati non finiti vengono rifiutati, non imputati.
- Checkout di base durante l'esecuzione: `4a2d6f4094ea70f2cea11bb8be2d6b5949b3f601`.
  Lo script era ancora non committato: l'identificatore esatto del codice eseguito
  e' il suo SHA256, `78635b782ebe871f6bd9a152378f1988696991f4e7b68c12f9273af087e341a2`.
  Il commit di pubblicazione si ricava dalla storia Git di questo documento.
- Dati e grafici rimangono fuori Git. Nel repository vengono salvati codice,
  test e il riepilogo numerico aggregato in
  `results/capgmyo_grid_20261007/summary.json`.

## I quattro controlli

1. **Mappature alternative.** Indici delle 128 colonne assegnati a griglia senza
   modificare i segnali: ordine C, ordine F, serpentina per righe/per colonne,
   otto blocchi 8x2 con ordine interno C/F, riflessione globale e vecchia 8x16 C.
   Sono otto nomi ma solo cinque grafi di adiacenza distinti:

   | Rappresentante | Altre mappature con le stesse coppie di vicini |
   | --- | --- |
   | `row16x8_C` | `modules8x2_F_transposed`, `row16x8_C_mirrored` |
   | `column16x8_F` | `legacy8x16_C` |
   | `snake_rows` | nessuna |
   | `snake_columns` | nessuna |
   | `modules8x2_C_transposed` | nessuna |

   Si confrontano i cinque grafi, non otto ipotesi indipendenti. Ogni rettangolo
   ha 232 coppie: nessun collegamento aggiunto tra i bordi o avvolgimento assunto.

2. **Mappe RMS.** Finestre di 50 ms, passo 25 ms (39 per MAT); mappa illustrativa
   della mediana RMS delle finestre per gesto 1/ripetizione 1 di ogni soggetto,
   con scala cromatica condivisa tra i candidati dello stesso soggetto. La metrica
   di regolarita' e' la media delle differenze assolute di log RMS tra vicini su
   tutte le finestre: valori minori indicano maggiore regolarita'. Gli stessi
   valori sono solo ricollocati: una mappa visivamente liscia non e' prova autonoma.

3. **Correlazioni e cross-correlazioni.** Pearson firmata e assoluta sul segnale,
   Pearson degli inviluppi RMS e picco assoluto su ritardi da -3 a +3 ms, con
   centratura/normalizzazione ricalcolate sulla porzione sovrapposta a ogni ritardo.
   Analisi di sensibilita' separata sottraendo a ogni istante la mediana dei canali
   validi (CMR): non e' una modifica al preprocessing concordato. Registrate anche
   correlazioni su tutte le coppie come baseline. I picchi selezionati sono
   ottimisticamente distorti e non dimostrano propagazione delle fibre.

4. **Percorso degli autori.** Esaminati tre file, scaricati ma non eseguiti, del
   repository `Answeror/srep`, commit fissato
   `1726e9ed09ffdf70b4c2f33f2baee01efe9394d6`. Il loader carica `data` dal MAT
   senza riordinare le colonne; il percorso esaminato applica reshape NumPy C
   a 16x8. I filtri per canale esaminati conservano l'ordine delle colonne.
   Riferimenti primari:
   [dimensioni e loader](https://github.com/Answeror/srep/blob/1726e9ed09ffdf70b4c2f33f2baee01efe9394d6/sigr/data/capgmyo/__init__.py#L13),
   [reshape](https://github.com/Answeror/srep/blob/1726e9ed09ffdf70b4c2f33f2baee01efe9394d6/sigr/data/__init__.py#L368),
   [preprocessing](https://github.com/Answeror/srep/blob/1726e9ed09ffdf70b4c2f33f2baee01efe9394d6/sigr/data/preprocess.py).
   SHA256 dei tre snapshot e righe rilevanti sono nel report e nel riepilogo.

Con indice di colonna del MAT `i` a base zero:

```python
row = i // 8
col = i % 8
grid = data.reshape(data.shape[0], 16, 8, order="C")
```

Quindi canali 1-8 sulla prima riga, 9-16 sulla seconda, ..., 121-128 sull'ultima.
Il codice stabilisce gli indici software, non il significato anatomico degli assi.

## Risultati numerici

Medie delle metriche per trial, poi per soggetto e infine media dei tre soggetti;
nessun p-value, intervallo di confidenza o test di accettazione preregistrato.

| Grafo | Pearson assoluta vicini ↑ | Dopo CMR ↑ | Pearson inviluppi ↑ | Differenze log RMS ↓ | Picco cross-correlazione assoluta ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| **16x8 C** | **0.5205** | **0.4775** | **0.5903** | **0.5103** | **0.5736** |
| 16x8 F = vecchia 8x16 C | 0.3951 | 0.3443 | 0.4736 | 0.6120 | 0.4525 |
| Serpentina righe | 0.3909 | 0.3557 | 0.5180 | 0.5396 | 0.4607 |
| Serpentina colonne | 0.3587 | 0.3201 | 0.4640 | 0.6108 | 0.4232 |
| Blocchi 8x2 C trasposti | 0.3757 | 0.3310 | 0.4997 | 0.5470 | 0.4452 |

| Soggetto | 16x8 C, Pearson assoluta | Vecchia 8x16 C | 16x8 C dopo CMR | Vecchia dopo CMR |
| --- | ---: | ---: | ---: | ---: |
| 001 | 0.4477 | 0.3144 | 0.4130 | 0.2786 |
| 007 | 0.3609 | 0.2420 | 0.3308 | 0.2144 |
| 013 | 0.7530 | 0.6289 | 0.6888 | 0.5400 |

Il conteggio 72/72 si riferisce al confronto 16x8 C contro la vecchia 8x16 C,
non a 72 osservazioni statisticamente indipendenti ne' al confronto con ogni
alternativa su ogni metrica. I valori elevati del soggetto 013 non sono da soli
prova geometrica: la preferenza rimane anche nella sensibilita' CMR e negli altri
due soggetti.

## Artefatti e riproduzione

Output completo locale:
`/Users/simonebenatti/wearusfm_local/reports/capgmyo_grid_20261007_signals/`.
Contiene `report.json`, `scores.png`, `correlations.png`, tre `rms_s*.png` e tre
`signals_s*.png`: canali 33-40 (riga 4, indice zero) con segnali a scala comune,
sola sottrazione della media e offset verticali, posizione sulla griglia e RMS
nel tempo. Le unita' sono quelle del MAT, senza conversione fisica assunta.
La prima esecuzione senza esempi temporali rimane nella directory
`capgmyo_grid_20261007/`; metriche e hash degli input coincidono esattamente.
Il JSON include mappature di tutti i 128 canali, metriche per asse/trial/soggetto,
hash degli input e del codice; i grafici sono stati ispezionati visivamente.

```bash
cd /Users/simonebenatti/dev/wearusfm
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/wearusfm_capgmyo_mpl \
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python scripts/check_capgmyo_grid.py \
  --raw-root /Users/simonebenatti/wearusfm_local/samples/capgmyo_grid_20261007 \
  --out-dir /Users/simonebenatti/wearusfm_local/reports/capgmyo_grid_rerun \
  --source-dir /Users/simonebenatti/wearusfm_local/sources/srep_1726e9ed
```

L'output deve essere una directory nuova/vuota fuori dal repository. Per cambiare
campione: `--subjects ... --gestures ... --trials ...`; per le figure includere
gesto 1/trial 1, altrimenti usare `--no-plots`. Dipendenze: NumPy, SciPy,
Matplotlib (solo figure); nessuna installazione nuova fatta per questa analisi.

Test CPU con dati sintetici: invarianti delle mappature, vicinanze, riflessioni,
Pearson, ritardi e scala, segnale liscio noto, canali piatti, input non finiti,
metadati MAT, immutabilita' input, JSON e protezione degli output. Eseguiti anche
i test esistenti dell'ingestione CapgMyo: **24 passati**. L'esecuzione reale
ha verificato anche la generazione dei nuovi grafici dei segnali.

## Limiti e prossimo passo

Il campione non copre tutti i soggetti/sub-dataset. Segnale comune, crosstalk e
inversioni di fase influenzano le correlazioni. Transposizioni/riflessioni possono
conservare le adiacenze; i ritardi non risolvono da soli l'orientamento anatomico.
Per nominare gli assi fisici servono schema di acquisizione, figura documentata o
conferma degli autori. Non usare questa diagnostica per dichiararli risolti.

La correzione dell'ingestione a 16x8 C e la gestione dei sidecar esistenti sono
un passo successivo da concordare: cambiando metadati geometrici occorre valutare
invalidazione dei manifest congelati e degli artefatti dipendenti. Non viene
chiuso automaticamente alcun punto firmato di `docs/decisioni.md`.
