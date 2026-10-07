# Replica locale delle maschere LONG: prima e dopo la correzione

2026-10-07, GPT. Richiesta di Simone: confrontare le maschere LONG in locale.

La replica conferma il difetto e la sua correzione: il codice precedente lascia
sempre la seconda meta' senza token etichettati LONG; quello corretto la raggiunge
in tutte le 24 configurazioni provate. I limiti di durata dei singoli tubi,
padding, coerenza visible/kind e tetto del budget sono rispettati da entrambe
le versioni. Questa prova riguarda il campionamento, senza misurare apprendimento.

## Versioni e metodo

- Prima: modulo `training/masking.py` al commit
  `28674c9f3c931af606b479ce4911ae14d77f9a49`, letto con `git show` e caricato in
  memoria. Il checkout non viene riportato indietro.
- Dopo: modulo corretto al commit `91a6d28112f30a13b3b12458ab436b6ca1a2005d`.
- Script: `scripts/check_long_mask_positions.py`, SHA256 eseguito
  `ac70e60fc16b16b5eb7d6967b168ee47cec0e4710136385dca7433f12279b589`.
  Lo script diagnostico era nuovo/non committato al momento della prova; il
  report identifica separatamente il commit di base e gli hash dei moduli.
- Semi 0-9, identici nelle due versioni; 40, 41, 80 e 160 patch da 25 ms
  (1 s, 1,025 s, 2 s, 4 s); padding di 16 patch.
- Topologie costruite dai metadati dei test: anello da 16 canali e griglia da
  128 canali. Tutti validi; nessun file di segnali o soggetto reale viene letto.
  La geometria della griglia usa l'ingest attuale: non e' una verifica anatomica.
- Tre configurazioni: solo LONG con RVQ spento (replica dell'isolamento usato
  per individuare il bug), D10 completa con RVQ spento, D10 completa con RVQ
  acceso. Frazione nascosta richiesta 0,5 e tolleranza di budget 10%, come nel
  generatore. Totale: 24 configurazioni x 10 semi x 2 versioni = **480 maschere**.
- Per LONG si contano solo i tubi, etichettati LONG. Gli slab RVQ hanno
  etichetta SLAB separata e non entrano in questi conteggi.

Stessi semi/configurazione consentono la replica ma non garantiscono proposte
identiche fra le versioni: il cambiamento del dominio delle estrazioni e delle
sovrapposizioni modifica anche i tentativi successivi.

## Risultati del confronto isolato

Token LONG nella seconda meta' / totale token LONG, sommando i dieci semi:

| Topologia | Finestra | Prima | Dopo |
| --- | ---: | ---: | ---: |
| Anello 16 canali | 1 s | 0% | 52,09% |
| Anello 16 canali | 2 s | 0% | 49,17% |
| Anello 16 canali | 4 s | 0% | 47,98% |
| Griglia 128 canali | 1 s | 0% | 49,05% |
| Griglia 128 canali | 2 s | 0% | 49,64% |
| Griglia 128 canali | 4 s | 0% | 49,38% |

La replica dei vecchi conteggi sull'anello produce esattamente 3.200 / 0,
6.399 / 0 e 12.792 / 0 token LONG prima/seconda meta' per 1/2/4 s.
Dopo la correzione: 1.583 / 1.721, 3.344 / 3.235 e 6.908 / 6.371.
Con 1 s la durata di ogni tubo e' sempre 20 patch (500 ms): prima l'inizio
massimo era zero, dopo arriva a 20 patch (500 ms). Con 2/4 s rimangono durate
20-40 / 20-80 patch e le proposte possono arrivare alla fine della finestra.

Anche con D10 completa la seconda meta' viene raggiunta in tutti i casi
provati. Le quote realizzate non devono necessariamente dividersi a meta':
sovrapposizioni, rifiuti e priorita' delle etichette influenzano i conteggi.
Per esempio, sull'anello da 2 s con RVQ acceso, il 18,12% dei token LONG e'
nella seconda meta' su questi dieci semi; il conteggio esclude gli slab.
Il test non impone una nuova soglia di accettazione su questa percentuale.

## Lettura dei grafici

- `long_examples.png`: anello, solo LONG, seme 0, finestre da 1/2/4 s;
  blu = token LONG nascosti, tratteggio = meta' finestra. Ogni immagine mostra
  **l'unione di tutti i tubi**, che puo' superare il limite del singolo tubo
  per sovrapposizione o contatto.
- `long_coverage.png`: per ogni patch, frazione di canali nascosti come LONG
  mediata sui dieci semi, su anello e griglia.

Un inizio uniforme dell'intervallo non produce copertura uniforme dei token:
gli intervalli possono coprire il centro con piu' posizioni iniziali rispetto
ai bordi. La curva centrale piu' alta e' compatibile con il generatore corretto.
Il difetto era il confinamento assoluto alla prima meta'.

## Artefatti, verifica e replica

Report completo e due grafici, ispezionati visivamente:
`/Users/simonebenatti/wearusfm_local/reports/long_mask_20261007/`.
Il JSON registra commit, SHA256 di script/moduli, parametri, conteggi e copertura
per patch, estremi delle durate/posizioni delle proposte e controlli degli invarianti.
Eseguiti anche i test locali del generatore: **18 passati**.

```bash
cd /Users/simonebenatti/dev/wearusfm
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR=/private/tmp/wearusfm_capgmyo_mpl \
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python scripts/check_long_mask_positions.py \
  --out-dir /Users/simonebenatti/wearusfm_local/reports/long_mask_rerun
```

Usare una directory nuova/vuota fuori dal repository. `--seeds` modifica il
numero di semi; `--before-ref` sceglie un commit locale con il modulo storico;
`--no-plots` evita Matplotlib. La prova di oggi non usa GPU o job HPC.
