# Conservazione della valutazione lunga prima della purge

07/11/2026, raccomandazione scientifica per Simone e l'operatore Claude.
**Nessuna copia eseguita da GPT, nessuna proroga o riattivazione di monitor.**
La richiesta corrente autorizza la verifica e l'indicazione di cosa salvare;
questa nota non è una ricevuta di backup.

Radice originale:
`/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/consolidation_long_20261105/`.
Possibile destinazione persistente, da verificare per quota/esistenza da parte
dell'operatore prima di una copia:
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_long_20261105/`.
Non modificare i percorsi originali nei piani congelati: conservare un manifest
che mappa originale e copia con SHA256 e byte uguali. NPZ e PT restano fuori Git.

**Raccomando di preservare l'intero insieme degli artefatti pinnati del lungo:**
sono circa **19,61 GiB censiti**. Non è una nuova misura `du` della cartella,
né una misura della quota disponibile. Conservare l'intera radice evita di
perdere dipendenze del verificatore, oltre agli esiti scientifici.

| Contenuto sotto LONG_ROOT | File censiti | GiB | Perché conservarlo |
|---|---:|---:|---|
| Cache delle tre dosi: `19618/`, `39237/`, `49046/` | 21 | 16,408 | Sequenze, bersagli, maschere, metadati, scale e ordine necessari alle verifiche |
| `fits/pose/{19618,39237,49046}/*.npz` | 15 | 2,995 | Nove predizioni e sei coefficienti ridge |
| `fits_a1/classification/**/probe.oof.npz` | 69 | 0,086 | Predizioni appaiate, identità e fold delle sonde |
| `benchmarks/pose/*.npz` | 5 | 0,103 | Artefatti richiesti da `_benchmark_gate` anche nel confronto congelato |
| Report, JSONL, ricevute e altri metadati pinnati | 242 | 0,021 | Provenienza, selezione, aggregazione, accounting; parte già in Git |

[Inventario con percorsi, dimensioni e hash](../../results/infra/consolidation_long_20261105/gpt_review_20261107/preservation_inventory.json).
Il conteggio deriva dalle ricevute pubblicate, deduplicato per percorso;
non certifica oggi presenza o integrità remota, né sostituisce l'inventario
dell'operatore al momento della copia.

Se lo spazio fosse insufficiente, la prima priorità è preservare report,
OOF classificazione e audit, predizioni/coefficenti posa **insieme ai relativi
bersagli e supporti**. Le NPZ delle predizioni contengono soltanto `prediction`:
copiarle da sole non basta. Senza costruire nuovi formati, questo richiede le
tre cache `*/pose/val/val.npz` (12,834 GiB). Le tre cache train occupano
1,325 GiB; le 15 cache classificazione circa 2,249 GiB. I checkpoint da soli
non rendono un payload immediatamente riproducibile: servono anche raw,
sidecar, statistiche, selezioni e ambiente numerico originali.

Le feature sono rigenerabili e non sono risultati insostituibili; perderle
comporterebbe però nuova estrazione e nuova verifica, non autorizzate da questa
consegna. Per circa 19,61 GiB raccomando di conservare l'insieme completo invece
di dover risolvere dopo la purge dipendenze mancanti.

## Dipendenze fuori LONG_ROOT

Una copia della sola radice lunga **non è autosufficiente**. Servono anche:

- Le OOF, i report, gli audit e le cache storiche **T100-G 19.618/pool**, riusati
  senza refit. I report `p1.json`/`p2.json` sono sotto
  `$SCRATCH/wearusfm_runs/results/tokenizer_probe_20261103/19618/T100-G/`.
  Gli altri percorsi sono fissati da `historical_classification_plan`,
  `historical_extraction_plan` e dai rispettivi report nel piano lungo.
- I riferimenti della replica **T25-L−B19618**, inclusi gli input OOF se si
  vuole riprodurla a partire dalle predizioni. Il suo report aggregato è già
  versionato in `results/infra/tokenizer_evaluation_20261103/t25_b_repeatability_20261104/paired_report.json`.
- Piani delle fasi, bridge A1, manifest, 107 stati del ledger, 103 ricevute,
  107 log e sacct. Le copie archiviate in Git e sul Mac vanno mantenute insieme
  ai payload; non tutti gli audit JSONL remoti sono necessariamente già copiati.
- I checkpoint persistenti dei tre endpoint, già pinnati sotto
  `$WORK/wearusfm_runs/results/consolidation_20261104/continuation/checkpoints/`.
  Questa review verifica i pin archiviati, non una nuova lettura remota dei pesi.

I primi file risalgono al 05–06/11: senza rinnovi, i 40 giorni portano a metà
dicembre. Non usare la data di chiusura della campagna per presumere che ogni
file scada più tardi. Completare e verificare la conservazione **prima** della
scadenza effettiva, senza contare su rinfreschi o proroghe.

Le quattro vecchie cartelle `passo6` spostate su SCRATCH il 06/11 sono un'altra
consegna e non sono protette da un backup di LONG_ROOT. La loro conservazione
resta quella comunicata dall'operatore; qui non si promettono i rinfreschi.
