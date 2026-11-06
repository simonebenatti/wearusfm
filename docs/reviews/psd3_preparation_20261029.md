# Review della preparazione PSD-3 — 29/10/2026

Base di partenza `b1c1bda`; consegna limitata a codice/protocollo/piano/test.
Nessun risultato sperimentale nuovo, lancio Leonardo o fit sulle cache reali.

Tre revisioni incrociate hanno controllato ingressi, lettori, comparatore, CLI e
protocollo. Confermati separazione dei soggetti, scaler/PCA train-only condivisi,
selezione validation senza held, class weights solo train, MLP senza ensemble,
bootstrap cluster appaiato e conteggi20shard/80PCA/560logistiche/480MLP.

Correzioni accolte prima della pubblicazione:

- Identità dei sorgenti e ambiente numerico acquisita prima e dopo il calcolo;
  firma con SciPy, librerie BLAS e numero di thread oltre alle altre versioni.
- Firma dei parametri effettivi anche nelle API usate dai test; il CLI e il
  validatore dei risultati ammettono soltanto la ricetta congelata.
- Confronto subordinato a quattro fold e audit completi, pins/ricette coerenti,
  trasformazione/etichette train identiche per i due lettori. Una vista pool
  rinominata dec viene rifiutata dal confronto con la provenienza della vista.
  Soggetti/etichette OOF sono vincolati alle impronte dei dati caricati e ai
  ruoli del piano: ID sconosciuti o maschere held vuote vengono rifiutati.
- Classi predette assenti dai target OOF ammesse se presenti nel train del fold:
  sono errori di classificazione leciti, non necessariamente corruzione.
- ID soggetto numerici/testuali mescolati rifiutati prima della coercizione NumPy.
- Inventario header-only distinto dalla verifica SHA completa, con confronto
  esplicito dei pin e nessuna dichiarazione di finitezza di payload non letto.

**156 test locali passati in96,64s** sul gruppo PSD-3 e sulle regressioni
`test_channel_readout.py` / `test_downstream_ab.py`. Inclusi test reali su cache
sintetica→quattro fold→entrambi i lettori→validatore del rapporto, PCA indipendente
contro contaminazione held, selezione e RNG, label EPN testuali, corruzioni dei
file, report incompleti, bootstrap appaiato e calcolo della media fra semi.
Le guardie successive su vista/soggetti/etichette sono state ricontrollate con
**46 test CLI/ingressi passati in3,88s**; aggiunto poi il controllo byte-string
UTF-8 per mantenere le etichette testuali serializzabili senza coercire numeri.
Suite ingressi finale: **40 test passati in0,17s**, compreso quel controllo.
Comando del gruppo principale:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /Users/simonebenatti/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider tests/cpu/test_psd3_cli.py tests/cpu/test_psd3_readers.py tests/cpu/test_psd3_inputs.py tests/cpu/test_psd3_compare.py tests/cpu/test_channel_readout.py tests/cpu/test_downstream_ab.py
```

La generazione del piano dai dieci sidecar reali passa e coincide con i pin
archiviati. Non è una nuova validazione delle cache remote: accesso/stato della
chat semplicione non disponibile durante la preparazione, inventario remoto
non restituito. Nessun dato ricostruito per supposizione. Prima del lancio:
audit Claude, verifica effettiva delle cache, timing e piano risorse definitivo.

Il preventivo86core-ore è aritmeticamente coerente, non misurato. PCA128 e dose
MLP fissa limitano le conclusioni negative; dec e pool non isolano causalmente
il pooling; tempo mediato e teacher distinguono la prova dalla posa student.
Nessun impedimento scientifico alla preparazione; nessuna attestazione di
eseguibilità in1h/shard o di convergenza delle60epoche.
