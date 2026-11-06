# Review interna degli adattatori T100-LQ — 04/11/2026

Consegna successiva a `6d5b353`, in risposta alla review Claude `1d6f96a`.
Tre implementazioni con proprietà separate dei file; review incrociata
posa/sonde, review indipendente estrazione/inferenza e integrazione del
responsabile. Nessun lancio HPC, deploy o modifica ai sorgenti scientifici
storici. Questa review interna non sostituisce quella concordata con Claude.

**Esito: pronta per la review degli adattatori. Training non rilasciato.**

## Riscontri e correzioni durante la review

- Estrazione: caricamento strict del teacher, uguaglianza dei pesi e del gain,
  confronto della configurazione completa con i soli due default LQ ammessi
  per gli storici. I test D3 caricano checkpoint sintetici tramite il loader
  reale e attraversano `PE.extract` e `CR.extract_channel_readouts`: ramo zero
  uguale al lineare, ramo non nullo diverso; classificazione a 200 e 2.000 Hz,
  decoder e readout locali inclusi. I test locali non certificano kernel CUDA.
- Benchmark: stessi ingressi preparati, posa FP32 con testa, classificazione
  BF16 CUDA con decoder. Selezione train congelata; le prime due registrazioni
  preservano gli indici RNG della posa. Le prime finestre usate per la scala
  di sessione/soggetto sono conservate, anche al confine del batch P2.
- La prima versione legava la ricevuta di inferenza solo al checkpoint 5.000.
  Corretto prima di ogni lancio: ora verifica piano e codice del benchmark,
  inventario dei sorgenti, identità stabile di ricetta/input/radici, tempi
  grezzi, ordine/warmup/precisione, conteggi dalle provenienze storiche e
  ricostruzione del preventivo. L'aggiunta successiva del pin 19.618 e delle
  allocazioni non crea una dipendenza circolare fra hash dei piani.
- Nei lettori sono stati aggiunti controlli simmetrici fra ricevuta e
  provenienza: protocollo LQ, configurazione identica, flag, epsilon, finestra
  e attivazione. Hash aggiornati non rendono valida un'identità incoerente.
- Posa e sonde riusano le funzioni numeriche storiche. Il bootstrap a quattro
  celle conserva draw e intervalli marginali dei tre controlli; solo il
  candidato viene fittato. Nessuna riscrittura della provenienza storica.
- R1-b arriva nel report candidato e nel confronto di ciascun endpoint dalle
  ricevute di estrazione pinnate: tutte le 64 norme, student/teacher e famiglie,
  coerenza train/val, uso esclusivamente descrittivo. Nessuna soglia nuova.

## Verifica su artefatti storici effettivi

Verificati i sei report delle sonde e i loro audit archiviati: hash, 420 fit
storici, convergenza, riferimenti agli shard, codice/piano/cache/checkpoint e
fold coerenti con la campagna originale. Stesso contesto numerico storico.
I due piani posa e i relativi report effettivi sono accettati dal nuovo
importatore e i confronti storici sono riproducibili. Le OOF NPZ delle sonde
non sono locali: i pin coincidono con le ricevute di raccolta; la verifica
dei loro byte sarà effettuata sul cluster prima del confronto.

## Verifica integrata e limiti residui

**417 test passati:** 276 regressioni storiche/preflight/training in 134,90 s
e 141 test dei quattro adattatori in 15,86 s. Sintassi Bash dei launcher
LOGSQ valida e `git diff --check` pulito. Verificate 359 occorrenze di pin
locali, 137 file distinti, inclusi i 102 sorgenti scientifici del piano
training; copia della review Claude identica al commit originale.
Dettagli e comandi in
`results/infra/logsquare_evaluation_20261104/local_validation.json`.

I tempi di inferenza e la memoria CUDA restano da misurare. Un gruppo
rappresentativo per dataset non esaurisce forme e batch di coda; il rapporto
di forward applicato al walltime storico è una stima, non un'accelerazione
dimostrata. Il residuo della pipeline non separa ogni componente di I/O.

Per la ridge si misura il nuovo benchmark train. Per le sonde è proposta
esplicitamente a Claude la riutilizzazione delle misure storiche per shard:
ricetta, sorgenti numerici, ambiente, larghezza e conteggi devono coincidere.
Le feature possono cambiare le iterazioni del solver: nessuna garanzia di
durata e nessun aumento del tetto o retry automatico.

Piani `REVIEW_REQUIRED`, release training nulla, pin futuri nulli e rifiutati
dal runtime. Dopo approvazione degli adattatori: commit dedicato di rilascio,
checkout nuovo e sottomissione sotto l'autorizzazione operativa già presente.
Nessun nuovo consenso di Simone da richiedere per quella catena.
