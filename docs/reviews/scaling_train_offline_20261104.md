# Preparazione offline del training 300M — 4 novembre 2026

Questa consegna implementa il wrapper del solo screening T100-G-300M a 5.000
passi. Non certifica che la campagna sia pronta al lancio. Nessun accesso a
Leonardo, job, launcher, ricevuta di readiness o piano di esecuzione sono stati
creati durante questo lavoro offline.

## Identità scientifica e API

`scripts/scaling_train.py` orchestra il trainer storico, senza modificarlo.
`training_source_contract.json` registra gli SHA-256 di 103 sorgenti scientifici
e operativi, uguali byte per byte al riferimento
`afb644ea10ecc2a716160b4d8a4904e805c366fb`: l'intero albero `src/wearusfm` e i
cinque file di ingresso/controllo della catena tokenizer. Il contratto include
anche configurazione, CLI validata, ricevuta del controllo T100-G-30M a 5.000 e
pin degli ingressi. L'inventario viene confrontato con i file effettivi, non
soltanto con un sottoinsieme dichiarato.

`check_source_contract(contract_or_path, root=ROOT)` verifica il contratto usando
esclusivamente file locali e restituisce identità, configurazioni 30M/300M e
ricevuta storica. Questa API è riusabile dall'estrattore. `derive_recipe` conserva
la CLI T100-G, cambiando soltanto preset, directory di output e limite a 5.000;
aggiunge il timeout operativo. La configurazione 300M deve coincidere con quella
del collaudo approvato: WSD su 49.046 passi, warmup 1.962, batch globale 32,
quattro rank e accumulazione 1. Non carica pesi del controllo 30M.

Il comando offline `check-contract` dichiara esplicitamente
`runtime_readiness_checked: false`. La SHA Git del checkout eseguibile viene
passata al comando runtime e verificata realmente; piano e review si legano agli
hash delle sorgenti scientifiche e dei due wrapper. Questo evita un pin
autoreferenziale del commit che deve contenere il piano stesso.

## Gate prima dell'esecuzione

Il comando `run` richiede artefatti esistenti e verificati per hash, non segnaposto:

- collaudo congelato completo, con parità dei batch, tempi sulle 112 update
  misurate, margine memoria e preventivo coerente con le osservazioni;
- controllo della memoria lungo il percorso di produzione, includendo
  prefetch, valutazione fissa e pannello posa, con quattro misure per rank;
- estrattore, lettore posa, sonde di classificazione e aggregatore appaiato,
  inclusi controllo dimensionale secondario S5, verifica della convergenza,
  test, review e costi pubblicati;
- review scientifica vincolata ai tre artefatti precedenti, alle sorgenti del
  wrapper e all'allocazione effettiva;
- checkpoint e ricevuta storici del controllo 30M, con conservazione verificata
  fino al termine previsto di estrazioni e comparazione. La sola durata del
  training non basta; il limite dichiarato dello scratch resta il 13 dicembre.

Il wrapper verifica anche l'allocazione SLURM effettiva: un nodo, quattro GPU,
32 CPU, 480 GiB e il walltime pubblicato. In questa prima implementazione il
massimo è sei ore di walltime e non esiste fallback automatico a due
micro-batch. Se il benchmark richiede più risorse o una diversa accumulazione,
il gate si chiude e serve la revisione prevista dal protocollo.

Al momento mancano i lettori dimensionali, il controllo S5 e l'aggregatore:
`scaling_evaluation.py` da solo non può soddisfare la readiness. Mancano inoltre
i rapporti runtime di collaudo e memoria di produzione. Non è stato generato
alcun JSON che dichiari questi lavori completati.

## Completamento e limiti

Un exit code zero non produce da solo una ricevuta scientifica: vengono
controllati configurazione, riepilogo, assenza di STOP, dose, parametri,
checkpoint nominato a 5.000 e checkpoint finale. Le guardie strutturali
storiche verificano stati optimizer/RNG, hash e uguaglianza dello stato
addestrato fra i due checkpoint. Non dimostrano l'associazione completa di
ogni momento AdamW a ogni parametro; l'eventuale ripresa a 19.618 richiederà
il proprio controllo e non viene autorizzata da questa consegna.

La ricevuta del nuovo braccio usa `scaling-training-20261104-v1` e registra i
pin del contratto, delle sorgenti, degli artefatti di gate e del controllo,
oltre alle identità di configurazione e checkpoint. Non avvia continuazioni.

## Verifica locale

Il comando offline del contratto ha verificato le 103 sorgenti. La suite
`test_scaling_train.py` insieme a `test_tokenizer_train.py` ha completato
**86 test, tutti passati, in 3,10 secondi**. Include casi negativi su inventario,
ricetta, gate incompleti, risorse, retention, checkpoint mancanti/incompleti,
STOP, optimizer non finito, RNG assente e disaccordo fra endpoint.

I test usano checkpoint piccoli per esercitare i controller; non costruiscono
un modello 300M e non provano CUDA, memoria o throughput. Anche il parser dei
campi SLURM è verificato su fixture: la compatibilità con l'output reale resta
da controllare alla riapertura di Leonardo. La review indipendente esegue la
suite combinata dei controller dopo questa consegna.
