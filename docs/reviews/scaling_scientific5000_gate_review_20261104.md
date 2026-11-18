# Review finale dei gate — fresh300M5000, 4 novembre 2026

**VERDETTO: APPROVATO** per il training scientifico fresco T100-G-300M da zero a
5.000 passi descritto nel contesto sotto. Nessun blocco scientifico od operativo
rilevato nelle evidenze disponibili. L'approvazione riguarda questa sola fase
già autorizzata da Simone; il wrapper deve mantenere tutte le verifiche runtime
prima di avviare `torchrun`. Non autorizza una continuazione a 19.618.

Contesto esaminato: `results/infra/scaling_300m_20261104/scientific5000_gate_context.json`,
SHA256 `a8e1d7764857476859326222e21172843f5475d399a9e91c354207c658566361`.
Preparazione pubblicata a `34b878a23fe84e6a294ab6f5cd4827b686606bbc`.
Contratto scientifico SHA256
`14b044de1df7be3e55c836685342d94ae48f9d072dc6784b33cca197dab47df1`.

## Tre gate verificati

| Gate | SHA256 verificato | Esito |
|---|---|---|
| Preflight `preflight_63526191/report.json` | `90254bb4bd8aa525730cdd43cec2fa6c98237387754145f0704037e3f5ec8831` | `check_preflight` passato; 128 update per braccio, 112 misurati; dose e sorgenti corretti; preventivo minimo 19.800 secondi |
| Produzione S3 `production_memory_63527446/report.json` | `45acd95c81db3a550eb9bd1dabf5055dd5ae5d35e94a8cb0fabb39035144f7f9` | Job reale `63527446` COMPLETED 0:0; quattro rank e 500 update completi; audit indipendente delle copie passato |
| Readiness valutazioni `evaluation_readiness.json` | `7b30f54c8527e2fee251101b926d2dec4e9b99b5b32a777afb6554a1bf059b8b` | `check_evaluation_readiness` passato, 32 stati pinnati verificati; sorgenti/test/launcher, review e tetti completi |

L'audit S3 è stato rieseguito in sola lettura sui **15 file reali raccolti**:
risultato identico al JSON pubblicato salvo l'orario della verifica. Quest'ultimo
ha SHA256 `55e13354aa18508dbc55ceea885c2446bf8d2334054d552e3bc410d260db7482`.
Sono verificati config, identità del lancio, 500 update da 32 finestre globali,
prefetch, gradienti/diagnostiche al passo 500, valutazione fissa e pannello ai
passi 0 e 500, relativi supporti e quattro ricevute CUDA.

Picchi per rank: **29,61098 / 31,24165 / 29,71045 / 34,54403 GiB**. Il massimo
corrisponde al **54,4664%** della memoria del dispositivo, sotto il limite del
90%, e si riconcilia con il MAX del summary. Python/Torch/CUDA coincidono con lo
storico; cuDNN 8907 è registrato, senza inventare un confronto storico assente.
Contabilità: **1.619 s, 1,798889 GPU-ore e 14,391111 core-ore**, oltre al checkout
già contabilizzato separatamente. S3 dimostra il percorso esercitato; non prova
il picco di ogni possibile batch futuro né la qualità scientifica del modello.

## Ricetta, risorse e controllo

`check_source_contract` ha verificato tutti i **103 sorgenti**. Rihashati anche
i **9 file** del manifest frozen: nessuna differenza. I due sorgenti esecutivi
sono quelli del contesto: `scaling_train.py` SHA256
`d641c375cbeeb80bf275afef185dfb4f695938b22d908626cb9b829ddc8440ac` e
`scaling_preflight.py` SHA256
`0a3224190957986425cd17e24001a3b8bcc7ba98fa1c190d45bbcc1e48531e04`.
La verifica integrata pubblicata registra **571 test passati, zero falliti**,
legati agli stessi byte; non è stata ripetuta inutilmente in questa review.

`derive_recipe` ricostruisce un comando fresco con quattro processi, preset
300M, **5.000 passi**, batch globale 32, accumulo 1, sei worker, prefetch,
seme 0, GELU/100 ms, stessa ricetta di obiettivi e stessi dati. LR 1e-4, warmup
1.962 e schedule WSD 49.046 restano invariati. Nessun argomento `--resume`,
`--init-from` o `--branch-from`; nessun checkpoint S3 usato come inizializzazione.

Allocazione approvata: **1 nodo, 4 GPU, 32 CPU, 480G, 19.800 s**;
timeout operativo trainer **19.500 s**. Tetto **22 GPU-ore e 176 core-ore**.
Il costo è ricalcolato dal preflight con la formula congelata; l'esito più breve
di S3 non viene usato per ridurre il preventivo. Restano separati i tetti delle
valutazioni: 8,25 GPU-ore, 66 core-ore allocate con GPU e 180 core-ore CPU.

Gli output scientifici dichiarati sono distinti da quelli S3:

- run: `/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/scaling_300m_20261104/T100-G-300M-5000`;
- ricevute: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/scaling_300m_20261104/scientific5000_v1`.

Il wrapper rifiuta directory preesistenti e verifica l'allocazione effettiva:
la loro assenza remota non è stata simulata né attestata da questa review locale.

Il controllo è il T100-G 30M a 5.000, checkpoint SHA256
`49dd518b5eea632a16bc7ff88ab0fb013f30d61458164c9521867572c83daa80`,
454.381.728 byte, con ricevuta storica
`d0a50fe777d1bf66df7670158324b48444875c645c1a9bbcb3d55e40560d96c7`.
Pin e ricevuta coincidono con contratto e verifica del file del 4 novembre.
L'orizzonte operativo comune è **11 novembre 2026, 00:00 UTC**, successivo alla
durata prevista e precedente al limite conservativo del contratto sullo scratch
(13 dicembre). È una scadenza operativa, non immunità dalla perdita dei file:
esistenza/hash e struttura del controllo vengono ricontrollati al lancio.

## Verifiche integrali da conservare sul nodo di calcolo

Il checkpoint tecnico S3 da 4.673.783.740 byte non è stato scaricato o caricato
localmente. L'audit delle copie non si spaccia per `validate_report` integrale.
È stato verificato nel codice che, **prima di `torchrun`**, il wrapper:

1. rilegge i gate pinnati e richiama `check_production_memory` /
   `validate_report`, rihashando anche il checkpoint tecnico remoto e
   ricontrollando le evidenze originali e l'assenza di STOP/endpoint scientifici
   nella cartella S3;
2. verifica il checkpoint storico 30M per hash e struttura, firma dei dati,
   dose e configurazione, oltre a ricevuta e retention;
3. rilegge input e sorgenti, controlla directory nuove e allocazione reale;
   soltanto dopo scrive la ricevuta di lancio e avvia il trainer.

I guard non devono essere sostituiti dall'audit locale né aggirati. Un loro
fallimento arresta la fase; nessun retry cieco o variazione di ricetta.

La readiness dei lettori riguarda implementazioni, regressioni, proiezioni S5,
launcher e budget già fissati. Non dichiara osservati benchmark delle cache
300M/5.000 ancora inesistenti: questi si misurano dopo le estrazioni e prima dei
fit CPU, nei tetti pubblicati. Nessun risultato downstream nuovo è stato letto
per questa decisione, nessun criterio S8 modificato e nessuna continuazione
automatica introdotta.

Verifiche locali completate il 4 novembre 2026 alle **20:28 UTC**. Unico file
scritto da questa attività: questa review; nessun SSH, job o commit.
