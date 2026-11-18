# SEEDS: acquisizione e verifiche autorizzate — 18/11/2026

Simone: «scarichiamolo e facciamo le verifiche. Teniamolo fuori dal pretraining e
prepariamo il confronto tra FM congelato, fine-tuning e TEMPONet». GPT è autore e
operatore di questa acquisizione separata; nessuna operazione sulle altre campagne.

Inventario ufficiale completato sui tre nodi OSF wa3qk, e7m5k, yp4vq:
**5.880 file, 59.737.275.276 byte (55,635 GiB)**. Sono 5.850 trial,
25 calibrazioni, quattro helper e un PDF demografico. Ogni soggetto ha234 trial e
una calibrazione; i tre nodi dichiarano CC BY4.0. Figure, video degli stimoli e
immagini wiki non sono segnali e sono esplicitamente esclusi dal download.
`results/infra/seeds_20261118/source_manifest.json` congela URL, versione OSF,
dimensione, SHA256 e MD5 di ciascun file. Il conteggio del manifest non sostituisce
la verifica del payload.

## Destinazioni e isolamento

- Segnali originali: `$SCRATCH/data/external_benchmarks/seeds`, mai sotto `processed`.
- Ricevute e audit: `$WORK/wearusfm_runs/results/seeds_20261118`.
- Codice: checkout detached nuovo `seeds-acquisition-20261118`, SHA pubblicato prima
  della sottomissione e verificato dal launcher.
- Nessun aggiornamento di manifest, quote, split o loader del pretraining. Marker
  obbligatorio `pretraining_allowed=false`; la CLI rifiuta altre radici di dati.
- SCRATCH è soggetto alla politica di cancellazione del cluster. La fonte pubblica,
  il manifest, gli hash e i report conservati permettono il recupero; non si dichiara
  un backup permanente locale dei60GB. I checkpoint storici non vengono modificati.

## Risorse e catena

| Fase | ID logico | CPU | RAM | Limite | Massimo core-ore | GPU-ore |
|---|---|---:|---:|---:|---:|---:|
| Checkout congelato | SEEDS-checkout-20261118 | 1 | 2GiB | 10min | 0,167 | 0 |
| Download e checksum | SEEDS-download-20261118 | 2 | 4GiB | 4h | 8 | 0 |
| Audit MAT completo | SEEDS-audit-20261118 | 2 | 16GiB | 2h | 4 | 0 |

**Tetto12,167 core-ore, zero GPU-ore.** Il limite corrente della partizione e QoS
`lrd_all_serial` è4h, verificato con scheduler il18/11; il download rispetta questo limite. Sono tetti di allocazione, non tempi previsti;
la velocità del deposito OSF e la coda non sono garantite. Quattro trasferimenti al
massimo, due retry aggiuntivi soltanto per errori di trasporto/429/5xx, entro il job.
Un hash errato ferma nuovi trasferimenti, conserva l'errore e non viene ritentato.
Almeno5GiB liberi oltre ai byte restanti, senza duplicare gli archivi compressi.

Submitter comune `scripts/submit_job.py`, ledger `~/wearusfm_local/jobs/ledger.tsv`.
Download solo dopo checkout riuscito; audit con dipendenza `afterok` sul download.
Nessun job duplicato, retry SLURM automatico, training o valutazione predittiva.
Se il download fallisce, si leggono causa e ricevuta prima di qualsiasi seguito.

## Controlli e limiti

Il download verifica size/SHA256/MD5 e riusa soltanto file già integri; un file
completo corrotto non viene sovrascritto. L'audit rilegge tutti i trial e controlla
identità, combinazioni, frequenze, forme, canali, valori finiti, calibrazioni,
durate e duplicati. Il campione locale è distinto dalla verifica completa.

I18 target sono letture del guanto normalizzate, non gradi. Valori fuori0–1 sono
ammessi e non vengono troncati. Guanto60Hz acquisiti,256Hz distribuiti; EMG2048Hz.
I due helper ufficiali differiscono: MATLAB interpola linearmente fra gli estremi,
Python ricampiona tramite FFT a8× e tronca alla lunghezza EMG. Si conservano i file
originali e si documenta la differenza; integrità strutturale non dimostra assenza
di ritardo elettromeccanico. Nessuna scelta del ritardo dai risultati test.

Il [protocollo del confronto](seeds_benchmark_protocol_20261118.md) prepara i tre
bracci con soggetti separati15/5/5 e checkpoint30M T100-G49k prespecificato. Il fit
richiede ancora gli adattatori SEEDS, una convenzione temporale comune verificata,
il collaudo della memoria con134canali e la ricetta/costo di ottimizzazione. Questa
acquisizione non afferma che quei fit siano già pronti o già eseguiti.
