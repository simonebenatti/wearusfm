# Step 1: pilota A/B accoppiato, 4.000 aggiornamenti

Proposta operativa del 08/10/2026, preparata su richiesta di Simone dopo la
sonda frozen e il collaudo CUDA del Perceiver. **Nessun nuovo job autorizzato
o sottomesso da questo documento.** Il codice e i criteri devono essere
committati, e il costo approvato, prima dei run. ID protocollo:
`step1_ab_4000_v1`; commit di esecuzione da fissare all'approvazione.

Aggiornamento08/10/2026: Simone ha autorizzato esplicitamente il pilota
(«sono partiti? autorizzo»), entro23GPUh=184ore locali GPU piu'40core-oraCPU.
Autorizzati i4training, le4estrazioni dopo audit valido, le4ridge e il confronto
descritti sotto; nessunretry, resume, seme aggiuntivo o P1/P2 downstream.
Codice/criteri congelati nel commitb0ec17d; il commit di sola documentazione
che registra questa approvazione sara' l'expectedcommit dei lanci. Da allora
non modificare/pushare il checkout, nemmeno solo docs, fino alla fine della
campagna: il wrapper verifica l'HEAD anche alla fine e le cache devono avere
la stessa source_commit. JobID e status intermedi si archiviano fuori Git.

Correzione operativa08/10, PRIMA di un job accettato: la prima richiesta
con boost_qos_bprod e' stata respinta (`QOSMinCpuNotSatisfied`), senza
jobID/allocazione. Il preflight precedente aveva omesso MinTRES: bprod
richiede almeno65nodi/2080CPU/260GPU. Si usa boost_qos_lprod, associata
all'account e senza quel minimo, mantenendo1GPU/8CPU/100G e cap4h/6h.
Questa correzione NON cambia codice scientifico, criteri, costi autorizzati
o numero di run. Launcher aggiornato prima del lancio; anche l'estrazione
da45min deve usare override `--qos=boost_qos_lprod`. Le descrizioni bprod
piu' sotto sono lo stato della proposta iniziale, superato da questa verifica.

## Domanda e confronto

La supervisione diretta del readout condiviso aumenta la leggibilita' della
forma spettrale rispetto alle sole ancore del decoder?

| Impostazione | A, controllo | B, intervento |
|---|---|---|
| JEPA + ancore multiscala con PSD corretta | si' | si' |
| Testa keep e identica calibrazione train-only | presenti | presenti |
| Peso keep | 0 | 0,05 |
| Semi del modello | 0 e 1 | 0 e 1, accoppiati |
| Aggiornamenti per run | 4.000 | 4.000 |

Quattro training NUOVI, da inizializzazione casuale, non warm-start dal vecchio
sanity. Testa presente anche in A per non spostare l'inizializzazione
dell'encoder. Si cambia soltanto `jepa.keep_weight`; il costo del forward/
backward aggiuntivo in B viene riportato, non pareggiato tagliando gli update.
Il valore 0,05 e' una singola ipotesi gia' usata nello smoke, NON un peso
validato o selezionato sul test. Nessuna ricerca di lambda in questa tranche.

Preset sanity d384/H6/K64, locale2/backbone8/decoder2, circa28,37M; bf16 e
activation checkpointing; batch32, 6 worker su8CPU. emg2qwerty soltanto,
finestre native1-4s, regoleW1, targetPSD e poolingQC-aware gia' collaudati;
RVQ spento, ancore0,2, EMA0,996, muscle dropout0,4. AdamW lr3e-4,
weight_decay0,05, clip1; warmup lineare1.000update, poi lr costante3e-4.
Calibrazione condivisa degli80soggetti train della sonda, senza ricalcolo.
L'estensione multi-dataset richiederebbe una nuova calibrazione coerente:
non si riusa quella emg2qwerty cambiando silenziosamente la firma del loader.

Questo e' un pilota limitato, NON la finestra1 completa D12 (4epoche) o
l'applicazione dello scheduleWSD D14. Mantiene lo schedule effettivamente
collaudato, senza aggiungere una modifica di scheduler al confronto della
keep. 4.000update sono128.000finestre campionate, non128.000finestre uniche
ne' un'epoca completa. Un esito negativo a questa durata non dimostra che la
keep sia inutile dopo un training completo.

## Identita' degli input

I cinque hash fisici sono fissati anche in `scripts/step1_ab.py`:

| Input | SHA256 |
|---|---|
| manifest v1.1 | `853f35ac5252ef901be390385164c6dc5658f45270441eb74557c6be317ed6d3` |
| scale59623700 | `76486d882b5deeca66f152c3541534e9f35efca35f944f6f8677b6478b877486` |
| durate59425316, chiave `window_s` | `e660e324a28a339c038ed07dbaa7d4c358f78f613cd32226eea3f522119a48b0` |
| calibrazioneNPZ | `4640dfcf2c6819fb8ba2870e1506f08e4b024cdf38dda267c4cd37652cbc1f89` |
| split80/10/10, seed0 | `e064a1102bb4a7dd1b01a70caefa8fa2c58d8c9c751bafeafe6bd2892fff7a30` |

Percorsi sotto `$WORK/wearusfm_runs/results/passo6/`:
`capgmyo_grid_fix/manifest_v1_1.json.gz`,
`loader_59623700/session_scales.json`, `window_seconds_59425316.json`,
`step1_0710/calibration.npz`, `step1_0710/splits.json`.
Root processato `$WORK/data/processed` e `$SCRATCH/data/processed`.

## Audit accoppiato e arresti

Run unico continuo per ciascun braccio/seme, senza concatenare allocazioni.
Il resume ordinario ricrea i worker con un seme dipendente dallo step: due
interruzioni a step diversi non garantirebbero gli stessi crop ai medesimi
update. Non si modifica qui quella semantica; l'audit rifiuta resume e
warm-start prima di scrivere e richiede una directory nuova.

`pairing.json` registra SHA256 degli stati student/teacher iniziali e numero
di worker; `metrics.jsonl` registra lo SHA256 di OGNI batch consegnato,
inclusi segnali, target, crop, codici anatomici, QC, maschere, topologie e
sessioni saltate. Gli hash non consumano RNG. Le quattro configurazioni,
per coppia di seed, devono differire SOLO nel peso keep. Per ogni coppia:
stati iniziali uguali,6worker uguali,4.000batch identici e step1..4.000 senza
buchi/duplicati. Non si promette identita' bitwise degli update CUDA.

Diagnostiche di collasso e salvataggi ogni500update; norme pesate delle
componenti sullo shared encoder ogni500 in B (campo identico anche in A,
senza keep attiva). Sono diagnostiche, non tuning del peso o dello stop.
Allarmi firmati invariati: collapse_ratio<0,05 o erank<10%d per3valutazioni
consecutive. Finitezza loss/gradiente prima di optimizer/EMA, STOP definitivo,
nessuna ripresa automatica. Il wrapper verifica anche stati finali finiti,
checkpointstep4.000 e `summary.stopped=max_steps`: uscita nonzero se la
durata interna, un allarme o un guasto impediscono l'endpoint. Un checkpoint
parziale non alimenta la valutazione `afterok`.

Fallimento tecnico: registrare l'esito e non completare la coppia con un
resume improvvisato. Nessun retry, seme aggiuntivo, prolungamento o tuning
automatico. Prima delle estrazioni si verificano i quattro summary e gli
audit; un mismatch invalida il confronto. Non cambiare codice durante la
campagna. Dataloader, guard completo500ms e CapgMyo virtuale restano separati.

## Valutazione frozen e incertezza

Endpoint unico: teacherEMA finale step4.000. Nuove cache per TUTTI i modelli,
mai confronto causale con `sanity_0410`, che aveva RVQ e maschere precedenti.
100finestre per100soggetti, split80/10/10 gia' congelato; seed di estrazione0
anche per il modello seed1. SubsetSHA richiesto:
`9ba2ddbab3b18558b64f6c3bea186e1e90d49d5e89dca308a26af4df940890d2`.
Le cache devono avere X10000x768 finita, y/valid10000x32 tutte valutabili,
identici target/QC/soggetti/split e provenance completa/commit unico.

Ridge rifittata in ciascuna vista backbone/locale/concatenatoP1P2; scaler e
standardizzazione del target fit SOLO train; alpha {0,1;1;10;100} scelto
SOLO validation, separatamente per coordinata/modello/vista. Test non usato
per scegliere lambda, durata, checkpoint o vista. Viste backbone/locale
descrittive; vista primaria fissata al readoutP1P2, NON scelta a posteriori.
R2 negativi mantenuti. Tutte32coordinate devono essere valutabili: non si
eliminano quelle sfavorevoli per far passare la soglia.

Per ciascuna coordinata si calcola R2 sul test pooled; ogni famiglia e' la
media delle sue coordinate. Endpoint primario: media equiponderata di
R2 formafast e formaslow, poi differenza B-A mediata sui due semi del
modello. Energiafast/slow riportate separatamente, non confuse con forma.

Bootstrap accoppiato2000ricampionamenti dei10soggetti test, seed20261008,
CI percentile95%. Stesso campionamento di soggetti nei due bracci e in
entrambi i semi: non si fingono20soggetti indipendenti. Da statistiche
sufficienti n/media/M2/SSE per soggetto si ricostruisce il R2 pooled di ogni
ricampionamento, NON la media dei R2 dei soggetti. Anche un soggetto con
target localmente costante rimane incluso. Denominatore nullo in un draw:
misura non valutabile, nessuna omissione automatica. CI condizionato sui
due modelli addestrati: NON quantifica tutta l'incertezza fra semi di
training. Dieci soggetti e due semi restano un pilota, non evidenza definitiva.

Il trainer usa tutti i soggetti del pretraining e puo' aver visto anche i
soggetti test della ridge, con target fisici e senza etichette dei task.
Lo split separa soggetti della SONDA, non soggetti mai visti dall'encoder.
Questa misura e' leggibilita' spettrale entro pretraining, non generalizzazione
downstream o conservazione della dinamica temporale.

## Decisione prespecificata

Il pilota e' **promettente per passare a P1/P2**, non una prova di adozione,
soltanto se tutti i criteri tecnici e TUTTE queste condizioni sono soddisfatti:

1. Delta primario medio >=0,02 punti assoluti di R2.
2. Limite inferioreCI95% accoppiato del delta primario >0.
3. Delta primario positivo in entrambi i semi del modello.
4. Nessuna delle due famiglie forma peggiora nella media sui semi.
5. Nessuna famiglia (forma/energia), in alcun seme, perde piu' di0,01R2.

Soglie pratiche nuove, proposte prima dei NUOVI A/B; non provengono da
un risultato A/B gia' letto. La vecchia baseline e' gia' stata osservata:
non presentare questo pilota come una conferma completamente indipendente.
Mancato superamento: «non promettente a lambda0,05/4.000update», non «keep
inutile». Nessun rilancio sullo stesso test per superare la soglia.

Passo successivo condizionale: P1/P2 downstream sui quattro checkpoint con
cacheQC-aware nuove, probe/split seed identici fra A/B e senza selezione sul
test. Guardia proposta: media B-A sui due semi non inferiore a-2punti
percentuali in ciascuno di P1 e P2; risultati per dataset e per seme riportati,
nessuna compensazione P1/P2. E' una guardia pratica sul valore puntuale,
NON un test statistico di equivalenza/non-inferiorita'. Se supera il pilota,
preparare e approvare separatamente la tranche downstream (anche uncertainty
dei confronti) PRIMA delle sue misure. Nessuna conclusione «embedding
migliore per i task» prima di quei risultati.

## Costi e budget proposti

Passo6 residuo circa87,4247/100GPUh, dopo il Perceiver59720752.
Misure archiviate: PSD625step circa2,457s/update, tutti i dataset (proxy
per A, NON misura del controllo omogeneo); smokeB update51-200 media
3,95684s/update suemg2qwerty, prima del regime. Stima training:
2*4.000*(2,457+3,95684)/3600 =14,253GPUh, piu' setup/diagnostiche;
4estrazioni con riferimento29min43s ciascuna =1,981GPUh. Atteso totale
**circa16-18GPUh =128-144ore locali GPU**. Non e' un'ETA: coda e andamento
a regime non misurati. Lo smoke memoria29,472GiB non garantisce il picco
del training lungo; ciascun job ha una A10064G e100G RAM host.

| Job | Quantita' | Cap per job | Cap totale |
|---|---:|---:|---:|
| TrainingA,1GPU/8CPU | 2 | 4h | 8GPUh |
| TrainingB,1GPU/8CPU | 2 | 6h | 12GPUh |
| Extractteacher,1GPU/8CPU | 4 | 45min | 3GPUh |
| Ridge,8CPU/30G | 4 | 1h | 32core-oraCPU |
| Compare/audit,8CPU/30G | 1 | 1h | 8core-oraCPU |

**Autorizzazione da chiedere: massimo23GPUh (184ore locali GPU) piu'
40core-oraCPU**, senza retry. Consumo passo6 al massimo35,5753/100GPUh,
residuo **64,4247GPUh**. I40core-oraCPU sono aggiuntivi, non40GPUh;
wallclock dei job CPU fino a5h aggregate. Costo effettivo da accounting.
P1/P2 NON inclusi: stima separata fino a6GPUh (4estrazioniP1*30min e
4P2*1h), piu' fino a128core-oraCPU per8probe8CPU/2h; verificare ancora
risorse/provenance/paired-report e autorizzare dopo il segnale spettrale.

leonardo-ops ha verificato in sola lettura: boost_usr_prodMaxTime24h,
boost_qos_bprodMaxWall24h autorizzata all'account; risorse sopra compatibili.
boost_qos_dbgMaxWall30min NON adatta a45min/4h/6h. Ridge/compare su
lrd_all_serial,30G entro MaxMem30800M gia' verificato. Coda vuota,
checkout remoto8bc8c64 pulito al controllo pre-preparazione.

## Artefatti e sequenza dopo approvazione

Tutto esterno al repo. RUNS:
`$SCRATCH/wearusfm_runs/runs/step1_ab_0810/` con
`A_seed0`, `B_seed0`, `A_seed1`, `B_seed1` nuovi.
EVAL: `$WORK/wearusfm_runs/results/passo6/step1_ab_0810/`, cache
`A_seed0.npz` etc., report ridge `A_seed0.json` etc., finale `comparison.json`.
Log in `$WORK/wearusfm_runs/logs/`. Non creare in anticipo le directory
dei singoli run: l'audit ne richiede l'assenza. Non conservare su scratch
come archivio permanente; la scadenza dati25/10 resta da gestire separatamente.

1. Registrare autorizzazione con tetti e commit; push identico origin/Leonardo,
   controllo remoto pulito. Nessun codice modificato mentre i run sono attivi.
2. Sottomettere i4training usando `step1_ab_train.sbatch`, expectedcommit
   esatto; B con `--time=06:00:00`, A default4h. Nessuna catena di resume.
3. Verificare COMPLETED,summary/finitezza,provenance e pairing di tutte le
   coppie con `scripts/step1_ab.py audit --expected-commit <commit>
   --runs-root <RUNS>` (lettura breve, nessuna GPU); se incompleto o mismatch
   fermarsi prima di spendere su estrazioni.
4. Per ogni checkpoint completo, hash finale letto in sola lettura;
   `spectral_step1_extract.sbatch` con override
   `--qos=boost_qos_bprod --time=00:45:00` e input congelati. Poi ridge
   `spectral_step1_probe.sbatch` in dipendenza `afterok` dall'estrazione.
5. `step1_ab_compare.sbatch` in dipendenzaafterok da TUTTE le4ridge;
   `scripts/step1_ab.py compare` verifica audit,hash/cache e calcola bootstrap
   per tre viste. Decisione soltanto dalla vista primariaP1P2.
6. Archiviare log/config/metrics/summary/report verificati e registrare esiti,
   costo effettivo e guardie; decidere la tranche P1/P2 senza chiamare il
   miglioramento spettrale una vittoria downstream.

Nessuna allocazioneinterattiva (`salloc`); tutti i lanci passano da
leonardo-ops DOPO autorizzazione. La presente preparazione non ne ha lanciati.

## Verifica degli strumenti prima della pubblicazione

738test CPU unici passati,1skipMPS. I5casi downloader bloccati dalla bind
localhost nel sandbox sono poi passati con permesso (6test nel file,
1duplicato);19test mirati dopo l'ultimo refactoraudit/compare superati.
SintassiBash, CLI e `git diff --check` verificati. Tre update del trainer
small reale verificano l'accoppiamento e che l'hashing non modifichi pesi/RNG;
cache sintetiche verificano pipeline comparatore, bootstrap e rifiuto drift.
Nessun nuovo training reale o collaudoCUDA dell'audit; la patchPerceiver
CPU/CUDA e il vecchio smoke restano le verifiche gia' concluse, non risultatiA/B.
