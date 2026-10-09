# Step 1: tranche downstream P1/P2 sui quattro teacher finali

Protocollo `step1_downstream_4000_v1`, congelato il 09/10/2026 PRIMA delle
nuove misure. Simone ha autorizzato «procedi», «lancia quando sei pronto»,
«riparti» e richiesto commit e aggiornamento del registro condiviso.
Tetto: **6 GPU-ora = 48 ore locali GPU, piu' 128 core-ora CPU**.
Nessun nuovo training, tuning, seme aggiuntivo o retry automatico.

## Input e domanda

Si valuta se il vantaggio spettrale della keep a0,05 conserva l'accuratezza
dei task downstream rispetto ad A keep0. Quattro teacherEMA finali a4000
update, semi modello0/1, training emg2qwerty. Commit training
`69fa2da8757b322ea74875b4e9d104a89bbf6d5d`; hash checkpoint fissi in
`scripts/step1_downstream.py`, gia' verificati sul cluster.
La campagna spettrale e' conclusa: quattro training e nove job frozen
COMPLETED0:0, pairing valido su8000batch. Delta forma medio+0,0525109691R2,
CI95%[0,0446510973;0,1240582165], tutti5criteri superati.
SHA del predecessore `step1_ab_0810/comparison.json`:
`e2ca2d71118755f9389cacf5747fd845b377659d78e1e54004d9946e9c280e44`.

## Misure congelate

Si riusano le sonde gia' collaudate `probe_p1.py` e `probe_p2.py`.
Nuove cacheQC-aware per tutti i checkpoint; nessun confronto causale con
il vecchio sanity. Teacher, batch64, seme della sonda0 per entrambi i semi
modello. Stesso preprocessing, finestre, scala e readout concatenato gia'
firmati. Encoder congelato; scaler train-only; regressione logistica
bilanciata, C nella griglia {0,001;0,01;0,1;1;10;100}, scelto su validation,
max_iter3000. Il test non sceglie C, checkpoint, vista o lambda keep.
Gli eventuali avvisi di convergenza si riportano senza rilanci o tuning.

P1: NinaProDB2/DB3/DB6, finestre1s senza overlap, restimulus positivo,
splitv1 con validation10% dei soggetti pretraining (seed0). Test8/3/2,
pesi8/13,3/13,2/13. Classi assenti dal train escluse e conteggiate secondo
protocollo firmato; nessuna esclusione aggiuntiva. SHA split:
`a18e7565157d5dbba8870ffa67c2342ff90de479da480dccb456922b6f6f4ed7`.
I quattro encoder di QUESTO pilota hanno visto soltanto emg2qwerty: i
soggetti NinaPro denominati pretraining negli splitv1 non sono stati usati
per allenare questi encoder. Si mantiene la definizione della sonda.

P2: EPN-612 e UCI-EMG, stesse finestre della replica, split per soggetto
70/10/20 ai semi0,1,2; media dei tre split per dataset, poi media dei due
dataset. Nessun limite al numero di utenti. Scala per soggetto dalle sue
prime10finestre, come il protocollo esistente: e' adattamento senza
etichette anche per il soggetto test, non scaler di feature fit sul test.

## Provenienza e incertezza

L'estrazione registra hash delle finestre realmente passate all'encoder,
etichette, ordine soggetti/sessioni/ruoli, frequenza e montaggio. Il confronto
richiede hash identici in tutti i quattro modelli, cache finiteNx768, stesso
commit di valutazione, SHA fisici di checkpoint/cache/report e split firmati.
Il commit di valutazione e' nuovo e distinto da quello dei training.
Output solo fuori Git, nuove directory; una cache vecchia o modificata
blocca la catena. Non modificare/pushare il checkout durante la campagna.

Le sonde salvano conteggi di campioni e predizioni corrette per
soggetto/classe; l'accuratezza bilanciata e' ricostruita sul test pooled,
non mediando accuratezze per soggetto. Si verifica l'identita' dei
denominatori e la corrispondenza tra conteggi e metriche dei report.

Bootstrap1000draw, seed20261009. Per ogni componente, stesso draw di
soggetti perA/B e per i due semi modello; delta medio dei due modelli.
P1: stratificato per dataset, aggregazione con i pesi firmati, CI percentile95%.
P2: bootstrap accoppiato separato per ciascuno split; gli split si
sovrappongono. SE del delta medio di un dataset = radice della media degli
SEsplit al quadrato, senza divisione per sqrt(3); combinazione fra i due
dataset con somma quadratica/2. Intervallo descrittivo delta+/-1,96SE,
non un CI percentile con split trattati come indipendenti. Per split sono
riportati anche i CI percentile. P1 e P2 riportano dataset e semi separati.
Incertezza condizionata sui due modelli addestrati, non tutta la variabilita'
del training. Meno di due classi in un draw rende il confronto non valutabile:
nessuna eliminazione automatica di draw o soggetti.

Guardia gia' proposta PRIMA delle nuove misure: **delta medio B-A >= -0,02
separatamente in P1 e in P2**. Nessuna compensazione fra task. Questa e'
una guardia pratica sul punto stimato, non non-inferiorita' statistica.
CI e risultati per dataset aiutano l'interpretazione, non introducono nuovi
gate post-hoc. Superare entrambe le guardie non autorizza automaticamente
adozione, training lungo, multi-dataset o nuovi esperimenti.

## Risorse, sequenza, artefatti

Otto estrazioni su boost_usr_prod/boost_qos_lprod,1GPU/8CPU/100G:
4P1 cap30min,4P2 cap1h, totale<=6GPUh. Otto probe su lrd_all_serial,
8CPU/30G/2h, totale<=128core-ora. Riferimento storico P1extract13:42,
P2extract41:19, sondeCPU22/33min; stima circa3,7GPUh=29,6ore locali GPU,
circa29core-oraCPU. I nuovi checkpoint/costi di hashing possono cambiare
questi tempi; il cap resta vincolante e non e' un'ETA di coda.

Il probe finale P2/B_seed1 dipende dalla propria estrazione E dagli altri
sette probe. Esegue poi il confronto dentro la stessa allocazione2h:
nessun nono jobCPU aggiuntivo. Gli altri probe dipendono dalla rispettiva
estrazione. Solo `afterok`; nessuna allocazione interattiva o retry.
Failure/timeout/mismatch: registrare e fermarsi, senza usare report parziali.

RUNS `$SCRATCH/wearusfm_runs/runs/step1_ab_0810`.
EVAL `$WORK/wearusfm_runs/results/passo6/step1_downstream_0910`.
Cache `EVAL/{p1,p2}/{A_seed0,A_seed1,B_seed0,B_seed1}/`, report accanto;
finale `EVAL/comparison.json`. Log `$WORK/wearusfm_runs/logs/`.
JobID e status si archiviano fuoriGit; risultati e registro finale dopo
chiusura della campagna. Scratch non e' archivio permanente; scadenza25/10
da gestire separatamente.

Costo precedenteA/B:15,2558GPUh training+1,5289estrazioni=16,7847GPUh;
ridge/confronto0,1311core-ora. Passo6 circa29,36/100GPUh usate,
70,64residue; dopo il nuovo cap6 restano almeno64,64GPUh.

Verifica locale pre-lancio:23testCPU unici passati (sondeP1/P2 end-to-end
su dati sintetici con inputaudit, readout, regressione logistica, conteggi
pooled, bootstrap accoppiato, aggregazioneP2, wrapper/cache/provenienza e
confronto finale). SintassiBash,CLI e gitdiffcheck passati. Nessun nuovo
testCUDA o misura sui task reali prima del congelamento.
