# Step 2C: rami PSD e wavelet allenati sopra l'encoder raw congelato (protocollo `step2c_frozen_branches_v1`)

Proposta di Claude del 13/10/2026 su richiesta di Simone («intanto prepara il protocollo del raw congelato»), punto 2 della sequenza di GPT
(«Provare PSD e wavelet con raw congelato: confrontare raw, raw+PSD, raw+wavelet e tutti insieme [...] Serve anche completare il controllo con
encoder casuale»). **Disegno approvato da Simone il 14/10/2026** («per questa mi piace la tua proposta, direi di procedere»), con le tre scelte
che restavano aperte:
1. «Quanti training: uno per teacher con i due rami, contributi per sottrazione»;
2. «un breve collaudo su dbg della curva di loss, circa 0,25 GPU-ora, che misurerebbe anche i tempi»;
3. teacher A e B: «Io proverei entrambe le soluzioni e valutiamo quale è la migliore, non abbiamo problemi di ore sulla GPU».

Codice pronto (`scripts/step2c.py`, launcher `scripts/slurm/step2c_*.sbatch`, test CPU); **nessun job lanciato**. Prima del lancio: review
indipendente di GPT, checkout di campagna, controlli in sola lettura, via di Simone col preventivo. Soglie e criteri fissati qui e in
`docs/decisioni.md` PRIMA di qualunque misura (collaudo compreso).

## Domanda

A encoder raw invariato, **quanto aggiungono a P1/P2 i rami spettrali allenati**, insieme e ciascuno? E il guadagno viene dall'allenamento dei
rami o gia' dai loro ingressi spettrali? Con quale famiglia di teacher (A: senza la keep dello Step 1, B: con la keep) l'embedding con i rami e'
migliore? Con il raw congelato l'apporto dei rami si misura senza che il raw cambi (parere GPT d5c53c1: il training congiunto misurerebbe rami +
loro effetto sul raw, un esperimento diverso, punto 4).

## Disegno

- **Encoder raw:** i teacher EMA finali **A_seed0, A_seed1, B_seed0, B_seed1** dell'A/B dello Step 1 (SHA in `scripts/step1_downstream.py`;
  B_seed0/B_seed1 sono gli stessi del test 2A), caricati e congelati con `freeze_encoder` (caricamento controllato chiave per chiave; EMA solo
  sui rami). A fine training il wrapper confronta l'encoder del run (studente e teacher, checkpoint finale e del passo 0) con quello del teacher
  d'origine: **ogni tensore uguale bit per bit**, o il run non passa all'estrazione. La testa dello Step 1 (congelata, ricalibrata con lo stesso
  file, non usata dai rami ne' dalle sonde) si registra soltanto.
- **Un training per teacher, con entrambi i rami** (`spectral_branches = ("psd", "wavelet")`), solo il termine dei rami sul readout con peso 1:
  (1 - alpha) testa fusa su [raw, PSD, wavelet] + alpha media delle teste PSD e wavelet, alpha 0,5. Il contributo di ciascun ramo si misura in
  sonda togliendo le colonne dell'altro (R+P, R+W) dallo stesso modello: misura il ramo allenato insieme all'altro, non un ramo allenato da solo.
- **Controllo «rami non allenati»:** lo stesso modello al passo 0 (rami inizializzati e con le statistiche d'ingresso calibrate, teste mai
  allenate), `checkpoint_init.pt` salvato dal run stesso col suo hash in `spectral_branches.json`. Separa l'effetto dell'allenamento dei rami da
  quello dei loro ingressi spettrali.
- **Dati e configurazione del training:** quelli del braccio dell'A/B del teacher (`scripts/step1_ab.py`: sanity con le regole della finestra 1,
  solo emg2qwerty, batch 32, 6 worker; manifest, scale, window_seconds e calibrazione dei target con gli SHA congelati in `step1_ab.py`, verificati
  prima e dopo il training). Seme del run = seme del teacher. Rami: dimensione 128, hidden 48, wavelet fissi, 16 batch di calibrazione.
  Ottimizzazione dei soli rami e delle loro teste: AdamW, **lr 1e-3, warmup 100, poi costante, weight decay 0,05, clip 1,0, 3.000
  aggiornamenti** (i rami partono da zero e sono piccoli: ~120 mila parametri). Nessuna ricerca di iperparametri, nessuna ripresa automatica.
- **Feature P1/P2:** con il raw congelato la parte raw dell'embedding e' quella del teacher d'origine, gia' estratta e verificata nella tranche
  `step1_downstream_4000_v1` (M, 768 colonne, per tutti e quattro i teacher). Si estraggono **solo le uscite dei rami** (2 x 128, insieme di
  feature "branches", nessun encode raw) sulla CPU, per il modello allenato e per quello al passo 0, con lo stesso ingresso delle sonde (filtro,
  scala, ritagli, bordo di 0,1 s). Su P2 le finestre si leggono una volta per i due modelli (cache identiche a quelle di `probe_p2`, provato).
  Si concatenano a M solo se l'hash degli ingressi e le righe coincidono (come nel 2A). Test: l'embedding completo del run allenato, encode
  compreso, coincide bit per bit con la variante R+PW.
- **Sonde:** P1 e P2 ufficiali invariati, come nel 2A. Varianti per teacher: R (il report della tranche downstream, verificato per provenienza
  e per hash nel suo confronto), R+P, R+W, R+PW, R+PW al passo 0 (R+PW0).

## Collaudo su dbg (prima dei quattro training)

Un job di `B_seed0` su `boost_qos_dbg` (1 A100, 30 minuti, limite del training 25 minuti), stesso codice e stesso launcher dei run, cartella
`collaudo/` (non e' un run del protocollo: l'estrazione lo rifiuta). Misura i tempi e la curva della loss dei rami; **nessuna sonda**. Verdetto
congelato (`step2c.collaudo_verdict`), scritto dal job in `run_provenance.json`:
- integrita': nessuno STOP, encoder del teacher intatto (finale e passo 0), controllo al passo 0 salvato;
- almeno 600 aggiornamenti;
- apprendimento: media della loss dei rami sugli ultimi 100 aggiornamenti <= 0,9 x media sui primi 100;
- stabilita': loss finita e media mobile su 50 aggiornamenti, dopo i primi 100, mai oltre 1,5 volte il minimo raggiunto fin li';
- tempo dei run da 3.000 aggiornamenti = (avvio + 3.000 x mediana dei secondi per aggiornamento dal 51esimo) x 1,25 + 10 minuti, arrotondato
  al quarto d'ora: e' l'allocazione dei quattro run; **oltre 2 ore il collaudo non passa**.

Passa -> i quattro run con gli iperparametri sopra e l'allocazione misurata (nuovo via di Simone). Non passa -> ci si ferma e si propone una
modifica (nuova decisione, nessun retry automatico). Il collaudo non ritocca gli iperparametri.

## Endpoint e criteri (congelati)

Confronti accoppiati (`harness.downstream_ab.compare_reports`, bootstrap accoppiato sui soggetti, 1.000 estrazioni, seme 20261009; P1
intervallo percentile, P2 intervallo descrittivo con SE RMS sugli split sovrapposti), sui due teacher di una famiglia, delta = accuratezza
bilanciata della variante - quella di riferimento, media dei due teacher. Regola del 2A: «utile» in un task se delta >= +0,01, limite inferiore
dell'intervallo > 0 e delta > 0 con tutti e due i teacher; «non dannoso» se delta >= -0,01; **positivo** = utile in almeno un task e non dannoso
in tutti e due.

- **Primario, «rami utili», per famiglia (A e B separatamente):** R+PW contro R.
- **Secondario, «l'allenamento conta», per famiglia:** R+PW contro R+PW0.
- **Secondario, «meglio delle statistiche fisse», solo famiglia B:** R+PW contro R+S del test 2A (stessi teacher e finestre; il report del 2A
  deve avere il suo hash nel confronto del 2A e le stesse finestre). Se il 2A non e' concluso, questo confronto si rimanda; il primario no. E' il
  riferimento economico che i rami appresi devono superare (GPT).
- **Secondario, «quale famiglia» (Simone, 14/10):** R+PW di B contro R+PW di A, teacher accoppiati per seme come nell'A/B. «B migliore» se la
  regola di positivo vale su delta = B - A; «A migliore» se vale su -delta; altrimenti nessuna differenza oltre le soglie (nessuna famiglia
  favorita a parita').
- **Descrittivi:** R+P e R+W contro R (contributo di ciascun ramo) e R+PW0 contro R, per famiglia, dataset e teacher; l'A/B sul solo raw (R di
  B contro R di A, gia' nella tranche downstream); loss dei rami (prime e ultime 100 medie per testa); frame validi della calibrazione per ramo e
  scala (le wavelet lente sotto ~48 Hz non hanno supporto in finestre da 1 s); il controllo con encoder casuale di GPT
  (`step1_random_downstream_4000_v1`, protocollo suo) come contesto su quanto ha imparato il raw.

**Lettura** (per la famiglia scelta come base del punto 4; la scelta resta di Simone):
- positivo e meglio delle statistiche fisse -> si prepara il training congiunto (punto 4: raw scongelato con la sua keep, `branch_weight`
  fissato prima dai gradienti pesati);
- positivo ma non meglio delle statistiche fisse -> le statistiche fisse sono l'alternativa piu' semplice; i rami tornano in gioco solo con un
  obiettivo self-supervised (punto 3);
- negativo -> i rami cosi' allenati non aggiungono informazione utile alla sonda lineare: prima il punto 3, nessun training congiunto.
Per la famiglia A il confronto con le statistiche fisse non c'e' (il 2A usa solo B): la sua lettura si ferma al primario e all'allenamento.
Un esito positivo dice che le uscite dei rami aiutano la sonda lineare, non che quell'informazione mancasse dal raw. Incertezza condizionata ai
due teacher di una famiglia; piu' confronti senza correzione (il primario decide, gli altri interpretano). Rami allenati solo su emg2qwerty,
come i teacher. Nessuna adozione automatica.

## Esecuzione e costi (stime dai job dell'A/B, della tranche downstream e da una misura locale; non ETA di coda)

| Fase | Job | Risorse | Tempo massimo | Stima |
|---|---|---|---|---|
| Collaudo (B_seed0) | 1 | 1 A100, 8 CPU, 100 GB, `boost_qos_dbg` | 30 min | ~0,7-1 s/passo dedotti (in gran parte dati) |
| Training dei rami, raw congelato | 4 (A_seed0, A_seed1, B_seed0, B_seed1) | 1 A100, 8 CPU, 100 GB, `boost_qos_lprod` | dal collaudo, al piu' 2 h ciascuno | ~1 h ciascuno |
| Uscite dei rami (allenato e passo 0) e varianti | 8 (teacher x P1/P2) | 8 CPU, 30 GB, `lrd_all_serial` | P1 1,5 h, P2 3 h | P1 ~40 min, P2 ~1,5 h (lettura di EPN-612 ~40 min) |
| Sonde (RPW+RPW0, RP+RW) | 16 (2 per teacher e task) | 8 CPU, 30 GB, `lrd_all_serial` | P1 2 h, P2 2,5 h | 39-43 min per sonda P1, 48-53 P2 |
| Confronto | nell'ultimo job di sonda | | | minuti |

Misura locale (Mac, un thread, 128 finestre): uscite dei rami 16,6 ms per finestra P1 (1 s, 2 kHz) e 41,2 ms per finestra EPN (5 s, 200 Hz,
due ritagli), contro 3-4 ms delle statistiche del 2A; finestre: P1 ~102 mila, P2 ~92,5 mila per modello.

**Tetto: 8,5 GPU-ora (= 68 ore locali GPU: collaudo 0,5 + 4 x 2) + 432 core-ora CPU** (estrazioni 4 x 1,5 h + 4 x 3 h, x 8 CPU = 144; sonde
8 x 2 h + 8 x 2,5 h, x 8 CPU = 288). Stima ~4,5 GPU-ora + ~270 core-ora. Dal budget del passo 6 (dopo la tranche downstream ~33,08/100 GPU-ora
usate secondo la bozza di GPT; da riconciliare col consuntivo, e col controllo con encoder casuale di GPT, tetto suo 3 GPU-ora). Disco ~15 GB
su `$WORK` (varianti ~3 GB per teacher, uscite dei rami 0,4 GB, checkpoint ~0,45 GB per run).

## Sequenza di lancio (dopo review di GPT e via di Simone)

Percorsi: A/B `$WORK/wearusfm_runs/runs/step1_ab_0810`, tranche downstream `$WORK/wearusfm_runs/results/passo6/step1_downstream_0910`; nuovi:
runs dei rami `$WORK/wearusfm_runs/runs/step2c_<data>`, valutazione `$WORK/wearusfm_runs/results/passo6/step2c_<data>`. Riferimento di deploy
`deploy/step2c-<data>` allo SHA rivisto, checkout di campagna, `scripts/submit_job.py` con dipendenze `afterok`.
1. Collaudo: `step2c_train.sbatch B_seed0 <runs A/B> <runs rami> collaudo` (`--qos=boost_qos_dbg --time=00:30:00`). Verdetto -> Simone.
2. Training: `step2c_train.sbatch <teacher> ...` (`--qos=boost_qos_lprod --time=<allocazione del verdetto>`), i quattro teacher.
3. Per teacher e task, `afterok` del suo training: `step2c_extract.sbatch <task> <teacher> ...` (P2 `--time=03:00:00`).
4. Per teacher e task, `afterok` della sua estrazione: `step2c_probe.sbatch <task> <teacher> RPW,RPW0 ...` e `... RP,RW ...` (P2
   `--time=02:30:00`); l'ultimo con `COMPARE=1` (e `STEP2A=<cartella del 2A>` se concluso), `afterok` di tutti gli altri.

## Codice

`scripts/step2c.py` (fasi train, extract, probe, compare); `model/spectral_branches.check_head_calibration` (review GPT del 14/10: stessa
calibrazione dei target per tutte le teste, controllata a ogni passo e quando il trainer prepara o riprende un modello); nel trainer, a encoder
congelato, il totale e' `branch_weight` x termine dei rami (review GPT del 14/10) e `checkpoint_init.pt` del passo 0; insieme di feature
"branches" in `harness/fm_features.py` e `harness/feature_sets.py` (le sonde registrano le colonne con `fm_features.feature_width`); test in
`tests/cpu/test_step2c.py` e `tests/cpu/test_spectral_branches.py`.
