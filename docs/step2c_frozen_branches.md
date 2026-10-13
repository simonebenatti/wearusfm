# Step 2C: rami PSD e wavelet allenati sopra l'encoder raw congelato (protocollo `step2c_frozen_branches_v1`)

Proposta di Claude del 13/10/2026 su richiesta di Simone («intanto prepara il protocollo del raw congelato»), punto 2 della sequenza di GPT
(«Provare PSD e wavelet con raw congelato: confrontare raw, raw+PSD, raw+wavelet e tutti insieme [...] Serve anche completare il controllo con
encoder casuale»). **Da approvare; nessun job lanciato; il codice del wrapper si scrive dopo l'accordo sul disegno** (le parti gia' pronte sono
indicate sotto). Soglie e criteri fissati qui PRIMA di qualunque misura.

## Domanda

A encoder raw invariato, **quanto aggiungono a P1/P2 i rami spettrali allenati**, insieme e ciascuno? E il guadagno viene dall'allenamento dei
rami o gia' dai loro ingressi spettrali? Con il raw congelato l'apporto dei rami si misura senza che il raw cambi (parere GPT d5c53c1: il training
congiunto misurerebbe rami + loro effetto sul raw, un esperimento diverso, punto 4).

## Disegno

- **Encoder raw:** i teacher EMA finali **B_seed0 e B_seed1** dell'A/B dello Step 1 (gli stessi del test 2A, SHA in `scripts/step1_downstream.py`),
  caricati e congelati con `freeze_encoder` (caricamento controllato chiave per chiave; EMA solo sui rami: il raw del teacher resta bit per bit
  quello di partenza, provato).
- **Un training per teacher, con entrambi i rami** (`spectral_branches = ("psd", "wavelet")`), solo il termine dei rami sul readout:
  (1 - alpha) testa fusa su [raw, PSD, wavelet] + alpha media delle teste PSD e wavelet, alpha 0,5. Il contributo di ciascun ramo si misura in
  sonda togliendo le colonne dell'altro (R+P, R+W) dallo stesso modello. *Scelta per il costo* (un training invece di tre per teacher): misura
  il contributo di ciascun ramo allenato insieme all'altro, non di un ramo allenato da solo. Se l'esito e' positivo, i due training separati
  sono un'aggiunta da decidere a parte.
- **Controllo «rami non allenati»:** lo stesso modello al passo 0 (rami inizializzati e con le statistiche d'ingresso calibrate, teste mai
  allenate): `checkpoint_init.pt` salvato dal run stesso. Separa l'effetto dell'allenamento dei rami da quello dei loro ingressi spettrali.
- **Dati e configurazione del training:** quelli dell'A/B (`scripts/step1_ab.py`: sanity con le regole della finestra 1, solo emg2qwerty, batch
  32, 6 worker; manifest, scale, window_seconds e calibrazione dei target con gli SHA congelati in `step1_ab.py`). Seme del run = seme del
  teacher. Rami: dimensione 128, hidden 48, wavelet fissi, 16 batch di calibrazione. Ottimizzazione dei soli rami e delle loro teste: AdamW,
  **lr 1e-3, warmup 100, poi costante, weight decay 0,05, clip 1,0, 3.000 aggiornamenti** (i rami partono da zero e sono piccoli: ~120 mila
  parametri). Nessuna ricerca di iperparametri, nessuna ripresa automatica.
- **Feature P1/P2:** con il raw congelato la parte raw dell'embedding e' quella dei teacher B, gia' estratta e verificata nella tranche
  `step1_downstream_4000_v1` (M, 768 colonne). Si estraggono **solo le uscite dei rami** (2 x 128) sulla CPU (nessun encode raw), per il modello
  allenato e per quello al passo 0, con lo stesso ingresso delle sonde (filtro, scala, ritagli, bordo di 0,1 s); si concatenano a M solo se
  l'hash degli ingressi e le righe coincidono (come nel 2A) e se i pesi raw del teacher del run coincidono con quelli del teacher B d'origine.
- **Sonde:** P1 e P2 ufficiali invariati, come nel 2A. Varianti per teacher: R (= M: i report della tranche downstream, verificati per hash),
  R+P, R+W, R+PW, R+PW al passo 0.

## Endpoint e criteri (congelati)

Confronti accoppiati sui due teacher (`harness.downstream_ab.compare_reports`, bootstrap accoppiato sui soggetti, 1.000 estrazioni, seme
20261009; P1 intervallo percentile, P2 intervallo descrittivo con SE RMS sugli split sovrapposti), delta = accuratezza bilanciata della
variante - quella di riferimento, media dei due teacher.

- **Primario, «rami utili»:** R+PW contro R. Utile in un task se delta >= +0,01, limite inferiore dell'intervallo > 0 e delta > 0 con tutti e
  due i teacher; non dannoso se delta >= -0,01. **Positivo** = utile in almeno un task e non dannoso in tutti e due.
- **Secondario, «meglio delle statistiche fisse»:** R+PW contro R+S del test 2A (stessi teacher e finestre; richiede il 2A concluso e
  verificato), stessa regola. E' il riferimento economico che i rami appresi devono superare (GPT).
- **Secondario, «l'allenamento conta»:** R+PW contro R+PW al passo 0, stessa regola.
- **Descrittivi:** R+P e R+W contro R (contributo di ciascun ramo), per dataset e per teacher; curva della loss dei rami; frazione di frame
  validi per ramo e scala (le wavelet lente sotto ~48 Hz non hanno supporto in finestre da 1 s); il controllo con encoder casuale di GPT
  (`step1_random_downstream_4000_v1`, protocollo suo) come contesto su quanto ha imparato il raw.

**Lettura:**
- positivo e meglio delle statistiche fisse -> si prepara il training congiunto (punto 4: raw scongelato con la sua keep, `branch_weight`
  fissato prima dai gradienti);
- positivo ma non meglio delle statistiche fisse -> le statistiche fisse sono l'alternativa piu' semplice; i rami tornano in gioco solo con un
  obiettivo self-supervised (punto 3);
- negativo -> i rami cosi' allenati non aggiungono informazione utile alla sonda lineare: prima il punto 3, nessun training congiunto.
Un esito positivo dice che le uscite dei rami aiutano la sonda lineare, non che quell'informazione mancasse dal raw. Incertezza condizionata ai
due teacher; piu' confronti senza correzione (il primario decide, gli altri interpretano). Rami allenati solo su emg2qwerty, come i teacher.

## Esecuzione e costi (stime dai job dell'A/B e della tranche downstream; non ETA di coda)

| Fase | Job | Risorse | Tempo massimo | Stima |
|---|---|---|---|---|
| Training dei rami, raw congelato | 2 (B_seed0, B_seed1) | 1 A100, 8 CPU, 100 GB, `boost_qos_lprod` | 1,5 h ciascuno | ~0,7-1 s/passo (in gran parte dati), ~0,6-0,9 h |
| Estrazione delle uscite dei rami (allenato e passo 0) | 4 (teacher x P1/P2) | 8 CPU, 30 GB, `lrd_all_serial` | P1 1 h, P2 2 h | P2 dominato dalla lettura di EPN-612 (~40 min) |
| Sonde (R+P, R+W, R+PW, R+PW passo 0) | 8 (2 per teacher e task) | 8 CPU, 30 GB, `lrd_all_serial` | P1 2 h, P2 2,5 h | 39-43 min per sonda P1, 48-53 P2 |
| Confronto | nell'ultimo job di sonda | | | minuti |

**Tetto: 3 GPU-ora (= 24 ore locali GPU) + 192 core-ora CPU** (estrazioni 6 h x 8 CPU = 48; 16 sonde in 8 job, 18 h x 8 = 144); stima
~1,5 GPU-ora + ~125 core-ora CPU. Dal budget del passo 6 (dopo la tranche
downstream ~33,08/100 GPU-ora usate secondo la bozza di GPT; da riconciliare col consuntivo). Disco ~1 GB su `$WORK`.

## Codice: pronto e da scrivere

Pronto (commit `3dba14c`, CPU, 884 test): rami nel modello, `freeze_encoder` con caricamento controllato, EMA solo sui rami, solo termine dei
rami, calibrazione, provenienza; feature con identita' registrata (`harness/feature_sets.py`); assemblaggio e confronto del 2A
(`scripts/step2a.py`) da riusare. **Da scrivere dopo l'accordo:** `checkpoint_init.pt` del passo 0 in `freeze_encoder`; insieme di feature
«solo rami» in `extract_features` (senza encode raw); wrapper `scripts/step2c.py` (training con gli SHA d'ingresso dell'A/B, controllo dei pesi
raw contro il teacher B, estrazione, assemblaggio, sonde, confronto e criteri) e launcher; test CPU sintetici come per il 2A. Poi review di GPT,
checkout di campagna e controlli in sola lettura prima del lancio.

## Da decidere prima del codice

1. Un training per teacher con i due rami e contributi per sottrazione di colonne (proposta, ~1,5 GPU-ora), oppure tre training per teacher
   (R+P, R+W, R+PW separati, ~4,5 GPU-ora).
2. Iperparametri dei rami (lr 1e-3, warmup 100, 3.000 aggiornamenti, alpha 0,5): proposta senza misure; alternativa, un breve collaudo su dbg
   della curva di loss prima di congelarli (~0,25 GPU-ora).
3. Teacher B (proposta: stessi del 2A, e lo step 2 estende B) oppure anche A (GPT propone A come riferimento di lavoro della finestra 1): con
   A il costo raddoppia e il confronto con R+S del 2A non c'e' (il 2A usa solo B).
