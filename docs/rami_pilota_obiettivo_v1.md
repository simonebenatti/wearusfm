# Rami spettrali, punto 2: pilot dell'obiettivo temporale (protocollo `rami_pilota_v1`, PROPOSTA)

Proposta di Claude del 20/10/2026 per il punto 2 della consegna sui rami (prompt di GPT del 20/10, base 3b97655). Scelte di Simone (20/10):
«A, solo famiglia B, includi il controllo di LR»; dopo la review del disegno del `critico` (DA RIVEDERE), «se consigli di aggiungere,
aggiungi»: braccio PWf. Punto 5 della tabella comune; i verdetti 2A/2C restano invariati. **Nessun job lanciato.** Prima del lancio: review
del `critico` del codice, lettura congelata in `docs/decisioni.md` col commit che precede il lancio, preventivo e via di Simone. Il lancio
segue l'esito della diagnostica del punto 1 (`docs/rami_diagnostica_v1.md`), che si legge prima ma non e' una condizione del pilot. La
review di GPT arriva dopo e non blocca.

## Perche' cambiare il compito dei rami (ipotesi, non un risultato)

Nel 2C i rami (raw congelato) imparano a predire i 32 target globali dello Step 1 (`spectral_step1.global_spectral_targets`): forma ed
energia dello spettro **medi nel tempo e sui canali**, alle stesse due scale e nelle stesse bande delle log-potenze che il ramo riceve in
ingresso (per le wavelet l'energia e' costruita uguale alla potenza di banda). Il compito e' quindi vicino a una media dell'ingresso del ramo;
un ramo che diventa un «mediatore» lo risolve, e puo' perdere dinamica temporale e differenze fra canali che al passo 0 restano leggibili.
Gia' nel 2C (decisioni.md, 14/10) era registrata come ipotesi. Il pilot confronta obiettivi; non identifica una causa. LR, schedule, dominio
di training e geometria dell'embedding restano spiegazioni alternative.

## Obiettivo nuovo: predizione di rappresentazioni locali mascherate nei rami (scelta A)

Dentro ogni ramo e ogni scala, `ScaleEncoder` calcola un token per **canale e frame** (`h`, 48 numeri) prima della media sui frame e sui
canali (`ScaleEncoder.tokens`, gli stessi valori del forward attuale, provato bit per bit). L'obiettivo nasconde tratti di tempo e chiede di
predire, dai frame visibili, i token del teacher nei tratti nascosti.

- **Teacher e target:** il teacher e' la media esponenziale (EMA, momento 0,996 come nel 2C) dello studente **sui soli parametri allenati**
  (gli strati fissi restano identici bit per bit anche nel teacher); calcola i token sull'ingresso **completo**. Target = token del teacher
  nei frame bersaglio, normalizzati per token con LayerNorm senza parametri, **stop-gradient**.
- **Contesto:** lo studente calcola gli spettri su una copia del segnale coi **tratti nascosti azzerati** e i token sui **soli frame di
  contesto** (gli altri sono «non validi»: zeri prima e dopo ogni strato dell'encoder di scala). Il banco wavelet filtra nel dominio della
  frequenza (risposta all'impulso con code gaussiane, e un salto a f <= 0 nella banda veloce piu' bassa): azzerando, gli spettri dello
  studente **non dipendono dai campioni nascosti per costruzione**; la regola del supporto qui sotto limita la distorsione dell'azzeramento.
  Misura (rumore a 2 kHz, ritagli da 4 s, 10 maschere): sui frame di contesto la differenza di log-potenza fra studente e teacher e' 0 per la
  PSD, al piu' 3,4e-3 per le wavelet veloci e 4e-5 per le lente; limite dichiarato nel test 1e-2.
- **Predittore:** per ramo e scala, un piccolo transformer sui frame di ciascun canale (2 strati, dimensione 64, 4 teste, posizione = tempo
  del centro del frame, sinusoidale), token di maschera nelle posizioni bersaglio, attenzione solo verso i frame di contesto dello stesso
  canale e campione. Inizializzato con un generatore separato (l'inizializzazione dei rami e il flusso dei dati restano quelli del 2C).
- **Perdita:** errore quadratico medio fra predizione e target sui frame bersaglio, media per ramo e scala, poi media sui rami e scale col
  almeno un bersaglio nel batch. **Riferimento banale = predittore costante** (la media dei target fra i token, raggiungibile col solo bias
  d'uscita): la sua perdita e' la varianza dei target fra i token, media sulle 48 coordinate, per batch, ramo e scala. Si registra la
  **perdita normalizzata** = perdita / quel riferimento (1 = non meglio di una costante).

### Maschere senza fughe d'informazione

- **Una maschera temporale per campione**, uguale per tutti i canali, i due rami e le due scale: tratti di **200 ms**, non sovrapposti, a
  posizioni casuali dentro la parte utile del ritaglio (fuori dalle zone di bordo `edge_guard` e a **almeno 100 ms** da esse), in numero
  **k = max(1, arrotondamento per eccesso a meta' di 0,25 x durata utile / 200 ms)**, al piu' quanti ne entrano; nessun tratto (campione
  escluso dall'obiettivo) se la parte utile e' piu' corta di un tratto. Generatore numpy proprio, seme `[seme del run, 20261020, passo]`.
- **Frame bersaglio:** centro dentro un tratto nascosto, valido per il teacher.
- **Frame di contesto (per banda):** il supporto intero del frame (finestra piu' raggio del filtro: 0 per la PSD, quello della wavelet di
  quella banda, piu' lungo alle frequenze basse) non tocca nessun tratto nascosto; un frame bersaglio non e' mai contesto. Un frame ne'
  bersaglio ne' contesto e' escluso.
- **Nessun altro ramo, nessun raw nel contesto:** il predittore di un ramo vede solo i token di quel ramo e di quella scala.
- **Bersagli senza contesto:** una combinazione (campione, canale, ramo, scala) senza frame di contesto si esclude e si conta (niente
  attenzione su un insieme vuoto).
- **Prove nei test CPU:** (1) cambiando arbitrariamente il segnale dentro i tratti nascosti, spettri, token di contesto e predizioni dello
  studente restano **identici bit per bit**, anche per le wavelet; cambiandolo fuori, cambiano. (2) Sui frame di contesto, la differenza fra
  le log-potenze dello studente (segnale azzerato) e del teacher (segnale completo) resta sotto il limite dichiarato nel test.
- **Copertura (stima):** con ritagli quasi tutti da 4 s, la scala veloce ha ~110 frame di contesto e la PSD lenta ~10 per ritaglio; le bande
  wavelet lente sotto ~40 Hz (raggio 0,2-0,36 s) ne hanno 1-4; sui ritagli piu' corti di ~1,5 s la scala lenta ha pochissimo contesto. I
  pesi di `project` per quelle bande ricevono poco gradiente: **asimmetria dichiarata nella lettura**. Il collaudo misura la copertura reale
  per ramo, scala e banda. Misura su rumore a 2 kHz, 4 s, caso peggiore su 10 maschere: frame-banda di contesto sul totale dei validi 68% PSD
  veloce, 60% wavelet veloce, 8% PSD lenta, 3% wavelet lenta.

### Anti-collasso (per ramo e scala, sui target del batch)

EMA, stop-gradient e target normalizzati non bastano: un token del teacher quasi costante, normalizzato, diventa un target costante o rumore.
Si monitorano per ramo e scala: **diversita'** (media sulle coordinate della deviazione standard fra i token dei target normalizzati),
**rango effettivo centrato** della matrice dei target normalizzati e **varianza assoluta** dei token del teacher non normalizzati (media
sulle coordinate della varianza fra i token: la lezione di `run.py`, «rapporto e rango non vedono un'uscita quasi costante, le varianze
si'»). **Riferimento:** la media di ciascuna sui passi 1-10 (warmup, LR <= 1e-4). **Allarme e STOP** (congelato prima del collaudo): ogni 10
aggiornamenti dopo i primi 100, se per un ramo e una scala diversita' < 0,05 x riferimento, oppure varianza assoluta < 0,05 x riferimento,
oppure rango effettivo < 0,10 x riferimento, in **3 controlli consecutivi**. Un run fermato dall'allarme e' «collassato» e non batte il
riferimento.

## Bracci, dati, esposizione (famiglia B)

| Braccio | Obiettivo | Strati dei rami allenati | LR | Fonte |
|---|---|---|---|---|
| PW0 | nessuno (passo 0) | nessuno | | run del 2C, `checkpoint_init.pt` (gia' estratto) |
| PW | 32 target globali (2C) | tutti (token, `output`, `combine`) e teste | 1e-3 | run del 2C (gia' estratto) |
| PWh | nessuno (ibrido, CPU) | token di PW, `output` e `combine` di PW0 | | costruito dai checkpoint del 2C |
| **PWl** | 32 target globali | tutti e teste | **3e-4** | nuovo |
| **PWf** | 32 target globali | **token e teste; `output` e `combine` fissi al passo 0** | 1e-3 | nuovo |
| **PWm** | **predizione locale mascherata** | **token e predittori; `output` e `combine` fissi al passo 0** | 1e-3 | nuovo |

Teacher B_seed0 e B_seed1 (raw congelato), seme del run = seme del teacher, come il 2C. Stessi dati e configurazione del 2C: emg2qwerty,
stesso manifest, scale, durate e calibrazione (SHA congelati), batch 32, **3.000 aggiornamenti**, AdamW (weight decay 0,05, clip 1,0), warmup
100 poi costante; rami da 128 colonne, hidden 48. Gli strati fissi (frontend, statistiche d'ingresso, e per PWf/PWm `output` e `combine`; per
PWm anche le teste del vecchio obiettivo, non usate) sono esclusi dall'ottimizzatore e dall'EMA.

**Stessa inizializzazione del 2C, per costruzione:** ogni run nuovo esegue lo stesso percorso del 2C (calibrazione compresa, col suo
generatore) e poi **carica i tensori dei rami, le statistiche d'ingresso e le teste dal `checkpoint_init.pt` del 2C dello stesso teacher**
(SHA dalla provenienza del 2C); registra se la propria calibrazione coincideva (descrittivo). Per PWl e PWf la perdita del passo 1 (calcolata
prima del primo aggiornamento, quindi indipendente dall'LR e dagli strati fissi) deve coincidere con quella del 2C in `metrics.jsonl`
(tolleranza relativa 1e-4): **se non coincide, STOP**, nessuna ri-estrazione di ripiego. Controllo a fine run, studente e teacher: raw uguale
al teacher d'origine bit per bit; strati fissi, frontend, statistiche d'ingresso e teste fisse uguali al `checkpoint_init.pt` del 2C bit per
bit.

## Endpoint e lettura (si congelano col commit che precede il lancio)

Si estrae il **teacher EMA del checkpoint finale** (3.000 aggiornamenti), come nel 2C; nessuna selezione di checkpoint. Validation interna
della diagnostica del punto 1 (`spectral_diagnostic.py`: stesse cache raw, soggetti, fold e sonda; P1 pretraining, P2 fuori dai tre test),
insiemi PW e RPW (rami da soli e concatenati al raw). Confronti accoppiati per la famiglia B (due teacher), regola dei bracci e sua speculare
(«migliora» / «peggiora» / «regola non superata», che non esclude una differenza neanche oltre 1 punto).

- **Primario: PWf -> PWm su RPW** (confronto fra obiettivi a parita' di strati allenati, LR, dati, semi ed esposizione). **Positivo** =
  «migliora» in almeno un compito (P1 o P2) e **delta >= -0,01 sulla stima puntuale** nell'altro (il «non dannoso» del 2A/2C). Si legge
  **solo se** PWm ha imparato il suo compito (perdita normalizzata media sugli ultimi 100 aggiornamenti <= 0,9 con tutti e due i teacher) e
  nessuno dei due bracci e' collassato. Accanto si riporta PW0 -> PWm.
- **Descrittivi:** PWf -> PWm su PW; PW -> PWm (le ricette complete); PW0 -> PWm; PW -> PWl («a 3e-4 l'obiettivo vecchio: migliora /
  peggiora / regola non superata rispetto a 1e-3»; un LR piu' basso sugli stessi 3.000 passi e' anche meno spostamento dal passo 0); PW ->
  PWf (effetto di fissare `output` e `combine` col vecchio obiettivo); PW0 -> PWh e PWh -> PW (dove sta il cambiamento del vecchio
  obiettivo: token o strati finali); per braccio la distanza relativa ||theta - theta0|| / ||theta0|| degli strati dei token.
- **Limiti dichiarati:** un seme per teacher e due teacher di una sola famiglia (pilot, incertezza condizionata a quei modelli); piu'
  confronti senza correzione; asimmetria delle bande wavelet lente; le perdite dei due obiettivi non si confrontano; nessuna conclusione
  causale ne' adozione automatica; i test ufficiali P1/P2 non entrano (un'eventuale conferma ufficiale e' un passo successivo, con protocollo).

## Collaudo su dbg (PWm, B_seed0; criteri congelati prima)

Un job su `boost_qos_dbg` (1 A100, 30 minuti, limite del training 25 minuti), stesso codice e launcher dei run, cartella `collaudo/`; nessuna
sonda. Verdetto scritto dal job:
- **integrita':** nessuno STOP, raw e strati fissi uguali al passo 0 del 2C (studente e teacher), inizializzazione caricata dal 2C;
- **apprendimento:** perdita normalizzata media sugli ultimi 100 aggiornamenti <= 0,9;
- **monitor:** nessun allarme di collasso; valori di riferimento registrati;
- **copertura:** ogni ramo e scala ha almeno un bersaglio con contesto in almeno il 50% degli aggiornamenti; conteggi per banda registrati;
- **tempo:** (avvio + 3.000 x mediana dei secondi per aggiornamento dal 51esimo) x 1,25 + 10 minuti, arrotondato al quarto d'ora, e' l'allocazione
  dei run; **oltre 1 h 30 il collaudo non passa**;
- **stabilita' (descrittiva, come per il 2C dopo la decisione di Simone del 14/10):** la regola del collaudo del 2C sulla media mobile della
  perdita mascherata, riportata ma non bloccante.
Passa -> i sei run con l'allocazione misurata (nuovo via di Simone). Non passa -> stop e proposta (nessun retry automatico).

## Esecuzione e costo (stime; preventivo definitivo dopo il collaudo)

| Fase | Job | Risorse | Tempo massimo | Stima |
|---|---|---|---|---|
| Collaudo (PWm, B_seed0) | 1 | 1 A100, `boost_qos_dbg` | 30 min | |
| Training | 6 (PWl, PWf, PWm x 2 teacher) | 1 A100, 8 CPU, 100 GB, `boost_qos_lprod` | 1 h 30 ciascuno | PWl, PWf ~0,6 h; PWm ~0,8 h |
| Ibridi PWh | 1 | 1 CPU | 15 min | minuti |
| Uscite dei rami | 16 (8 modelli x P1/P2) | 8 CPU seriali | P1 1 h 30, P2 3 h | P1 ~54 min (misurato sulla seriale), P2 ~1 h 30 |
| Sonde interne (PW, RPW) | 16 | 8 CPU seriali | P1 4 h, P2 2 h | P1 ~2 h 15, P2 ~50 min |
| Confronto | 1 | 2 CPU | 30 min | minuti |

**GPU: tetto 9,5 GPU-ora (76 ore locali) = collaudo 0,5 + 6 x 1,5;** stima ~4,6. Dal budget del passo 6: 64,29 usate su 100 (33,08 prima del
2C + 29,32 del 2C + 1,89 della baseline casuale di GPT); dopo il tetto 73,79, **residuo 26,21**. **CPU: tetto ~674 core-ora** (uscite 8 x 4,5
h x 8 = 288; sonde 8 x 6 h x 8 = 384; confronto, ibridi e checkout ~1,4), stima ~350; ore locali CPU dell'account. **Calendario:** sulla
seriale gira un job da 8 core alla volta: ~43 h di uscite e sonde, dopo le ~30 h della diagnostica, con le stesse regole (al piu' 2 job
seriali dell'utente, `--nice`, nessuna tranche a ridosso delle valutazioni di C3 e C4). Disco: ~3 GB di checkpoint e uscite dei rami.

## Codice

- `model/spectral_branches.py`: `ScaleEncoder.tokens` (stessi token del forward, test d'uguaglianza bit per bit).
- `model/branch_masking.py`: maschere, ruoli dei frame per banda, segnale azzerato dello studente, predittore, obiettivo mascherato, monitor.
- trainer (`training/run.py`) e modello: obiettivo dei rami a encoder congelato (default: l'obiettivo vecchio, invariato bit per bit), strati
  fissi esclusi da ottimizzatore ed EMA, caricamento dei rami dal passo 0 del 2C, allarme di collasso dei rami. I campi nuovi della
  configurazione hanno un default retrocompatibile (`_config_with_defaults`); il controllo d'identita' della selezione di C1-C4 normalizza le
  due parti allo stesso modo (i run di C1-C2, fatti col codice 72cf24a, restano verificabili: test sui `config.json` veri).
- `scripts/branch_pilot.py`: training dei bracci nuovi (come `step2c.train`, identita' e controlli suoi, collaudo), ibridi, uscite dei rami
  (come `step2c.extract`); lettura con `spectral_diagnostic.py` esteso alle cache del pilot.
