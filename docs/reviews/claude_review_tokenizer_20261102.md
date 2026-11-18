# Review di Claude della proposta di GPT sul tokenizer (25/100 ms e GELU) — 02/11/2026

Richiesta: `docs/prompt_claude_review_tokenizer_20261102.md` (GPT, 532d177 su `origin/codex/r3-long-collaudo-20261028`), consegnata da
Simone: «Ti chiedo una review scientifica e tecnica della proposta di GPT sul tokenizer di WearUsFM, prima di provarla. Questa consegna e'
SOLO DI REVIEW: non modificare il codice, non lanciare job, training, estrazioni o nuovi esperimenti. Puoi rieseguire test CPU pertinenti.»
Oggetto: commit `c052792f0a796ba72d2804c81b3e5721a3010d93` (base `7e1c26f`). Letti con `git show`, senza merge e senza toccare il worktree di
GPT: `docs/tokenizer_window_nonlinear_protocol_20261102.md`, `docs/piano_scientifico_20261102.md`, i sei JSON di
`results/infra/frontend_window100_20261102/`, il diff dei cinque file di codice e dei tre di test, e i percorsi invariati che il diff tocca
di riflesso (`model/frontend.py`, `fm.py` encode e `zero_hidden_samples`, `local_encoder.py`, `perceiver.py`, `query_decoder.py`,
`training/masking.py`, `data/pretraining_loader.py`, `training/run.py` resume/branch/frozen). Il codice del modello e del training in
7e1c26f e' identico a quello del mio ramo (d93e22e, diff vuoto): il diff di GPT e' l'unica differenza fra i due rami su questi file.

**Nessun job, nessun dato, nessuna modifica al codice di runtime.** Eseguiti sul Mac: i test CPU di c052792 (export con `git archive`,
ambiente `wearusfm-tok`) e tre misure su segnali sintetici e sul generatore di maschere del repo, riproducibili con
`scripts/check_tokenizer_window_review.py` (aggiunto da questa review; solo rumore sintetico, nessun dataset). La review del critico
(`docs/reviews/tokenizer_window_nonlinear_review_20261102.md`) l'ho letta dopo aver scritto il giudizio: confronto in §9.

## 0. Verdetto: **concordo con modifiche**

Il codice fa cio' che il protocollo dichiara (§4) e gli 87 test che ho rieseguito passano. Il confronto, cosi' com'e' disegnato, ha pero'
quattro problemi che non riguardano il codice ma il disegno, e che si correggono prima del training con modifiche piccole (§8):

1. **Il raggio comune 2 non ha un meccanismo con l'obiettivo B.** Protegge il «supporto nominale» dei token nascosti, ma nello studente i token
   nascosti non sono mai chiavi (encoder locale e Perceiver) e le ancore si calcolano dal segnale grezzo, non dal teacher: il contenuto di quei
   token non arriva alla perdita qualunque sia il raggio. Il raggio cambia solo quali posizioni contano nella perdita: il compito non e' piu'
   quello di B, il B storico non e' piu' un riferimento, T25-L diventa un braccio obbligato invece di una replica di B, e il 15 % di maschere
   corte resta nel budget senza produrre quasi query (§3).
2. **La GELU lavora a un punto di lavoro non scelto.** Con il guadagno scalare calibrato su rumore sintetico, due terzi delle 64 risposte
   (spline e SIREN) hanno deviazione 0,07-0,3 a 1-2 kHz (0,1-0,6 a 200 Hz): li' la GELU e' quasi la retta 0,5·x, con un termine quadratico
   che vale da un decimo a un quarto di quello lineare; solo la famiglia di Fourier (deviazione 0,9-2,3) viene rettificata davvero. T100-G misura meno di quanto dichiara (§2.4).
3. **L'endpoint primario (posa a 2 kHz) e' poco legato all'ipotesi di Simone (200 Hz).** Misurato sul banco: a 200 Hz un token da 25 ms porta
   7-8 numeri indipendenti e uno da 100 ms 19-22; a 2 kHz sono gia' 36-40 contro 55-64. L'ipotesi «a 200 Hz 25 ms sono troppo pochi
   campioni» predice l'effetto piu' grande su EPN-612, che nel piano e' meta' di un endpoint secondario (§2.1, §6).
4. **I criteri usano un'incertezza che non e' quella della domanda.** Il bootstrap per utente misura la variabilita' degli utenti, non quella
   fra due pretraining a seme unico; il margine di 1 pp in P2 non e' raggiungibile con gli IC osservati finora (§6).

Il costo e' un quinto problema, aritmetico: tre bracci a 19.618 con margine e valutazioni superano il residuo documentato (§7). Con le
modifiche di §8 farei il confronto: permetterebbe di decidere se i bracci successivi (7a/7b) usano il tokenizer da 100 ms, se la GELU resta,
e se l'ipotesi dei 200 Hz regge su EPN-612, invece di un esito su posa a 2 kHz che non risponde alla domanda di partenza.

## 1. Cosa ho eseguito

| Cosa | Esito |
|---|---|
| `tests/cpu/test_frontend.py`, `test_frontend_window.py`, `test_tokenizer_activation.py`, `test_tokenizer_comparison.py`, `test_jepa.py` su c052792 | **87 passed in 28,0 s** (GPT ne dichiara 62 sui primi quattro). I due test DDP (`test_objective_arms -k torchrun`) non li ho ripetuti. |
| Rango effettivo del banco per fs e finestra, energia dei kernel fuori dalla finestra nominale | §2.1, tabella 1 |
| Deviazione per feature di `fe_gain · e(x)` (ingresso della GELU) su rumore a banda | §2.4, tabella 2 |
| Query sopravvissute con raggio 2 rispetto a raggio 0, per tipo di maschera, montaggio e durata | §3.2, tabella 3 |

## 2. L'ipotesi e la posizione della GELU (domanda 1)

### 2.1 Quanto porta davvero una finestra da 100 ms: il rango del banco

Il token e' lineare nel segnale a pesi fissati: `e = K x Δt`, con `K` (64 × istanti del supporto) i kernel filtrati alla Nyquist di quella fs
(`frontend.kernels_at`). Il numero di numeri indipendenti che un token puo' portare e' il rango di `K`, non 64 e non il numero di campioni.

**Tabella 1.** Rango di `K` (valori singolari sopra 1e-2 e 1e-3 del primo), pesi iniziali (seme 0); energia media dei kernel fuori dalla
finestra nominale e oltre 0,5 / 1,5 / 2,5 patch dal centro della patch (cioe' fuori dalla regione nascosta con raggio 0 / 1 / 2).

| Finestra | fs | Campioni nominali | Supporto numerico | Rango 1e-2 | Rango 1e-3 | Per famiglia (F/S/M, 1e-2) | Fuori nominale | Oltre 1,5 patch | Oltre 2,5 patch |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 25 ms | 200 Hz | 5 | 46 | **7** | 8 | 7 / 6 / 6 | 3,7 % | 0,5 % | 0,1 % |
| 25 ms | 1 kHz | 25 | 226 | 23 | 27 | 22 / 16 / 9 | 0,1 % | 0,0 % | 0,0 % |
| 25 ms | 2 kHz | 50 | 451 | 36 | 40 | 24 / 16 / 9 | 0,0 % | 0,0 % | 0,0 % |
| 100 ms | 200 Hz | 20 | 62 | **19** | 22 | 18 / 19 / 13 | 0,2 % | 0,9 % | 0,1 % |
| 100 ms | 1 kHz | 100 | 302 | 53 | 63 | 24 / 24 / 16 | 0,0 % | 0,3 % | 0,0 % |
| 100 ms | 2 kHz | 200 | 601 | 55 | 64 | 24 / 24 / 16 | 0,0 % | 0,3 % | 0,0 % |

Lettura.
- L'intuizione di Simone e' giusta e si quantifica: **a 200 Hz un token da 25 ms porta 7-8 numeri indipendenti** (non 5: le code
  dell'anti-aliasing ne aggiungono due o tre), uno da 100 ms ne porta 19-22. A 1 kHz il rapporto e' 23 contro 53, a 2 kHz 36 contro 55.
- Quindi la finestra lunga e' una modifica grande per EPN-612 (200 Hz) e UCI (1 kHz) e una modifica modesta per emg2pose, NinaPro e Hyser
  (2 kHz e oltre), dove il banco da 25 ms ha gia' 36-40 gradi di liberta' e il Perceiver puo' integrare nel tempo attraverso il backbone.
  **L'endpoint primario del piano (posa su emg2pose a 2 kHz) sta dove l'ipotesi predice l'effetto piu' piccolo.** Non e' un motivo per non
  misurare la posa; e' un motivo per non chiamarla l'unico primario (§6).
- Il «supporto numerico di 300 ms» e' un limite superiore, non dove sta l'energia: fuori dalla finestra nominale c'e' il 3,7 % dell'energia
  a 200 Hz per il banco corto e lo 0,2 % per quello lungo; oltre 2,5 patch dal centro (il bordo del raggio 2) resta lo 0,1 % in entrambi.
  Le code non sono un problema pratico per nessuno dei due raggi.

### 2.2 I 5.184 coefficienti in piu'

Il conto e' giusto (Fourier 24 × (208 − 52) = 3.744, spline 24 × (80 − 20) = 1.440). Ma a 200 Hz il passa-basso a 90-100 Hz azzera le
armoniche sopra 100 Hz: dei 104 coefficienti per uscita ne restano attivi una decina. La capacita' effettiva in piu' del banco lungo e' quella
della tabella 1 (+12 dimensioni a 200 Hz, +19 a 2 kHz), non 5.184 parametri. Il confronto T100-L − T25-L resta «finestra piu' regola di scala
delle basi», come il protocollo dichiara; un controllo a coefficienti identici non e' possibile con questa regola e non lo chiederei.

### 2.3 GELU dopo l'integrazione, rettificazione prima dell'integrazione, encoder convoluzionale

Tre cose diverse, da tenere separate come chiede il prompt.

- **GELU sulle risposte integrate (T100-G):** `y_j = GELU(g · ∫ k_j x)`. Un filtro lineare per kernel, poi una non linearita' puntuale a 40
  Hz. E' un banco di filtri apprendibili con rettificazione a semionda: per la famiglia di Fourier, la parte reale o immaginaria di un
  coefficiente di Fourier finestrato, tenuta se positiva e quasi azzerata se negativa. Le cancellazioni dentro i 100 ms avvengono prima; non e'
  un'energia ne' un RMS. Il modulo di un coefficiente si ottiene solo se il banco contiene anche il kernel opposto (`-k_j`): all'inizio non
  c'e' (coseno e seno non sono opposti), i kernel sono apprendibili e possono formare coppie di segno, al prezzo di dimezzare le 64 uscite
  utili. Nessun parametro in piu' e nessuna modifica al contesto temporale: e' il passo minimo possibile.
- **Rettificazione o quadratura prima dell'integrazione:** `∫ k_j |x|`, `∫ k_j x²` oppure `|∫ h_j x|²` con `h_j` a banda stretta (energia per
  banda nella finestra). Sono le feature classiche dell'EMG (MAV, RMS, potenze per banda) ed esattamente cio' che le ancore usano come
  bersaglio. E' probabilmente quello che Simone intende con «calcolare le feature su finestre da 100 ms». Con il banco attuale la versione
  minima sarebbe il modulo quadrato delle coppie coseno/seno di Fourier (12 energie per banda) con compressione logaritmica; cambia la natura
  del token (positivo, quadratico in ampiezza, serve un pavimento al logaritmo) e la sua inizializzazione. Non e' minimo e non e' quello che
  GPT ha preparato: lo lascerei come passo successivo se la GELU non mostra nulla.
- **Encoder temporale convoluzionale non lineare** (stile wav2vec 2.0: convoluzioni, normalizzazione, GELU, passo fino a 40 Hz): piu'
  capacita', centinaia di migliaia di parametri, ma legato a una frequenza di campionamento. Romperebbe la proprieta' che regge tutto il
  frontend attuale, gli stessi pesi a ogni fs con anti-aliasing per fs (gate D8): richiederebbe il ricampionamento del corpus. Non e' una
  modifica del tokenizer, e' un altro modello.

**Vantaggio plausibile della posizione scelta da GPT.** Oggi la prima operazione non lineare che mescola canali e' l'attenzione del primo
strato locale, i cui valori sono lineari in `LN(token)`: il primo mescolamento spaziale avviene su forme d'onda con segno, che fra canali
non in fase si cancellano in parte, e la prima trasformazione puntuale per canale e' la MLP dopo quell'attenzione. Con la GELU prima della
proiezione, il token porta gia' una versione rettificata delle risposte: il primo strato locale, il bias geometrico e il Perceiver aggregano
quantita' simili ad ampiezze invece di forme d'onda. E' il motivo per cui gli encoder su forma d'onda grezza (wav2vec 2.0, SincNet, LEAF)
mettono una non linearita' subito dopo il primo banco di filtri. Il vantaggio e' plausibile, piccolo, e condizionato al punto di lavoro: cio'
che segue.

### 2.4 Il punto di lavoro della GELU (difetto)

`fe_gain` e' uno scalare, calibrato alla costruzione su rumore a banda 20-450 Hz a 2 kHz perche' la deviazione **globale** delle 64 uscite sia 1
(`patch_tokens._calibrate_gain`). Con l'identita' questo basta: la proiezione lineare e la LayerNorm del primo strato assorbono qualunque
scala per feature. Con la GELU no: la GELU e' una funzione fissa di `x`, e il suo effetto dipende dalla scala di ciascuna feature.

**Tabella 2.** Deviazione per feature di `fe_gain · e(x)` su rumore a banda 20-450 Hz adattata alla Nyquist (lo stesso rumore della
calibrazione), senza bordi; `|GELU(x) − 0,5·x| / |x|` misura quanto la GELU si discosta dalla retta di pendenza 0,5 (0 = lineare).

| Finestra (fe_gain) | fs | Quantili 10 / 50 / 90 delle deviazioni | Fourier min/med/max | Spline min/med/max | SIREN min/med/max | Scostamento dalla retta |
|---|---:|---|---|---|---|---:|
| 25 ms (982) | 2 kHz | 0,30 / 0,59 / 1,61 | 1,05 / 1,43 / 2,14 | 0,28 / 0,43 / 0,72 | 0,22 / 0,40 / 0,75 | 0,35 |
| 100 ms (549) | 200 Hz | 0,19 / 0,44 / 1,75 | 0,92 / 1,49 / 2,28 | 0,18 / 0,38 / 0,61 | 0,10 / 0,21 / 0,43 | 0,35 |
| 100 ms (549) | 1 kHz | 0,09 / 0,27 / 1,71 | 1,26 / 1,57 / 2,02 | 0,20 / 0,26 / 0,32 | 0,07 / 0,09 / 0,18 | 0,37 |
| 100 ms (549) | 2 kHz | 0,09 / 0,28 / 1,70 | 1,20 / 1,60 / 2,03 | 0,21 / 0,27 / 0,33 | 0,07 / 0,09 / 0,18 | 0,37 |

Per `|x|` piccolo, `GELU(x) ≈ 0,5·x + 0,4·x²`: a deviazione 0,1 il termine quadratico vale un decimo di quello lineare, a 0,3 un quarto. Nel
banco da 100 ms **40 uscite su 64 (spline e SIREN) stanno in quel regime** a 1-2 kHz, e poco sopra a 200 Hz: per loro T100-G e' quasi T100-L
con la proiezione dimezzata. La rettificazione agisce sulle 24 uscite
di Fourier. Il punto di lavoro non e' scelto da nessuno: e' il risultato di una calibrazione su rumore sintetico pensata per un'altra cosa
(il peso del contenuto rispetto al bias all'inizializzazione). Su dati reali le scale per dataset e per fs saranno diverse ancora (a 200 Hz il
rumore adattato alla Nyquist gia' sposta le spline da 0,27 a 0,38). I kernel sono apprendibili e in principio il training puo' spostare le
scale, ma con LR 1e-4 e weight decay 0,05 in 5.000-19.618 passi e' una speranza, non un disegno.

Il protocollo dice «la diversa distribuzione delle risposte e' parte dell'intervento: media, deviazione, rango e gradienti iniziali sono
registrati, non corretti». Registrare e' giusto; ma cosi' il braccio non misura «una non linearita' sulle risposte dei filtri», misura «una
non linearita' su un terzo dei filtri, dove capita». Il test D8 dopo GELU (stesso segnale a 2 kHz e 200 Hz entro il 2 %) e' corretto e non
riguarda questo punto: e' per costruzione, perche' le feature lineari coincidono e la GELU e' puntuale.

### 2.5 Modifica minima che propongo

**Standardizzazione fissa per feature prima dell'attivazione, in tutti e tre i bracci:** media e deviazione delle 64 risposte dopo
`fe_gain`, misurate una volta sui batch train del collaudo (identici per i tre bracci, mai sulla val), salvate come buffer nel checkpoint
come oggi `fe_gain`, applicate come `(e − μ) / σ` prima della GELU (T100-G) o dell'identita' (T25-L, T100-L). E' quello che fanno gli encoder
su forma d'onda (wav2vec 2.0: normalizzazione per canale prima della GELU del primo strato). Nessun parametro, nessun consumo di RNG. Per i
bracci lineari una trasformazione affine fissa prima di una `Linear` e' una riparametrizzazione: la funzione rappresentabile non cambia,
cambia solo marginalmente l'ottimizzazione (AdamW e' quasi invariante a una scala per ingresso). Cosi' T100-G misura davvero «GELU a scala
unitaria su tutte le 64 risposte» e T100-L/T100-G restano a parametri e supporto identici.

Se GPT preferisce non toccare i bracci: ridurre la pretesa dell'ipotesi nel protocollo («GELU al punto di lavoro della calibrazione
sintetica, che rettifica la sola famiglia di Fourier»), pre-registrare la tabella 2 misurata sui batch reali come parte della lettura, e
accettare che un esito nullo di T100-G non dica nulla sulla non linearita' in se'. Non aprirei una ricerca di attivazioni o larghezze.

## 3. Validita' del confronto (domanda 2)

### 3.1 Il raggio 2 non ha un meccanismo con l'obiettivo B

Che cosa fa il raggio: `core_hidden` (`training/jepa.py:121-132`) restringe l'insieme delle **query** della perdita ai token nascosti con
almeno `radius` patch nascoste per lato. Non cambia l'ingresso dello studente (`zero_hidden_samples`, `fm.py:139-156`, azzera i campioni
grezzi di tutte le patch nascoste, qualunque sia il raggio), non cambia le maschere, non cambia i bersagli.

Perche' esisteva (R5): con JEPA acceso il bersaglio e' il **teacher**, che vede tutto il segnale; il suo token al bordo di un buco contiene
campioni visibili e predirlo e' banale. Con JEPA = 0 quel bersaglio non c'e'. I bersagli delle ancore vengono dal segnale grezzo
(`gather_anchor_targets`), la posa dalle annotazioni. Resta la domanda: lo studente puo' «imbrogliare» attraverso il proprio token nascosto
al bordo, che con una finestra da 100 ms legge 37,5 ms di campioni visibili? No, per costruzione: un token nascosto **non e' mai chiave** per
altri canali nell'encoder locale (`local_encoder.py:98-101`, se stesso soltanto), e' escluso dalle chiavi del Perceiver
(`perceiver.py:57-58`), il backbone lavora sui latenti e il decoder interroga per identita' di canale e istante, non per contenuto del token
(`query_decoder.py`). Il contenuto dei token nascosti dello studente **non raggiunge la perdita** con nessun raggio. Il «supporto nominale»
che il raggio 2 protegge e' una proprieta' di un tensore che nessuna perdita legge.

Con JEPA acceso in un braccio futuro il raggio tornerebbe necessario (bersaglio del teacher); con RVQ i codici vengono dal tokenizer
congelato sul grezzo, come le ancore. Questa conclusione vale per l'obiettivo B e va scritta cosi'.

### 3.2 Cosa cambia nel compito con il raggio 2 (stima sintetica)

**Tabella 3.** Generatore di maschere del repo (`training.masking.generate_mask`), quote storiche di D10 (0,15 / 0,25 / 0,20 / 0,40, ratio
0,5), RVQ spento come in B, 200 estrazioni per riga; «query r2» = patch nascoste che restano query con raggio 2.

| Montaggio | Patch | Nascoste | Query r2 / nascoste | Corte | Medie | Lunghe | Spaziali | Finestre senza query |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EPN myo8 (8 ch) | 40 / 80 / 160 | 51-52 % | 77-79 % | 24-25 % | 67-69 % | 87-95 % | 100 % | 0 % |
| emg2pose (16 ch) | 40 / 80 / 160 | 51 % | 77-80 % | 22-25 % | 68 % | 87-95 % | 100 % | 0 % |
| NinaPro DB2 (12 ch) | 40 / 80 / 160 | 51 % | 78-80 % | 25-27 % | 68-69 % | 87-95 % | 100 % | 0 % |
| CapgMyo (128 ch) | 40 / 80 / 160 | 51 % | 79-81 % | 27-28 % | 69-70 % | 87-95 % | 100 % | 0 % |

Dunque: nessuna finestra resta senza query (bene, la riduzione per finestra non incontra finestre vuote), ma **un quinto delle patch nascoste
esce dalla perdita, e tre quarti delle patch delle maschere corte**: il 15 % del budget di mascheramento continua a togliere segnale allo
studente senza quasi insegnargli nulla. Per R5 proprio per questo Simone aveva firmato `share_short` = 0 con quote 0,294 / 0,235 / 0,471
(decisioni.md, 24/10/2026: «Firmo la regola di lettura del pilot (testo del §5) e lo scostamento da D10 per R5»); il piano attuale tiene 0,15
con raggio 2, in contraddizione con quel precedente. L'erosione e'
piu' mite di quanto il protocollo scriva («i buchi da 1-2 patch non producono query core, quelli da 4-12 ne conservano 0-8»), perche' i tubi si
sovrappongono e le corte non restano mai da sole; la misura sui batch reali resta necessaria (canali QC, durate reali, montaggi virtuali).

Effetti del raggio 2, riassunti: (a) T25-L non e' B: cambia il compito delle ancore (query solo dentro i buchi, niente interpolazione a corto
raggio); (b) il B storico, con tutta la sua catena di valutazioni (E1-E3, sonde, PSD-3, PCA, temporale, slot), smette di essere un
riferimento; (c) serve un terzo training (4,24 GPU-ora a 5.000, 16,6 a 19.618); (d) nessuno dei tre bracci e' confrontabile con R3c o con
i bracci futuri 7a/7b a raggio 0, a meno di portare il raggio 2 anche li'.

### 3.3 Proposta: raggio 0 in tutti i bracci, T25-L come replica di B

Raggio 0 per T25-L, T100-L e T100-G. Le query sono tutte le patch nascoste, identiche fra bracci (dipendono solo dalle maschere), numeratori e
denominatori identici: la parita' che GPT vuole c'e' lo stesso. In piu':

- il compito e' esattamente quello di B, e **T25-L diventa una replica di B** (stessa configurazione, stesso seme, stesso stream): la
  differenza T25-L − B a 5.000 e a 19.618 misura la variabilita' fra due run identici (non determinismo GPU piu' eventuali differenze di
  checkout), **il dato che manca a tutta la campagna** per leggere qualunque delta fra pretraining a seme unico (§6);
- B5k e B19618 restano confrontabili con i nuovi bracci, con la loro catena di valutazioni;
- se il budget non copre tre bracci a 19.618 (§7), il piano puo' dichiarare prima del training il ripiego: T25-L fermo a 5.000 (replica e
  pavimento di rumore allo screening), T100-L e T100-G a 19.618 con **B19618 come riferimento**, previa estrazione delle feature del teacher di
  B con la stessa pipeline dei nuovi bracci (circa 1-2 GPU-ora).

Costo nel codice: `masked_query_radius` (`jepa.py:98-103`) oggi rifiuta un raggio sotto `core_radius`, quindi con `window_ms` 100 il raggio 0 e'
vietato. La soglia e' giustificata solo dal bersaglio del teacher: va ammesso un override esplicito sotto `core_radius` quando `jepa_weight`
e' 0 (e registrato nel config), con il test `test_override_cannot_weaken_nominal_kernel_protection` condizionato a JEPA acceso. E' una
modifica di poche righe, di GPT.

Se GPT tiene il raggio 2: va scritto che il compito non e' piu' quello di B, va portato `share_short` a 0 con le quote rinormalizzate come per
R5 (scostamento da far firmare), e va accettato che la catena storica di B non e' piu' un riferimento. Non e' sbagliato; e' piu' caro e meno
informativo.

### 3.4 Quote raw, coefficienti, obiettivo B, R5

- Quote raw mantenute nel config: si'; le quote effettive nella perdita sono quelle della tabella 3 con raggio 2, uguali a quelle raw con raggio 0.
- Coefficienti del banco lungo: §2.2; dichiarato, accettabile.
- Obiettivo B invariato: giusto, una modifica per volta. Nota che in B la posa (peso 1) domina le ancore (0,2): il tokenizer viene selezionato
  soprattutto dalla regressione supervisionata su emg2pose a 2 kHz, cioe' dai dati meno sensibili alla finestra (tabella 1).
- R5 (150 ms, raggio 3, JEPA 1 + ancore 0,2, `share_short` 0): P1 −0,53 [−1,07; −0,02], P2 +0,35 [−2,14; +2,68] a 2 epoche; **su EPN-612
  −2,45 [−2,97; −1,92] (pool) e −2,89 [−3,44; −2,40] (dec)**, su DB6 +1,25 e +1,57. Il precedente dice il contrario dell'ipotesi «a 200 Hz
  serve una finestra lunga»: a 200 Hz R5 era peggiore, a 2 kHz migliore. Obiettivo, raggio e quote diverse impediscono di trasferire il
  numero, non il segno del rischio: e' il motivo principale per cui EPN-612 va letto come endpoint, non come dettaglio.

### 3.5 Controlli sui batch train prima di spendere il budget

Sullo stesso insieme di batch (≥ 64 batch da 32 finestre, stesso seme, stesso stream), per i tre bracci e per B a raggio 0:

1. hash dei batch consegnati, delle maschere e dei bersagli: identici fra bracci (gia' previsto);
2. conteggi delle query e denominatori per tipo di maschera, dataset, fs e componente della perdita: identici fra bracci; e, se si tiene il
   raggio 2, la differenza rispetto a raggio 0 (la tabella 3 misurata sui dati veri);
3. **media e deviazione per feature di `fe_gain · e(x)` per dataset e fs** (tabella 2 sui dati veri), e rango effettivo del banco per fs:
   e' il controllo che decide se la GELU lavora dove si vuole, e la base per la standardizzazione di §2.5;
4. perdita iniziale per componente e norma dei gradienti di `frontend.*` e `proj` al passo 0 e dopo 100 passi, nei tre bracci: che i kernel
   si muovano davvero (ampiezza relativa dell'aggiornamento), non solo che il gradiente sia finito;
5. quota di finestre a 200 Hz e a 1 kHz nello stream di training: se e' piccola, l'effetto a 200 Hz si misura su una frequenza che il
   tokenizer vede di rado in training, e la lettura su EPN-612 va scritta con questo limite;
6. memoria e secondi per passo per braccio (previsto): il frontend lungo costa il 19 % in piu' di moltiplicazioni a 2 kHz
   (`PatchTokenizer.flops`: 601 contro 451 campioni di supporto), su una parte che e' meno dell'1 % del modello; l'effetto sul costo del
   passo dovrebbe essere trascurabile, da confermare.

## 4. Il codice implementa quanto dichiarato? (domanda 3)

Si'. Verifiche puntuali, tutte sul tree di c052792:

| Punto | Dove | Esito |
|---|---|---|
| Posizione della GELU | `patch_tokens.py:67-70`: dopo `f * fe_gain` e il cast a float32, prima di `proj` | come dichiarato; GELU in float32 |
| Gradienti ai filtri | GELU puntuale nel grafo; test su Fourier, spline, SIREN e `proj` con gradienti finiti e non nulli | ok |
| Calibrazione del gain | invariata, globale, prima dell'attivazione; `fe_gain` identico fra identity e gelu | ok (limite in §2.4) |
| Scala delle basi | `window_kwargs(25, 100)` = 104 armoniche, 80 nodi, omega 24; 5.184 coefficienti in piu' | ok |
| Default storico | `FMConfig.tokenizer_activation = "identity"`, `JEPAConfig.masked_loss_radius = None`, `_config_with_defaults` aggiunge entrambi senza mutare il dizionario; flag CLI opzionali | ok, bit per bit nei test |
| Masking prima del frontend | `fm.py:199-203`: `zero_hidden_samples` poi `tokenizer`; test con perturbazioni dei campioni nascosti a 200/1000/2000 Hz | ok |
| Limiti del prefiltraggio | `pretraining_loader._filter`: Butterworth 4 zero-fase 20-min(450, 0,45 fs) e notch Q 30 a 50/60 Hz, applicati **prima** del mascheramento; la risposta del notch dura ~0,2 s | i campioni nascosti influenzano quelli visibili vicini in tutti i bracci e in B; limite comune, non asimmetria |
| Conteggi e perdita | `loss_counts` e `_jepa_compute` usano lo stesso `masked_query_radius`; test: `counts == terms.den` per chiave, anche con JEPA 0 | ok |
| Supporto posa | `pose_time_valid(enc.patch_valid, counts)`: indipendente dal raggio | ok |
| Resume / branch / frozen | `RESUME_OPERATIONAL` non contiene `model` ne' `jepa`; `_check_branch` confronta tutto tranne schedule e `max_steps`; `_load_frozen_encoder` confronta il modello escluso `BRANCH_MODEL_FIELDS`, che non contiene `tokenizer_activation` | ok, con test |
| Validazione precoce | `train()` chiama `masked_query_radius` prima di dati e modello; la CLI valida `--masked-loss-radius` dopo `--window-ms` | ok |

Osservazioni minori, nessuna bloccante:
- La soglia `radius >= core_radius` e' la sola cosa da cambiare per §3.3; oggi e' un vincolo senza meccanismo con JEPA 0.
- `--masked-loss-radius` omesso con `--window-ms 100` da' raggio 2 in silenzio (il default nominale): coerente con R5, ma va detto nel log
  del run (c'e': «raggio query ... patch»).
- I checkpoint nuovi portano `tokenizer_activation` nel config: un checkout **precedente** a c052792 (per esempio quelli delle valutazioni
  PSD-3 e slot) li rifiuta in `FMConfig(**...)`. Le valutazioni dei nuovi bracci vanno fatte da un checkout ≥ c052792; i vecchi checkpoint
  restano leggibili dal nuovo codice.
- Il test D8 dopo GELU prova la coerenza fra fs dello **stesso** segnale; non dice nulla sulle scale fra dataset (§2.4), e non pretende di dirlo.
- `flops()` non conta la GELU: dichiarato, trascurabile.

Non ho trovato difetti di implementazione che renderebbero sbagliati i risultati dei tre bracci come configurati.

## 5. Dose e valutazione (domanda 4)

- **5.000 poi 19.618 con ripresa dal proprio checkpoint:** e' la pratica con cui B19618 e' stato fatto (17 job in ripresa);
  `_check_resume_recipe` ammette solo `max_steps` e campi operativi, lo stato dello stream e' nel checkpoint, lo schedule WSD e' funzione del
  passo globale (`total_steps` 49.046, decay dal 39.237): la continuazione e' identica a un run non interrotto. Concordo che un esito nullo a
  5.000 non chiuda nulla; il piano deve dire gia' ora che a 5.000 si decide solo per cause tecniche (instabilita', allarmi di rango, NaN), che
  tutti i bracci proseguono insieme, e qual e' il ripiego di budget (§7) se non proseguono tutti.
- **Teacher per regressione e classificazione:** coerente (le sonde P1/P2 lo usavano gia'); per la posa cambia la sorgente rispetto a E3,
  temporale e slot (student), come GPT scrive. Con la momentum 0,996 il teacher e' una media del passato con costante di 250 passi: a 5.000
  e 19.618 e' quasi lo student. Va bene; costa estrazioni nuove per ogni riferimento, B compreso se B resta il controllo (§3.3).
- **Bordo comune 6 patch:** 150 ms per lato. Dall'energia dei kernel (tabella 1) 4 patch basterebbero anche al banco lungo; 6 costa 4 istanti
  per finestra in piu' e toglie poco. Una delle due, fissata prima: 6 e' difendibile per simmetria con il supporto numerico dichiarato.
- **Angoli centrali a 25 ms, nessuna media:** giusto, e' l'unico modo di misurare la risoluzione. Avvertenza da scrivere prima: la finestra di
  Hann da 100 ms ha una larghezza a meta' altezza di 50 ms; T100 puo' migliorare MAE e ampiezza e peggiorare `E_dyn` sulle variazioni rapide
  (e' cio' che R5 mostrava: «cambia la risoluzione temporale piu' che la qualita' media»). La tabella di lettura del protocollo lo prevede.
- **Soglia dinamica storica:** mantenerla.
- Testa student come diagnostica separata: d'accordo; non seleziona nulla.

## 6. Metriche e criteri (domanda 5)

1. **Endpoint primario.** `E_dyn` sulla posa resta un primario legittimo per la domanda «100 ms prima del pooling aiuta la rappresentazione»,
   ma non per la domanda di Simone. Propongo un **co-primario congelato: la balanced accuracy su EPN-612** (gia' calcolata dentro P2) per il
   contrasto T100-L − T25-L, con il bootstrap appaiato per soggetto gia' previsto, e la regola scritta prima: l'ipotesi «a 200 Hz 25 ms sono
   troppo pochi» e' sostenuta solo se il delta su EPN-612 e' positivo con l'IC sopra lo zero; e' contraddetta se l'IC sta sotto lo zero (come
   per R5); altrimenti non risolta. UCI (1 kHz) secondario con la stessa lettura. Senza questo, un esito nullo sulla posa a 2 kHz verra'
   letto come «la finestra non serve», e non e' quello che il confronto avrebbe misurato.
2. **Il 2 % relativo con IC superiore sotto zero.** Il bootstrap per utente misura l'incertezza degli utenti a checkpoint fissati: nel lettore
   temporale la sua risoluzione era ±0,0022 su 0,86 (±0,25 % relativo). Fra due pretraining diversi a seme unico la variabilita' e' ignota e
   quasi certamente piu' grande: B e R3c distano il 15 %. Con questa regola un effetto del 2 % «significativo» puo' essere rumore di run. Il
   rimedio che non costa nulla e' la replica di §3.3: dichiarare **segnale utile** solo se la riduzione relativa e' ≥ 2 %, l'IC appaiato sta
   sotto lo zero **e** l'effetto supera in modulo la differenza T25-L − B allo stesso passo (il pavimento di rumore). Se la replica non si fa,
   la soglia del 2 % va tenuta per quello che e' (una grandezza minima pratica) e il protocollo deve dire che il confronto non distingue un
   effetto architetturale dal rumore fra run.
3. **Non inferiorita' a 1 pp in P1 e P2.** Con gli IC osservati (B − R3c: P1 [−1,10; −0,02], semiampiezza 0,55; P2 [−2,96; +0,65],
   semiampiezza 1,8), in P2 «limite inferiore > −1» richiede una stima puntuale sopra +0,8 pp: in pratica chiede un miglioramento. Scriverlo
   prima: P1 decide la non inferiorita' a 1 pp; P2 si riporta, e si dichiara **danno** solo se la stima e' sotto −1 pp con l'IC superiore
   sotto lo zero; «conservazione non dimostrata» in P2 e' l'esito atteso anche per un braccio neutro e non va letto contro il candidato.
   L'alternativa, un margine di 2 pp per P2, e' altrettanto arbitraria; una delle due, congelata ora.
4. Ridge sul solo train, sonde convergenti senza PCA, bootstrap appaiato con lo stesso seme: d'accordo. Val gia' osservata, seme unico, posa
   supervisionata: i limiti sono scritti bene nel protocollo; aggiungere che il co-primario su EPN-612 e' una classificazione interna come
   P1/P2, non un test intatto.

## 7. Costo, budget e cosa manca (domanda 6)

Aritmetica verificata: B5k 3.814 s su 4 GPU = 4,2378 GPU-ora per 5.000 passi; tre bracci 12,71 a 5.000 e 49,88 a 19.618 (incluso lo
screening). B19618 e' costato 14,62 GPU-ora per 14.618 passi di training (12,39 al ritmo di B5k) piu' le sue valutazioni: **circa 2,2 GPU-ora
di estrazioni, valutazioni e overhead per un modello a un endpoint**, la cifra che manca al preventivo.

| Scenario | Training | Con margine 15 % | Valutazioni e preflight (stima) | Totale | Residuo su 63,29 |
|---|---:|---:|---:|---:|---:|
| Tre bracci a 19.618 (piano GPT) | 49,9 | 57,4 | ≈ 3 × 2,2 + 1,5 (screening) + 1 (preflight) ≈ 9 | ≈ 66 | **negativo** |
| Tre bracci a 5.000 (solo screening) | 12,7 | 14,6 | ≈ 1,5 + 1 | ≈ 17 | ≈ 46 |
| Raggio 0: T25-L a 5.000, T100-L e T100-G a 19.618, riferimento B19618 | 4,2 + 33,3 = 37,5 | 43,1 | ≈ 2 × 2,2 + 1,5 + 1,5 (teacher di B5k e B19618) + 1 ≈ 8 | ≈ 51 | ≈ 12 |

Le cifre delle valutazioni sono stime da B19618, non timing; il margine del 15 % copre code e ripartenze, non il frontend lungo (che costa
poco, §3.5). Il piano GPT a tre bracci completi non entra nel residuo documentato una volta aggiunte le valutazioni: GPT stesso lascia 5,9
GPU-ora dopo il margine. Il terzo scenario entra con margine e conserva la replica di B allo screening; e' il ripiego che scriverei nel piano
prima di partire, con la regola «T25-L prosegue a 19.618 solo se dopo lo screening il residuo stimato resta sopra X», X fissato ora.

**Vale il costo?** Il confronto spende tra 50 e 57 GPU-ora dei 63,29 documentati (80-90 % del residuo) su un seme. Con il disegno attuale
compra: un delta su posa a 2 kHz dove l'effetto atteso e' piccolo, un delta P1/P2 letto con margini che P2 non puo' soddisfare, e nessuna
stima del rumore fra run. Con le modifiche di §8 compra: la risposta all'ipotesi dei 200 Hz su EPN-612, la risposta su posa e P1/P2 con il
pavimento di rumore di una replica, la scelta del tokenizer per 7a/7b, e il riferimento B conservato. A quel punto si', lo farei: la
decisione che permette e' «i bracci successivi usano 25 o 100 ms, con o senza GELU», e si prende una volta sola.

Cosa manca per collaudo e preventivo completo, gia' previsto da GPT e non in discussione: supporti e hash sui batch reali; secondi per passo e
memoria per braccio; valutatori congelati (ridge, sonde, bootstrap) con i loro costi; tetti di allocazione per job; checkout di campagna da
SHA pubblicato; sottomissione da semplicione con ledger; stima delle estrazioni del teacher per ogni checkpoint conservato.

## 8. Consegna

**Difetti dimostrati** (nel disegno; nessuno nel codice):
- D1. Il raggio 2 non protegge nulla con l'obiettivo B (token nascosti mai chiavi, ancore dal grezzo, posa indipendente) e cambia il compito
  rispetto a B: circa un quinto delle patch nascoste e tre quarti delle corte escono dalla perdita (tabella 3); il B storico esce dai
  riferimenti; `share_short` 0,15 resta in contraddizione con lo scostamento firmato per R5.
- D2. Punto di lavoro della GELU non controllato: 40 risposte su 64 con deviazione 0,07-0,3 restano quasi lineari (tabella 2); T100-G misura
  «GELU sulla famiglia di Fourier al punto di lavoro di una calibrazione sintetica».
- D3. Endpoint primario poco sensibile all'ipotesi: la finestra lunga vale 7-8 → 19-22 gradi di liberta' a 200 Hz e 36-40 → 55-64 a 2 kHz
  (tabella 1); il primario e' a 2 kHz, EPN-612 e' dentro un secondario; R5 su EPN-612 era −2,45 pp.
- D4. Criteri: IC per utente usato come incertezza di un effetto fra pretraining a seme unico; margine di 1 pp in P2 non raggiungibile con gli
  IC osservati (semiampiezza ≈ 1,8 pp).
- D5. Preventivo: tre bracci a 19.618 con margine e valutazioni (≈ 2,2 GPU-ora per modello-endpoint da B19618) superano il residuo di 63,29.

**Correzioni indispensabili prima del training:**
- C1. Decidere il raggio con il suo meccanismo scritto: raggio 0 nei tre bracci (replica di B in T25-L; override sotto `core_radius` ammesso
  con `jepa_weight` 0 e test aggiornato), oppure raggio 2 con `share_short` 0 e quote rinormalizzate come R5, dichiarando che il compito non e'
  quello di B e che B non e' un riferimento.
- C2. Co-primario congelato su EPN-612 (T100-L − T25-L, bootstrap per soggetto, regola a due lati scritta ora); UCI secondario.
- C3. Standardizzazione fissa per feature prima dell'attivazione in tutti i bracci (buffer da batch train, mai dalla val), oppure riduzione
  esplicita della pretesa dell'ipotesi di T100-G con la tabella 2 misurata sui dati veri come parte della lettura.
- C4. Criteri scritti prima dei risultati: pavimento di rumore dalla replica (se C1 = raggio 0) accanto al 2 %; P1 decide la non inferiorita',
  P2 descrittivo con la sola regola di danno; nessuna soglia nuova dopo lo screening.
- C5. Preventivo completo con le valutazioni e il ripiego di budget dichiarato prima (quali bracci arrivano a 19.618 e con quale riferimento),
  e la quota delle frequenze basse nello stream di training riportata.

**Completamenti operativi gia' previsti** (non difetti): parita' dei batch e dei conteggi sui dati reali; timing e memoria per braccio;
valutatori e costi delle estrazioni; tetti di allocazione; checkout di campagna da SHA pubblicato; sottomissioni da semplicione e ledger;
valutazioni dei nuovi checkpoint da checkout ≥ c052792.

**Miglioramenti facoltativi:** bordo 4 invece di 6 patch (energia oltre 2,5 patch ≤ 0,1 %); registrare rango effettivo per fs e norma dei
gradienti per famiglia all'inizio e alla fine del training (se e quanto i kernel si muovono); se T100-G non mostra nulla, la variante a
quadratura delle coppie di Fourier (§2.3) come passo successivo, non ora; un secondo seme del solo contrasto vincente, solo se il budget lo
consente e come campagna a parte.

## 9. Confronto con la review del critico

La review del critico e' una verifica del codice e delle configurazioni, ed e' corretta: concordo su tutti i punti della sua tabella
(default, posizione della GELU, parita' dei parametri, apprendibilita', coerenza conteggi/perdita, posa, checkpoint, resume/branch/frozen,
masking prima del frontend, 306 contro 388 query nello scenario di test). Non tratta le domande di disegno (meccanismo del raggio con JEPA 0,
allineamento dell'endpoint all'ipotesi, punto di lavoro della GELU, incertezza dei criteri, budget): non c'e' contraddizione, c'e'
complementarita'. Una sola frase sua la preciserei: «core_radius = 2 protegge il supporto nominale da 100 ms» e' vera per il tensore, ma con
l'obiettivo B non protegge la perdita da nulla (§3.1).

## 10. Cosa non ho verificato

Niente su GPU, niente su dati reali, niente in DDP. Le tabelle 1-3 sono su pesi iniziali, rumore sintetico e maschere generate, non sul
corpus: danno ordini di grandezza e vanno rifatte dal collaudo sui batch veri (§3.5). Non ho verificato le fonti dell'audit del budget oltre
l'aritmetica (68,7075 + 63,2925 = 132; 212,18 + 318,82 = 531). La parita' dello stream fra B e un eventuale T25-L a raggio 0 non e'
verificabile dai log di B (`log_batch_origin` era spento): vale per costruzione (stesso seme, stesso stream, codice del loader invariato),
e la replica misurerebbe anche un'eventuale differenza li'. Nessuna delle misure qui sopra autorizza un job: la decisione sulle correzioni e'
di GPT e la firma di Simone.
