# Perche' il pretraining non rende: diagnosi e idea di soluzione (Claude, 23/10/2026)

**Stato: ANALISI E PROPOSTA, non firmata.** Richiesta di Simone (23/10/2026): «dobbiamo capire perche la parte del tokenizer, degli
embedding e quindi il pretraining non va, e immaginare possibili soluzioni. Guarda cosa e' stato fatto finora e dammi una tua idea di
soluzione». Nessun job, nessuna soglia nuova. Le interpretazioni sono marcate come tali; i numeri vengono dai file citati.

## 1. I fatti (tutti gia' nel repo)

| # | Misura | Numero | Fonte |
|---|---|---|---|
| F1 | Encoder allenato (Step 1, 4.000 passi su emg2qwerty) contro **lo stesso encoder casuale**, stesse sonde | A: P1 16,0 -> 17,9 (+1,9 punti), P2 73,1 -> 79,8 (+6,7); B: +2,3 / +7,5 | `results/passo6/step1_random_downstream_20261015.json` |
| F2 | Bracci dell'obiettivo a 30M, 4 epoche sul corpus intero: **ancore-only contro A** (JEPA + ancore) | S: P1 +0,51, P2 -0,40; ufficiali: P1 +0,07, P2 -0,89. Nessuna differenza oltre la soglia | `results/infra/w1arm_20261021/valutazione/esito_confronto_objective.json` |
| F3 | **JEPA-only** (senza ancore), due semi | collassato: rapporto varianza fra campioni / fra query 0,13 -> 0,08, rango 70 -> 38 in 3.500 passi | `.../runs/W1ARM-JEPA-only-30M-s1/diagnostiche.json` |
| F4 | Rango effettivo dello studente nei primi 4.000 passi | ~67-87 a 30M, 100M **e** 300M (dim 384 / 640 / 896): non cresce con la larghezza. A fine training A: 144-146 su 384 | `results/infra/w1cal_c4_20261022/diagnostiche_erank_taglie.json`; decisioni.md 22/10 10:50 |
| F5 | Rapporto di collasso del braccio A **sano** lungo il training | scende da 2,8 a 1,8-2,0 (ancore-only: 1,26 stabile) | decisioni.md 22/10 10:50 |
| F6 | Sonda spettrale congelata: quanto l'embedding predice forma ed energia spettrale | R2 0,86 / 0,91 (fast), 0,80 / 0,92 (slow) | `results/passo6/step1_frozen_probe_20261008.json` |
| F7 | 2A: le 32 statistiche spettrali fisse **aggiunte** all'embedding | P1 -0,37, P2 +0,07: non aggiungono nulla | `results/passo6/step2a_comparison_20261014.json` |
| F8 | 2C: rami spettrali **allenati** sui 32 target contro gli stessi rami **al passo 0** (proiezione casuale delle log-potenze) | allenati peggio: B P2 -1,59; i rami al passo 0 aggiungono +2,34 su P2 | `results/passo6/step2c_comparison_20261014.json` |
| F9 | Ancora RVQ: codici del tokenizer congelato come target | perdita ferma all'entropia (0,983), top-1 0,05%: imprevedibili da questa rappresentazione | decisioni.md 04/10 (job 59328078) |
| F10 | Livelli assoluti, braccio A ufficiale | P1 21,3 (DB2 49 classi 0,20; **DB3 amputati 0,03 = caso**; DB6 7 classi 0,35); P2 86,0 (EPN-612 0,87, UCI 0,76). NeuroRVQ con fine-tuning su EPN-612: 95,4 (regime diverso) | esito dei bracci; decisioni.md 05/10 |
| F11 | Readout delle sonde | media sui canali dell'encoder locale + media su latenti e istanti del backbone: 2d numeri per finestra | `model/spectral_step1.pool_p1p2`, `harness/fm_features.py` |
| F12 | Corpus del manifest | 822 h: classe B (anelli, in gran parte emg2pose + emg2qwerty) 75%, classe A (sparso, caso di deployment e di P1) 20%, HD 5%. Etichette (angoli articolari, tasti, `restimulus`) **non ingerite** | `docs/manifest_annidati_d16.md`; `ingest/emg2pose.py` |

## 2. Diagnosi

**In una frase:** l'obiettivo attuale insegna all'encoder le stesse cose che le ancore chiedono (ampiezza e forma spettrale per patch
da 25 ms), e niente di piu'; il termine JEPA, che doveva portare struttura spaziale e temporale alla scala del movimento, o collassa sulla
query (F3) o non aggiunge nulla di misurabile (F2). Il risultato e' un encoder che una sonda lineare legge come un banco di potenze di
banda levigato: batte di poco una rete casuale (F1), contiene gia' tutte le statistiche spettrali (F6, F7), ha un rango che non dipende
dalla capacita' del modello (F4) e **perde** informazione che un ramo non allenato conserva (F8).

Le cause, in ordine di peso (*interpretazione mia, da verificare con le prove di §4*):

1. **Scala del target.** Il token e' una patch da 25 ms di un segnale stocastico. Di una patch nascosta e' prevedibile quasi solo
   l'ampiezza e la forma spettrale (v10 §6.1 lo dice): il teacher JEPA, scartando l'imprevedibile come deve, converge a un target che
   contiene cio' che le ancore gia' danno. Le altre due voci della lista di v10 §6.1 («pattern spaziale tra canali, struttura temporale
   alla scala del movimento») vivono a 100 ms - 1 s e fra canali: l'obiettivo non le chiede mai in forma esplicita, e il JEPA non le
   porta da solo.
2. **Target (b) e collasso da query.** Il target e' l'uscita del decoder a query del teacher: dipende dalla query (identita' + tempo)
   oltre che dal contenuto. Senza ancore collassa del tutto (F3, il fallimento previsto da v10 §5.6); con le ancore resta in piedi, ma
   la quota di contenuto del target **scende lungo il training** anche nel braccio sano (F5). Il braccio di controllo (a) previsto da v10
   §5.6 e §10.1 non e' mai stato eseguito.
3. **Readout delle sonde.** La media sui canali e sui latenti (F11) cancella il pattern spaziale, cioe' quale elettrodo e' attivo: il
   discriminante principale dei gesti su un bracciale a 8 canali. Sopravvive solo quello che la rete mescola in modo non lineare con
   l'identita' di canale (una rete casuale fa il 73% su EPN-612 proprio cosi', F1). Vale per tutti i confronti fatti finora allo stesso
   modo, quindi le **differenze** restano valide; i **livelli** (P1 al 21%) sono probabilmente sottostimati. La baseline classica in
   corso (4 feature di Hudgins per canale, 32 numeri per un Myo) e' il test diretto: se batte 768 numeri del modello, il readout e' un
   problema di primo ordine.
4. **Esposizione del caso A.** P1 misura il ramo sparso (NinaPro 8 + 4 mirati, amputati), che ha il 20% dell'esposizione; DB3 al caso
   dice che verso gli amputati non trasferisce niente. Qui il dato e' quello che e'; pesa comunque la causa 3.

**Cosa NON e' il problema (con le prove):** la risoluzione spettrale del front-end (F6, F7: lo spettro c'e' tutto); l'ancora RVQ (tolta;
i codici sono imprevedibili per costruzione, F9); LR e stabilita' (C1-C3 regolari); la **taglia** del modello (F4: il rango e' lo stesso a
30M, 100M e 300M, quindi la capacita' non e' il collo di bottiglia con questo obiettivo). Conseguenza pratica: la nuova C4 (300M) misura
un modello piu' grande dello stesso obiettivo; finche' l'obiettivo non cambia, e' spesa che non puo' cambiare il quadro.

## 3. Idea di soluzione

Principio: dare al pretraining un target che **contenga l'informazione che i compiti a valle usano** e che il segnale renda **davvero
prevedibile** — alla scala del movimento, non della patch — e leggere l'encoder in modo che il pattern spaziale non si perda.

### 3.1 Readout per canale (zero training)

Per le sonde: uscita dell'encoder locale mediata nel tempo ma **tenuta per canale**, concatenata nell'ordine del montaggio (Myo: 8 x d;
NinaPro: 12-16 x d), accanto al readout attuale; stessa sonda lineare, stessa validazione interna S. Sui checkpoint post-decay gia'
estratti (A s0/s1, ancore-only s0/s1) e sul casuale. Confronto a tre: readout medio, readout per canale, baseline classica. Dice se
l'encoder sa cose che la media butta via. Costo: una estrazione per checkpoint e compito (~0,75 GPU-ora l'una, come la valutazione dei
bracci) o CPU.

### 3.2 Obiettivo alla scala del movimento (il cambio vero)

a. **Target per regione, non per token.** Il target JEPA di una regione nascosta (tubo: gruppo di canali x 4-80 patch) e' la **media** delle
   rappresentazioni del teacher su quella regione (o su segmenti da 200 ms di essa), e lo studente predice quella media. Toglie dal
   target la parte dominata dal rumore della singola patch e allinea il compito alla scala dell'inviluppo (5-10 Hz). Modifica piccola in
   `training/jepa.py` (`_jepa_compute`: media di target e predizione per regione prima dell'errore); ancore invariate.
b. **Target (a)** (uscita per canale dell'encoder locale del teacher) al posto di (b): il braccio di controllo mai eseguito. Elimina il
   percorso query -> target. Gia' nel codice (`JEPAConfig.target = "a"`).
c. **Ancore fisiche di alto livello dove le etichette esistono** (v10 §6.5, «teste supervisionate attivate per batch secondo il dataset»;
   mai implementate). La piu' forte: **angoli articolari di emg2pose** (20 gradi di liberta' a 2 kHz, 423 h, 193 soggetti, gia' nei file
   HDF5 ma non ingeriti): una testa di regressione sulle uscite del backbone/decoder, attiva solo sui batch di emg2pose, a 50 Hz. Costringe
   la rappresentazione a codificare le variabili motorie — i gesti sono pose — che e' esattamente cio' che P1 e P2 chiedono e che da
   un bracciale e' prevedibile (lo mostra lo stesso emg2pose, anche fra utenti). Disgiunto da tutti i dataset di P1 e P2: nessuna
   etichetta dei compiti di test entra nel pretraining. **E' un cambio di principio rispetto a v10 §6.1 («tutti i target sono
   osservativi»): lo decide Simone.** Richiede l'ingest degli angoli (job CPU) e un campo nel formato processato (`labels.npz` esiste gia').
   Seconda candidata, piu' debole: i tasti di emg2qwerty.
d. (dopo) Predizione del futuro a 200 ms - 1 s su rappresentazioni di segmento (stile CPC) e penalita' sull'identita' del soggetto.

### 3.3 Pilot proposto (da congelare col `critico` prima di qualunque lancio)

30M, **1 epoca** (come C1: ~19 GPU-ora per run), seme 0, metrica S della validazione interna con **tutti e due i readout** (3.1), baseline
classica come riferimento esterno. Riferimento: **C1 seme 0 al LR 1e-4 esiste gia'** (S 18,21), nessun run in piu'.

| Braccio | Cosa cambia rispetto ad A | GPU-ora attese |
|---|---|---|
| R1 | target (a) | ~19 |
| R2 | target per regione (3.2a), target (b) | ~19 |
| R3 | ancore-only + testa di posa su emg2pose (3.2c), senza JEPA | ~19 |
| R4 | R2 + testa di posa | ~19 |

Totale ~76 GPU-ora attese (tetto ~95) piu' valutazione (~6-8); il seme 1 solo per il vincitore. Prima: ingest degli angoli (CPU, da
preventivare) e readout per canale sui checkpoint esistenti (3.1). **Regola di lettura (proposta, da congelare):** un braccio vince se
migliora S della P1 **e** della P2 di almeno +1 punto rispetto a C1 s0 a pari esposizione, confermato sul seme 1; l'encoder si dice utile
solo se, col readout migliore, batte la baseline classica di almeno +1 punto sulle stesse sonde.

## 4. Decisioni per Simone

1. Se fare il readout per canale (3.1) sui checkpoint esistenti: zero training, risponde alla causa 3 insieme alla baseline classica.
2. Se sospendere la nuova C4 (300M) finche' un braccio del pilot non mostra un guadagno: con l'obiettivo attuale la taglia non cambia il
   rango (F4) ne', a 30M, il JEPA aggiunge valore (F2).
3. Se ammettere l'ancora di posa (3.2c): e' l'unico braccio che cambia il principio «solo target osservativi».
4. Budget del pilot (~80-100 GPU-ora) e da quale voce.
