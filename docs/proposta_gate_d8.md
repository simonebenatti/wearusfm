# Proposta — gate di consistenza al ricampionamento (D8, passo 3): definizioni operative e prerequisito

**Stato: PROPOSTA di AG del 01/10/2026, NON firmata.** Le soglie sono gia' congelate (D8a, 29/09/2026: errore relativo RMS delle feature <= 5%;
sonda A-contro-B <= 55% con intervallo al 95% che contiene il 50%). Qui ci sono le tre cose che `decisioni.md` dice di scrivere **prima** del lancio
(banda condivisa, frequenze A e B, intervallo della sonda) e un prerequisito scoperto oggi. Scadenza del gate: **18/10**.

## Cosa verifica, in parole semplici

Il front-end deve dare **le stesse feature** per lo stesso segnale, che sia campionato a 2 kHz, a 1 kHz o a 200 Hz: stessi pesi, cambia solo la griglia
(v10 §4.2). Se non e' cosi', il front-end regala al modello l'identita' del dataset attraverso la frequenza di campionamento. Si prende una registrazione
a 2 kHz e se ne fanno due versioni con **lo stesso contenuto su griglie diverse** (v10 §7.1): se le feature coincidono e una sonda non le distingue,
il gate passa.

## Prerequisito: il front-end non esiste ancora

Nel codice c'e' solo `src/wearusfm/model/proxy.py`, un front-end **finto** scritto per misurare i costi del dataloader (media su una lunghezza fissa e
proiezione lineare). Il front-end di v10 §4.2 va scritto: banco di kernel continui in tempo fisico con **tre famiglie** (base di Fourier, spline, kernel
da MLP su `t`), **anti-aliasing per ciascuna famiglia** alla Nyquist della griglia, **fattore Δt = 1/fs** nella somma di convoluzione, patch in ms.
Test su CPU prima del gate: un seno puro deve dare la stessa ampiezza in uscita a 1 kHz e a 2 kHz (fattore Δt); un seno sopra la Nyquist della griglia
deve essere attenuato (anti-aliasing). **Il gate si puo' fare sul front-end all'inizializzazione**: la proprieta' e' dell'architettura, non
dell'addestramento, quindi non servono GPU ne' un modello addestrato.

## Definizioni operative proposte (da firmare)

1. **Dataset:** **Kaifosh** (2 kHz, 100 utenti, gia' ingerito). Il piano indicava emg2qwerty, che e' ancora in collaudo; Kaifosh e' lo stesso dispositivo
   alla stessa frequenza. *Alternativa:* aspettare emg2qwerty.
2. **Due casi, entrambi devono passare:**
   - **caso 1 kHz:** A = registrazione a 2 kHz filtrata passa-basso a **450 Hz** e lasciata a 2 kHz; B = A decimata a **1 kHz** (un campione ogni 2);
   - **caso 200 Hz** (il Myo di DB5): A = filtrata a **90 Hz** e lasciata a 2 kHz; B = A decimata a **200 Hz** (uno ogni 10).
   Filtro: Butterworth di ordine 8, a fase zero. Si filtra **prima** di decimare, quindi A e B hanno esattamente lo stesso contenuto.
3. **Banda condivisa:** sotto la frequenza di taglio del caso (450 Hz o 90 Hz). A e B contengono solo quella banda per costruzione; le feature si
   confrontano sulle stesse patch in **tempo fisico** (la griglia delle patch parte dallo stesso istante).
4. **Errore relativo RMS:** ‖F_A − F_B‖ / ‖F_A‖ sulle uscite del front-end, su tutte le patch e tutte le finestre. **Passa se <= 5% nel totale e in
   ciascuna delle tre famiglie** (proposta di AG: una famiglia sbagliata basta a dare l'identita' del dataset). *Alternativa:* solo il totale.
5. **Sonda A-contro-B:** finestre di 4 s; per finestra, media e deviazione standard nel tempo delle feature del front-end. Ogni finestra compare in
   entrambe le versioni, quindi le classi sono bilanciate. **Split per soggetto** 60/20/20 e le due versioni della stessa finestra sempre nella stessa
   parte. Sonda: **la piu' forte** fra regressione logistica e gradient boosting, scelta sulla validazione. Una sonda debole renderebbe piu' facile
   passare: e' l'errore corretto ieri nella regola dei rami. Accuratezza bilanciata sul test; **intervallo al 95% con bootstrap per soggetto** (1000
   ricampionamenti, come V3). Campione: 20 finestre per utente.
6. **Pesi:** front-end all'inizializzazione con **3 semi**; tutti e 3 devono passare.
7. **Costo:** CPU (sul Mac o un job seriale su Leonardo), zero GPU-ora per il gate all'inizializzazione.

## Se il gate fallisce

D8b, da decidere al momento: prima si controllano fattore Δt e anti-aliasing per famiglia (piano); una settimana per correggere; se il 18/10 non
passa, i front-end separati per frequenza diventano il default (piano, passo 3). Le scelte marcate «proposta di AG» (filtro, 4 s, 20 finestre, 3
semi, soglia per famiglia) non vengono dal piano: si firmano o si cambiano adesso, non dopo aver visto i risultati.

## Cosa si firma

- [ ] dataset (Kaifosh, oppure emg2qwerty)
- [ ] i due casi e il filtro (punto 2) e la banda condivisa (punto 3)
- [ ] errore RMS per famiglia e nel totale (punto 4), oppure solo nel totale
- [ ] la sonda e il suo intervallo (punto 5), i 3 semi (punto 6)
