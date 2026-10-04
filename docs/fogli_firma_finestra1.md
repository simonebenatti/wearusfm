# Fogli di firma per la finestra 1 — D12, D14 e conto del compute (bozza di AG del 04/10/2026, NON firmati)

Decisioni da chiudere **prima della finestra 1** (01-03/11; piano §12: «quattro su sei vanno chiuse prima della finestra 1»). D15 e D16 sono gia'
firmate il 04/10 («approvo tutto», punti 13 e 14). Ogni foglio: cosa si decide, numeri, opzioni, proposta, conseguenza se sbagliata. Le stime di
costo vengono dal sanity vero in corso (job 59318048) e si aggiornano a fine run.

---

## 0. Conto del compute rifatto sui numeri misurati (non si firma: e' la base dei fogli)

**Cosa e' cambiato rispetto a v10 §10.3-10.5.**
- **Dati:** `manifest-v1` ha **D_t = 1,185·10⁸** time-patch per epoca (822,9 h), un terzo dei 3,6·10⁸ su cui erano pianificati la ladder e il budget.
  A 4 epoche sono 4,74·10⁸ time-patch, cioe' ~2,96 M finestre da 4 s, **~92.600 passi col batch da 32**.
- **Efficienza misurata:** il 30M del sanity (28,3 M parametri) fa un passo in **2,46 s**: ~84 TFLOP per passo stimati dai conti per modulo
  (backbone ~415 GFLOP e decoder ~108 GFLOP per finestra in avanti; indietro e ricalcolo del checkpointing inclusi), quindi **~34 TFLOP/s, MFU
  ~11%**. v10 assumeva il 30-40%.
- I due effetti quasi si compensano: un terzo dei dati, un terzo dell'efficienza.

**Costo di un run a D (4 epoche), stima.** Il backbone domina il costo e scala con N; MFU invariata finche' non la si migliora.

| Rung | Costo a D, misurato o scalato | v10 §10.5 |
|---|---|---|
| 30M | **~63 GPU-ora** (misurato: 2,46 s/passo x 92.600 passi) | — |
| 100M | ~210 | 120-160 |
| 300M | ~640 | 400-500 |
| 1B | ~2.100 (forse ~1.300 se l'MFU sale con la larghezza) | 1.200-1.600 |

**Totali, stima, senza il vertice:**

| Blocco | GPU-ora |
|---|---|
| Finestra 1: ablation e calibrazione a 30-100M | ~1.500 |
| Finestra 1: pilot sulle epoche a 100M, 2 seed, con i rami di decay (D14) | ~1.000 |
| Finestra 1: asse dei soggetti a 100M e 300M, 2 seed, piu' il controllo al 25% | ~3.000 |
| Ladder 30M -> 1B, 3 seed (~9.000), piu' i controlli 2D (~4.000) | ~13.000 |
| **Totale** | **~18.500** |

Resta sotto i ~70.000-90.000 disponibili, come in v10: **il piano regge**. L'incertezza principale e' l'MFU. Portarla dall'11% al 25% (attenzione
senza maschere dove si puo', meno cicli Python per campione, il decoder che oggi guarda tutti i P x K latenti) dimezzerebbe tutto. Non e' sul
cammino critico.

**Token per parametro.** A 1B e 4 epoche sono **~0,47 time-patch per parametro** (v10: ~1,4). La «saturazione precoce» che v10 dava per l'esito
piu' probabile lo e' ancora di piu'. Va detto nel paper fin dall'inizio, come chiede v10 §10.3.

---

## D12 — Regola decisionale del pilot sulle epoche (v10 §10.3)

**Si decide:** come si sceglie E, le epoche nominali usate da tutti i rung, dai risultati del pilot a 100M con E in {1, 2, 4, 8}. La regola si
congela **prima** dei risultati.

**Numeri:** il pilot si legge su: cross-soggetto sul ramo sparso, transfer su dataset mai visto, sonde fisiche, dataset-ID, rango, perdite, per
topologia (v10). Per una regola servono poche **metriche primarie** e un errore standard.

**Proposta di AG** (rende concreto l'esempio di v10 §10.3):
- **Metriche primarie (2):**
  - **P1, cross-soggetto sul ramo sparso:** sonda lineare sull'encoder congelato, accuratezza bilanciata sui soggetti di test della classe A
    (split firmati);
  - **P2, transfer su dataset mai visti:** sonda lineare sull'encoder congelato su EPN-612 e UCI-EMG (harness del passo 5), accuratezza bilanciata,
    media dei due.
- **Errore standard:** bootstrap sui soggetti di test (1.000 ricampionamenti) per ogni seed, poi combinato sui 2 seed.
- **Regola:** **E = 4** salvo che, su almeno una metrica primaria, il passaggio da 4 a 8 migliori di **piu' di 2 errori standard e** di almeno un
  quarto del guadagno da 2 a 4 (Δ₄→₈ >= 0,25·Δ₂→₄). In quel caso **E = 8**. E = 1 e 2 si leggono e si riportano, non si scelgono: costerebbero
  meno, ma il piano e' costruito su E = 4.
- **Per topologia, mai solo aggregato:** la regola si applica a P1, che e' gia' sul ramo sparso, e a P2; le altre metriche si riportano.

**Dipende da:** l'harness del passo 5 (P2) e la sonda cross-soggetto (P1) devono esistere prima della finestra 1. Oggi non ci sono: sono nel
calendario di AG per le prossime settimane.

**Se sbagliata:** con E troppo basso la ladder sottostima la capacita' utile; con E troppo alto spende il doppio. In tutti e due i casi il controllo
2D (D14) rileva in parte l'errore.

**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo

---

## D14 — Schedule del learning rate (WSD), continuazione e teacher (v10 §10.4)

**Si decide:** la forma dello schedule, uguale per tutti i rung, e la regola di continuazione a 2D. v10 la vuole tale da permettere una vera
continuazione: «warmup -> lunga fase stabile -> decay separato».

**Proposta di AG:**
- **Warmup** lineare sul 2% dei passi a D, almeno 1.000 passi (30M: ~1.850 passi);
- **fase stabile** al learning rate di picco, scelto dalla calibrazione a 30-100M della finestra 1;
- **decay lineare a zero sul 20% finale** dei passi. Le forme alternative citate in letteratura (es. «1-sqrt») non si propongono senza averle
  verificate: se interessano, prima il fact-checker;
- **due checkpoint per rung:** pre-decay (all'80% dei passi a D) e post-decay (a D, quello valutato);
- **continuazione a 2D:** dal checkpoint pre-decay si prosegue la fase stabile fino all'80% di 2D, poi decay sul 20% finale. E' identico a un run
  a 2D fatto da zero con lo stesso warmup e lo stesso picco;
- **pilot con i rami di decay** (piano, «una sola run per seed»): un run stabile fino a 6,4 epoche e poi decay fino a 8; dai checkpoint stabili a
  0,8, 1,6 e 3,2 epoche partono rami di decay lunghi 0,2, 0,4 e 0,8 epoche, che danno i modelli a E = 1, 2, 4. Costo: **~9,4 epoche per seed
  invece di 15** con 4 run separati;
- **teacher EMA costante a 0,996** in tutte le fasi (come il sanity, decisione 18). Uno schedule che cresce verso 1 andrebbe legato al decay per non
  rompere la continuazione: piu' complicato, senza un motivo misurato. Se il sanity mostrasse un problema, si riapre;
- **ottimizzatore** come il sanity: AdamW, weight decay 0,05, clip del gradiente 1,0, batch da 32 finestre al 30M; il batch dei rung maggiori lo
  fissa la calibrazione.

**Se sbagliata:** un decay troppo breve lascia il modello sotto-ottimizzato a D; uno troppo lungo accorcia la fase stabile e rende meno pulita la
continuazione. Il 20% e' un valore di lavoro, non misurato: la calibrazione della finestra 1 puo' confrontare 10% e 20% a 30M (~63 GPU-ora per
punto).

**Tua decisione:** [ ] firmo · [ ] firmo con correzione · [ ] non firmo
