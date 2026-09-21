# Report — Passo 0: benchmark sintetico del dataloader

Applica le regole D6a e D6b congelate in [`docs/decisioni.md`](decisioni.md) ai risultati
misurati. Riferimento: [`docs/dataloader_bench_spec.md`](dataloader_bench_spec.md).

Job SLURM: `58134850` (prova funzionale flash_attn_varlen_func) · `58137579` (asse M) ·
`58137580` (asse L). File di risultato grezzi in [`results/step0/`](../results/step0/).

---

## D6a — Montaggi: al volo o precompute

**Regola congelata** (`docs/decisioni.md`): al volo resta il default se, col consumatore
del 30M, la frazione di tempo in attesa è ≤ 2% **e** il ritmo a vuoto è ≥ 1,5× il richiesto,
a cache calda e fredda su `$WORK`.

Ritmo richiesto per un modello da 30M (K=64, contesto 4s, patch 25ms, MFU 40%): **~68
finestre/s**.

| | al volo — fredda | al volo — calda | precompute — fredda | precompute — calda |
|---|---|---|---|---|
| waiting_fraction | 0,0 | 0,0 | 0,0 | 0,0 |
| finestre/s misurate | 241,3 | 238,4 | 267,8 | 273,2 |
| margine sul richiesto | 3,55× | 3,51× | 3,94× | 4,02× |

**Esito: AL VOLO.** Entrambe le soglie sono superate ampiamente, a cache fredda e calda:
attesa 0% (soglia ≤2%) e margine ≥3,5× (soglia ≥1,5×). Per la regola, "al volo" resta il
default — non si passa a `precompute` solo perché quest'ultimo è marginalmente più veloce
(prevedibile: legge meno byte, montaggio già ridotto). Nessuna conseguenza sulla semantica
di `D_c` (v10 §4.6): i montaggi restano augmentation stocastica, non dato fisso.

**Limiti della misura** (vedi nota in `docs/decisioni.md` sotto D6a): il benchmark non
modella il trasferimento worker→processo principale né la contesa I/O reale fra i 4 loader
dei 4 rank GPU dello stesso nodo. Il margine misurato (3,5×) è largo abbastanza da
assorbire ragionevolmente questi limiti, ma resta una misura indicativa, non una garanzia
a piena scala con 4 GPU e dati reali.

---

## D6b — Layout del batch: padding, packing o bucketing

**Regola congelata**: il padding è escluso in partenza se la frazione di token di padding
supera il 50%. Fra i rimanenti vince chi dà più token utili/s nel proxy; a parità entro il
10% vince il più semplice, nell'ordine L1 → L3 → L2.

Proxy a d_model=384 (rung 30M), 500 batch, batch_size=32, montaggi a 1000 Hz (classi A/B/C
rappresentate: NinaPro standard/DB6, emg2pose, CapgMyo).

| Arm | Padding | p50 tempo di passo | Passi/s | Memoria di picco |
|---|---|---|---|---|
| padding (L1) | **56,4%** | 0,101 s | 9,93 | 3373 MB |
| bucketing (L3) | 0% | 0,089 s | 11,18 | 3045 MB |
| **packing (L2)** | 0% | **0,079 s** | **12,59** | 3116 MB |

**Esito: PACKING.** Il padding è escluso subito (56,4% > 50%). Fra bucketing e packing la
differenza è del 13% (packing più veloce) — sopra la soglia di parità del 10%, quindi
**packing vince nettamente**, senza dover ricorrere al criterio del "più semplice". La
prova funzionale di `flash_attn_varlen_func` (job 58134850, forward+backward in bf16 su
A100 reale) e il proxy completo (job 58137580) confermano che il kernel varlen regge in
produzione, non solo isolato.

**Limiti della misura** (dichiarati in `src/wearusfm/model/proxy.py` e
`bench/measure_asse_l.py`): il bottleneck Perceiver qui attende su tutti i token
canale-patch di un campione in un colpo solo, non fattorizzato per singolo patch temporale
come vorrebbe l'architettura finale (v10 §5); il batch è a frequenza singola (1000 Hz), non
ancora col raggruppamento per frequenza reale del collate (spec §4, stadio 5). Il confronto
relativo fra i tre layout resta valido; i valori assoluti di tempo di passo non vanno presi
come predizione del costo del modello finale.

---

## Conseguenze per il manifest e il modello

- `D_c` (v10 §10.3): nessuna conseguenza, i montaggi restano augmentation stocastica
  (esito D6a = al volo).
- v10 §2.7 ("mai bucketing per numero di canali fra batch"): nessuna modifica necessaria,
  il bucketing (L3) non ha vinto.
- L'encoder locale userà attenzione sui vicini via `gather` (rev. 3, v10 §5.4); il kernel
  `flash_attn_varlen_func` si userà solo per la cross-attention del Perceiver (packing),
  come già previsto — confermato funzionante.

## Chiuso quando

- [x] `results/step0/*.json` esiste per ogni braccio/configurazione misurata
- [x] Questo report applica D6a e D6b
- [x] **Firma di Simone** (21/09/2026): esito approvato — al volo + packing, limiti
  dichiarati letti e accettati.

**Passo 0 chiuso il 21/09/2026.**
