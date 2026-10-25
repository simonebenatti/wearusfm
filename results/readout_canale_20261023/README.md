# Readout per canale sui checkpoint esistenti: esiti della campagna `readout-canale-20261023`

Protocollo `docs/readout_per_canale_v1.md` (lettura congelata in 63ac0dd il 23/10/2026); codice sul cluster 6242bc3; campagna chiusa il
25/10/2026. Rapporti delle sonde e predizioni fuori fold restano fuori da Git (`~/wearusfm_local/readout_canale_20261023/`, stesso albero del
cluster); qui stanno solo i confronti, il riepilogo delle sonde e il consuntivo.

- `confronti/*.json`: uscite di `scripts/channel_readout.py --stage compare|equality` (regola dei bracci, bootstrap congiunto a 1.000
  ricampionamenti, seme 20261009; `mode` nel rapporto dice che cosa si confronta).
- `riepilogo_sonde.json`: le 21 sonde (S, S per dataset, k effettivo per fold, `k_capped`, SHA del checkpoint e del codice, durata).
- `consuntivo_sacct_20261025.txt`, `consuntivo.json`: `sacct -X` dei 30 job del ledger (tutti COMPLETED): 2,74 GPU-ora di estrazioni (tetto 10,0)
  e 238,9 core-ore di sonde (tetto 672, attese ~110: le sonde P1 a k 768 sono durate 1 h 55-2 h 27 l'una contro i ~50 min stimati).

| File | Confronto (candidato - riferimento) | Delta medio dei due semi [IC95] | Per seme | Lettura |
|---|---|---|---|---|
| `uguaglianza_pool_s0_p1.json` | `pool` riestratto di A s0 contro S storica (P1) | 0,000 (soglia 0,1) | — | uguale, zero predizioni diverse |
| `confronto1_p1_local_vs_pool.json` | `local` k 768 - readout medio, P1 | -2,00 [-2,60; -1,36] | -1,99 / -2,00 | peggiora |
| `confronto1_p1_dec_vs_pool.json` | `dec` k 768 - readout medio, P1 | -0,42 [-1,20; +0,33] | -0,28 / -0,56 | regola non superata |
| `confronto1_p2_local_vs_pool.json` | `local` k 768 - readout medio, P2 | -2,49 [-4,59; -0,66] | -3,12 / -1,87 | peggiora |
| `confronto1_p2_dec_vs_pool.json` | `dec` k 768 - readout medio, P2 | +1,33 [-0,93; +3,54] | +0,72 / +1,93 | regola non superata |
| `td_vs_A_pool_p1.json` | readout medio di A - baseline classica (Hudgins), P1 | +4,01 [+2,97; +5,03] | — | migliora |
| `td_vs_A_pool_p2.json` | readout medio di A - baseline classica, P2 | +13,86 [+11,00; +17,03] | — | migliora |
| `A_vs_AO_p1_local.json` | A - ancore-only, `local` k 768, P1 (`--models`) | +0,20 [+0,01; +0,39] | +0,31 / +0,09 | regola non superata |
| `A_vs_AO_p1_dec.json` | A - ancore-only, `dec` k 768, P1 (`--models`) | -0,78 [-1,12; -0,45] | -0,78 / -0,78 | regola non superata |
| `A_vs_AO_p2_local.json` | A - ancore-only, `local` k 768, P2 (`--models`) | +0,65 [-0,01; +1,26] | +0,88 / +0,42 | regola non superata |
| `A_vs_AO_p2_dec.json` | A - ancore-only, `dec` k 768, P2 (`--models`) | -0,46 [-1,52; +0,47] | +0,13 / -1,05 | regola non superata |
| `descrittivo_p2_local_k0_vs_k768.json` | A `local` a larghezza intera - k 768, P2 | -0,01 [-0,31; +0,24] | +0,19 / -0,22 | descrittivo: la PCA non costa |
| `descrittivo_p2_dec_k0_vs_k768.json` | A `dec` a larghezza intera - k 768, P2 | +0,06 [-0,35; +0,38] | +0,36 / -0,24 | descrittivo: la PCA non costa |
| `descrittivo_p2_local_k0_vs_pool.json` | A `local` a larghezza intera - readout medio, P2 | -2,51 [-4,65; -0,55] | -2,93 / -2,09 | descrittivo (come a k 768) |
| `descrittivo_p2_dec_k0_vs_pool.json` | A `dec` a larghezza intera - readout medio, P2 | +1,39 [-1,09; +3,73] | +1,09 / +1,69 | descrittivo (come a k 768) |

Punti percentuali di S della validazione interna. Risposte del protocollo: (1) con la regola congelata il readout medio non nasconde
informazione (nessuno strato «migliora» in un compito); (2) l'encoder batte le feature classiche (candidato: il `pool` storico, scelto con la
regola prima di aprire la baseline); opzione ancore-only: «ancore-only = A» regge col readout per canale in tutte e quattro le letture.
Letture multiple senza correzione, dichiarate nel protocollo. Voci nel registro (`docs/claude_GPT_shared`) del 23-25/10/2026 e sintesi in
`docs/sintesi_pretraining_20261024.md`.
