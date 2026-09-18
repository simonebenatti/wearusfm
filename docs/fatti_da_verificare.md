# Fatti da verificare

Registro dei "fatti da verificare" di [`docs/fm_emg_reference_v10.md`](fm_emg_reference_v10.md)
§12. Un fatto qui dentro è un **compito**, non un dato: non entra nel documento di riferimento
come premessa finché non passa da `raccolto` a `verificato da SB`.

Formato per riga: **fatto** · **fonte** (URL o percorso esatto) · **citazione esatta** ·
**stato**.

Stati possibili: `raccolto` (un subagent — FACT o LIC — ha trovato e citato la fonte) →
`verificato da SB` (Simone ha controllato la fonte di persona ed è d'accordo) oppure
`non trovato` (nessuna fonte ufficiale lo riporta: resta un compito aperto, non si inferisce).

---

## Dalla v10 §12, elenco originale

| # | Fatto | Fonte | Citazione esatta | Stato |
|---|---|---|---|---|
| 1 | Frequenze native di NinaPro DB8 (~1111 Hz) e DB10 (~1926 Hz); presenza di amputati in DB8 e DB10; varianti del montaggio NinaPro (DB6 a 14 elettrodi, DB8 a 16); disposizione interna delle fasce di GRABMyo | — | — | da raccogliere |
| 2 | Sovrapposizione di soggetti fra i DB NinaPro | — | — | da raccogliere |
| 3 | Conteggio dei soggetti del corpus (~600) e delle ore dopo la rimozione di DB9 | — | — | da raccogliere |
| 4 | Dal codice del repo NeuroRVQ: banda di filtraggio usata nel pretraining del tokenizer (20–90 Hz?), normalizzazione attesa in ingresso, se il transformer mescola canali diversi | — | — | da raccogliere |
| 5 | Sovrapposizione fra i soggetti visti dal tokenizer NeuroRVQ e i soggetti di test di §8; licenza del checkpoint | — | — | da raccogliere |
| 6 | Lista degli 11 muscoli di Camargo 2021 | — | — | da raccogliere |
| 7 | Kaifosh et al.: presenza e forma delle curve di scaling nel numero di partecipanti; accesso, licenza e dimensioni del dataset rilasciato | — | — | da raccogliere |
| 8 | Orientamento della fascia documentato, dataset per dataset, per tutti gli anelli e le fasce (serve alla funzione atlante di §4.5) | — | — | da raccogliere |
| 9 | `flash_attn_varlen_func` in `cineca-ai/4.3.0`: import verificato (flash_attn 2.5.5) **e** prova funzionale forward + backward in bf16 su A100 reale, nella forma d'uso prevista per il Perceiver (cu_seqlens_q fisso a K=64, cu_seqlens_k variabile C ∈ {8,...,256}) | sessione corrente, job SLURM 58134850 su Leonardo (`boost_qos_dbg`, COMPLETED in 20s) via leonardo-ops; script `bench/verify_flash_attn_varlen.py` | output `results/step0/flash_attn_varlen_check_58134850.json`: `"status": "OK"`, `"forward_output_shape": [448, 8, 64]`, `"forward_output_dtype": "torch.bfloat16"`, `"backward_grads_finite": true`, `"gpu_name": "NVIDIA A100-SXM-64GB"` | raccolto — evidenza completa, in attesa della firma di Simone (D4) |
| 10 | MFU reale sul Booster, per sostituire la stima di §10.5 | — | — | da raccogliere |

## Dalla dataloader_bench_spec.md §13, aggiunte rev. 2

| # | Fatto | Fonte | Citazione esatta | Stato |
|---|---|---|---|---|
| 11 | Hyser: 256 canali a 2048 Hz, 20 soggetti, due sessioni in giorni diversi | PhysioNet (pagina ufficiale) | vedi dataloader_bench_spec.md §13 | raccolto — **le 4 griglie da 64 (due lato flessorio, due estensorio) vengono da fonte secondaria**, da verificare sulla pagina ufficiale o sul paper |
| 12 | putEMG: 24 elettrodi in 3 fasce da 8, a 45°, primo elettrodo di ogni fascia sull'ulna, numerazione oraria; 5120 Hz; registrazione monopolare; licenza CC BY-NC 4.0 | pagina del dataset e paper ufficiali | vedi dataloader_bench_spec.md §13 | raccolto — da firmare (D4) |
| 13 | `flash_attn`: interfaccia espone maschera causale, finestra scorrevole e pendenze ALiBi; nessun argomento per un bias additivo arbitrario | repository ufficiale flash-attention | vedi dataloader_bench_spec.md §13 | raccolto |
