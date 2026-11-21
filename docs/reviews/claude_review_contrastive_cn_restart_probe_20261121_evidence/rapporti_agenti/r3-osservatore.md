# R3 - osservatore della riduzione e hook dell'ottimizzatore (commit 9d34dc69)

Cartella di lavoro: /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_probe/agents/r3-osservatore
Script eseguiti: equiv.py (4 rank Gloo, A/B/C/D), layout.py (layouts() con i dati veri del logger DDP), params.py (modello vero CN), shim mmap.

## Esito in breve
- Osservatore (observe.py) corretto nella sostanza: local = buffer non scalato prima di div_, reduced = buffer ridotto = param.grad bit per bit; copertura 270 una volta per rank garantita dai controlli; offset/stride/dtype giusti; identita' id(param) regge.
- Hook neutrale: su Gloo 4 rank, dtype misti, gradienti subnormali, AdamW + clip, 5 update: con e senza hook bit per bit uguali (gradienti di ogni update e parametri finali), sia ordinaria (bucket di costruzione) sia riscaldata (bucket ricostruiti). Costruzione vs ricostruiti: bit diversi (il meccanismo dell'ipotesi esiste almeno su Gloo).
- Su CUDA 2.2 div_(4) con scalare CPU = mul per reciproco 0.25: stessa operazione del percorso nativo mul_out(·, 0.25). Non verificabile qui: NCCL, stream, ordine delle copie D2H.
- B-1 (gia' noto) confermato con shim; la correzione str(tensor_path) funziona e conserva gli offset.
- NUOVO BLOCCANTE (fuori dal mio perimetro stretto, in compare.py:248/301): l'endpoint della riscaldata ha per costruzione cuda_rng.1..3 diversi dal continuo (lettura cluster: continuo vs CN_A metadata_or_schema_different = cuda_rng.1/2/3) -> require fallisce -> INCONCLUSIVE in ogni caso; anche togliendo il require, exact_payload resta False e SUPPORTED e' irraggiungibile.
- record_optimizer_step: hook pre/post esistono in 2.2, leggono grad dopo clipping e stato subito dopo lo step, rimossi al 546, non alterano nulla (Gloo: 8 righe, traiettoria invariata).
- Costi: +114 MB GPU per rank (picco storico 18,5 GiB su 64), file 228,5 MB per rank per gamba (1,83 GB totali), IO dopo il backward, nessun rischio di deadlock riconoscibile.
