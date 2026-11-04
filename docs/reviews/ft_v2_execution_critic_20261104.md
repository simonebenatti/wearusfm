# Verifica indipendente del benchmark FT v2 e della matrice

Revisore: agente ft_v2_ops_check, 04/11/2026. **APPROVATO, nessun blocco.**

Ricevuta `c14b78b0d593a56f272192bd2c246003e657788551936c24c9dc1fba1eb0241d`:
85 controlli positivi su identità, policy, misure finite, memoria e aritmetica.
Code ae28305, source aa857fb, ricetta 2ba1b7f, checkpoint e input coerenti.
Quattro warm-up e 32 update per regime; solo soggetti train, nessun checkpoint
scientifico prodotto. Unità di valutazione ricostruite: 23.607 in tutti i regimi.

SLURM 63527476 COMPLETED 0:0, 450 s, 1 GPU / 8 CPU / 100 GiB:
0,125 GPUh e 1 coreh, senza sommare batch/extern al job padre.
Host proiettato 32,385 GiB sotto 85; CUDA riservata massima 6,584 GiB,
sotto l'85% dei 63,423 GiB disponibili.

Tutti i singoli candidati entrano nei gate; nessuna coppia entra: split in
frozen, finetune e scratch. Proiezione 122,465173 GPUh; cap 144 GPUh / 1.152 coreh.
Matrice: 24 ID e output unici, copertura esatta 4 fold × 3 regimi × 2 candidati.
I 12 candidati 0 dipendono dal checkout; ciascun candidato 1 dipende dal proprio
candidato 0. Pin, risorse, percorsi WORK e proiezioni coincidono nei piani.
Runtime invariato.

Prima degli invii restano pubblicazione e verifica SHA del nuovo checkout,
assenza di collisioni remote e nel ledger, mapping afterok tramite submitter.
Nessun lancio, modifica al repository o nuovo test della suite dal revisore.
