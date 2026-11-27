# C1 - Correzioni al confronto (contrastive_cn_restart_compare.py), diff 9a2ce7e7..8c72655c

Esito: nessun bloccante e nessun importante. Due note minori. B-2 e il punto sugli scalari sono corretti come da regola del 22/11.

Prove eseguite (cartella /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_probe2/agents/c1-confronto):
- check_c1.py / check_c1.out: 27 casi B-2 con payload nel formato vero (torch_rng 5056 B uint8; cuda_rng lista di 4 x 16 B uint8; rank_rng lista di 4 dict {torch_rng, cuda_rng}); classify sulle 32 combinazioni. Tutti OK.
- scalar_sim.py / scalar_sim.out: 200.000 casi fp32 a 4 addendi (24 ordini sequenziali + 24 a coppie), meta' con quasi annullamento. Rapporto vecchio: massimo 2,0, 97.208 casi sopra 1e-6. Rapporto nuovo: massimo 2,34e-7, 0 casi sopra 1e-6. Limite analitico 6u = 3,58e-7. Eccessi veri 1e-3 e 5e-6 rilevati su entrambi i bias.
- pytest test_cn_restart_compare.py + test_cn_restart_end_to_end.py: 27 passati.
- Modello dal config archiviato: 270 parametri; gli unici di 1 elemento sono 253 anchor_heads.rms.bias e 257 anchor_heads.env.bias.

Il file della lettura su Leonardo (confronto_checkpoint_veri_20261121.out.txt) non esiste piu' nello scratchpad (cartella leonardo vuota); il formato e' stato preso da run.py:1482, 1593-1597 e dalla review fc3eef6b, sezione B-2.

Note: (N-1) percorso vuoto "" nel motivo quando cambia una chiave di primo livello; (N-2) una differenza non float nella riscaldata da' INCONCLUSIVE anche se i locali differiscono (coerente con la regola, ma precedenza da sapere).
