# R6: i test e la loro forza (completo)

Clone al commit 9d34dc69, detached e non modificato (`git status` pulito). Cartella di lavoro:
/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_probe/agents/r6-test
(script: mutate.py, mutation_log.txt, shim/torch22_load_shim.py, layoutcheck/check_layout.py, e2e/e2e_compare.py)

## Eseguito
- I 4 file nuovi: 32 passati in 8,40 s. I 7 file già presenti: 138 passati in 24,82 s. Totale 170, come nella ricevuta (che dichiara 34,83 s).
- I due file Gloo (leg e observe) rilanciati altre 2 volte: 12 passati ogni volta (7,08 s e 6,30 s). Nessun test instabile.
- `--package` restituisce b0640ca6…dcfa3 e 9d5811b6…d6: coincidono con il blocco della richiesta.
- Shim torch 2.2 (simulazione: `torch.load` solleva ValueError se mmap=True e f non è str): test_cn_restart_observe dà 1 fallito e 7 passati. I 7 test parametrizzati di rifiuto passano lo stesso, perché `pytest.raises(ValueError)` senza `match` accetta anche l'errore dello shim.
- layouts() con i dati di logging DDP veri (torch 2.14, Gloo, 4 rank, mixed dtype, riscaldamento): esito LAYOUTS_OK, con 2 bucket nella ripresa ordinaria e 7 in quella riscaldata.
- compare() end-to-end in simulazione: tracce, metriche, config, start, summary e DDP log veri di CN_A e CN; inventario vero dei 270 parametri; osservazioni e checkpoint sostituiti. Ottenuti PARTIAL_SUPPORT, SUPPORTED e HYPOTHESIS_UNSUPPORTED. Con lo shim torch 2.2 sul percorso di load_observation il risultato è ValueError, quindi B-1 si riproduce.
- Mutazioni: 84 applicate, una alla volta, a una copia; 30 uccise e 54 sopravvissute.
  - Uccise: O6-9, O11, O12; C2-7, C14-16, C18, C22, C24, C25; L1-4, L7, L10, L11, L18, L19; G2, G3.
  - Sopravvissute: O1-5, O10, O13-17; C1, C8-13, C17, C19-21, C23, C26-30; L5, L6, L8, L9, L12-17, L20, L21; G1, G4-16.

## Rilievi
Il dettaglio è nell'oggetto strutturato restituito al chiamante (R6-1 … R6-8).
