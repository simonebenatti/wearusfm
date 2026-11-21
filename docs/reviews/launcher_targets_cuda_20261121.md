# Collaudo CUDA RMS100/fast125 — esecuzione operativa del 21 novembre

Istruzioni b0ce039d, esecuzione congelata 4e880b9f6787d7294712370d00ed485bfbe45735.
Piano rigenerato e confrontato byte per byte; pin e submitter comuni verificati,
nessun ID preesistente. Ref deploy pubblicato e SHA remoto verificato.

Prima di ciascun invio: saldo/quota/reservation/slot/ledger e calendario,
controlli login congelati con verifica dei 14 artefatti della calibrazione,
checkpoint originale, accounting audit/calibrazione, binding review,
make-job/cmp, build_command e validazione della ricetta. Stdout e snapshot
in results/infra/contrastive_targets_20261118/cuda_execution_20261121/.

Riserva 12 GiB per coorte più 18 GiB prudenziali per altre scritture;
WORK libero oltre 40 GiB. Saldo 653.928 ore locali, nessun job debug sbenatti.
Corsie UP; manutenzione confermata 23/11 08:00 CET–25/11 08:00 CET.
Anche includendo controlli e margine, la coppia entra nella finestra disponibile.

Checkout **65016695 COMPLETED 0:0**, 28 s, 4 CPU/2 GiB/4 minuti.
SHA, pulizia e permessi vanno confermati da before_cuda sul checkout effettivo
prima del secondo invio. La consegna termina con la raccolta del collaudo:
nessun benchmark o training viene rilasciato automaticamente.

before_cuda superato sul checkout effettivo: HEAD, pulizia, permessi,
binding/audit/calibrazione, job rigenerato e argv verificati nuovamente.
Secondo ID **65016794**, collaudo 4 GPU/32 CPU/240 GiB/30 min, export con
TORCH_DIST_INIT_BARRIER=1 attestato dalla ricevuta. Un solo invio per ID.
Esito e raccolta da completare.
