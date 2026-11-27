# Controprova R1 (billing e saldo degli impegni) — esito: REGGE, gravità importante

Verifiche fatte (sola lettura):
- Protocollo `docs/contrastive_cn_restart_probe_20261121.md` riga 43 (worktree di release): chiede di verificare «saldo netto degli impegni, billing corrente, quota ...». Confermato alla lettera.
- `docs/contrastive_targets_execution_20261119.md` riga 95: «Il launcher controlla il billing reale prima dell'invio». Confermato.
- `docs/decisioni.md` riga 8409: il preventivo distingue il residuo contabile dal saldo del progetto verificato dal launcher. Confermato.
- Precedenti: `reloads_execution_20261121/checkout_operational_snapshot.txt` contiene l'uscita di `saldo` (riga IscrB_WearUsFM, 67,3% consumato) e la coda; `readout_execution_20261121/checkout_initial_status.txt` riga 7 contiene `billing=4,cpu=4` in ReqTRES/AllocTRES. Confermato.
- Sequenza, passo 3 (righe 87-93): solo cartelle assenti, quota lfs, coda/posti dbg, reservation. Passo 6 (riga 147): solo quota e cartella. `grep -i "billing|saldo"` su sequenza e brief: nessuna occorrenza. Nessuna occorrenza nemmeno nelle letture del 27/11 in `rev_probe2/leonardo/`.
- Brief, passo 3 (riga 17): stessi controlli della sequenza, senza saldo né billing.

Gravità: non bloccante. Il saldo del progetto (circa 650.000 ore locali residue all'istantanea del 21/11) rende trascurabili le 8,27 ore locali della prova, e l'assenza della lettura non fa fallire il lancio né brucia un ID. Resta un controllo che il protocollo richiede e che i precedenti hanno archiviato, quindi «importante».
