# Controprova R2 (quota senza impegni e scritture concorrenti)

Esito: il rilievo REGGE. Gravita' corretta: importante.

## Il fatto e' vero
- Protocollo (worktree di release, docs/contrastive_cn_restart_probe_20261121.md, riga 43): chiede di verificare
  «saldo netto degli impegni, billing corrente, quota e scritture concorrenti» e che «restino almeno 4 GB per questa
  prova dopo gli impegni concorrenti».
- Sequenza (docs/contrastive_cn_restart_probe_launch_20261127.md, righe 90-92): «almeno 4 GB oltre al checkout» e
  «Coda dell'utente»; riga 147 (passo 6): «rileggere la quota». Nessun riferimento agli impegni o scritture concorrenti.
- Brief, riga 17: «quota (almeno 4 GB liberi oltre al checkout), coda». Stessa omissione.
- Lettura del 27/11 (stato_pre_lancio_20261127.out.txt): 991G usati su 1T; la coda letta e' solo quella dell'utente
  (una riga, 65361388 SCALING-49046, PENDING, 10 h, boost_usr_prod).
- Istantanee storiche della linea 3 con coda di progetto: results/infra/contrastive_targets_20261118/reloads_execution_20261121/checkout_replay_snapshot.meta.json contiene squeue con -A.

## Tentativo di confutazione, parziale
- Il job 65361388 citato scrive i risultati su $SCRATCH (ledger, riga 1889: ultimo argomento
  /leonardo_scratch/large/.../scaling_49046_20261118/training/49046), quindi non e' un buon esempio di scrittura
  concorrente su WORK. Questo indebolisce l'esempio, non il rilievo: gli altri 10 utenti della quota non compaiono
  nella coda dell'utente.
- Con circa 33G liberi oggi il margine di 4 GB e' ampio, quindi seguire la sequenza alla lettera non farebbe fallire
  il lancio con certezza: non e' bloccante. Resta pero' una deviazione da un controllo richiesto dal protocollo
  approvato, con rischio concreto su una quota condivisa: importante.

## Non verificato
- Quanto scriveranno davvero su WORK, nella finestra, i job degli altri utenti del progetto (nessuna lettura di coda
  per account disponibile).
