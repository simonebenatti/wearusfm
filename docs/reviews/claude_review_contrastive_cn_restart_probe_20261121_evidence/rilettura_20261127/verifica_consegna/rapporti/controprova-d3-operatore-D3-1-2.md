# Controprova D3-1 (passo 3 senza comandi)

Esito: il fatto regge (il passo 3 alle righe 87-93 della sequenza è un elenco senza comandi; «Conservare l'uscita» alla riga 91 non dice dove; la cartella locale nasce solo alla riga 120). La gravità «importante» però non regge: la porto a **minore**.

Perché le conseguenze sono piccole:
- Il passo 3 è tutto in sola lettura. Un comando scritto male (per esempio `$WORK` espanso sul Mac) dà un errore o un'uscita strana. Il brief (VINCOLI, riga 26) dice di fermarsi davanti a un errore inatteso. Il costo è un ritardo, non un ID bruciato.
- Cartella del checkout già presente: campaign_checkout.sbatch riga 35 esce con codice 2 senza toccare nulla. Cartella di uscita già presente: il driver la rifiuta, e la sequenza la ricontrolla subito prima del passo 6 (riga 147). Le due cartelle mancavano il 27/11 alle 15:08 (stato_pre_lancio_20261127.out.txt) e possono crearle solo questi due job.
- Slot dbg pieni o manutenzione: il job resta PENDING, non fallisce.
- Il passo 5 mostra già lo schema `ssh leonardo 'bash -s' ... <<'EOS'` da imitare. Il protocollo di GPT (riga 43) dà i comandi lfs esatti.

Resta un rischio reale ma piccolo: una lettura sbagliata della quota, e un'uscita conservata senza un posto fissato. La correzione proposta (blocco ssh unico, mkdir al passo 1) è utile e costa poco.
