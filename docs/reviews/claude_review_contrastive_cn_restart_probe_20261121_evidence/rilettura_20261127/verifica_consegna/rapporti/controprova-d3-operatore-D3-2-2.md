# Controprova del rilievo D3-2

Esito: il meccanismo descritto nel rilievo è falso. Resta un'imprecisione, di gravità **minore**.

## Verifica fatta
Sul Mac, nel worktree di release, con CODE_DIR non impostata:
    git -C "$CODE_DIR" rev-parse HEAD   -> b58a66f695c8744dfa5fe633829cafe13a5a1fc4, rc=0
    git -C "$CODE_DIR" status --porcelain | wc -l   -> 0
Con `-C ""` git resta nella directory corrente. Il passo 4 dice di lavorare «dalla radice del worktree di release», quindi
il comando non fallisce: restituisce lo SHA atteso e lo stato pulito, ma del worktree del Mac e non del checkout sul cluster.
L'operatore quindi non si ferma, come invece sostiene il rilievo. Ottiene però una conferma che non prova nulla.

## Conseguenze
Il tentativo non si brucia e il lancio non cambia esito. Al passo 5, sul cluster, `code_dir_setup`
(scripts/slurm/lib_code_dir.sh, righe 20-50) controlla davvero HEAD = EXPECTED_SHA, la pulizia e la sola lettura
del checkout. Inoltre il passo 4 fa già leggere nel log la riga «checkout di campagna pronto: … @ b58a…».
Il controllo errato è quindi ridondante.

## Correzione
Togliere la frase, oppure dire che la verifica la fa `code_dir_setup` al passo 5. In alternativa scrivere il percorso
per esteso, da eseguire via ssh.
