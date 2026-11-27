# Controprova D3-2 (controllo con "$CODE_DIR" al passo 4)

Esito: il rilievo regge nella sostanza, ma il fatto portato come prova è sbagliato. Gravità corretta: minore.

- Sequenza, riga 111: «Verificare poi `git -C "$CODE_DIR" rev-parse HEAD` e lo stato pulito», senza ssh. CODE_DIR compare solo nella tabella (riga 53) e dentro il blocco remoto del passo 5 (riga 125). Il brief non la nomina mai (grep: 0 occorrenze).
- Il fatto è falso: sul Mac il comando NON fallisce. Con CODE_DIR vuota, `git -C ""` resta nella cartella corrente. L'ho eseguito dal worktree di release con CODE_DIR non impostata: stampa `b58a66f695c8744dfa5fe633829cafe13a5a1fc4`, esce con 0, e `git status --porcelain` dà 0 righe. Il passo 4 dice di lavorare «dalla radice del worktree di release», quindi la verifica passa in silenzio e controlla il repo sbagliato, cioè quello del Mac e non il checkout su Leonardo.
- La conseguenza è limitata. Al passo 5, `code_dir_setup` (lib_code_dir.sh, righe 25-60) controlla sul cluster che HEAD sia uguale a EXPECTED_SHA, che il checkout sia pulito e che sia in sola lettura, e lo fa prima del job GPU. Con `set -eo pipefail` un checkout sbagliato ferma la validazione. Nessun tentativo viene bruciato e nessuna regola viene violata.
- Correzione: scrivere il comando via ssh con il percorso per esteso, oppure togliere la frase e rimandare a `code_dir_setup` del passo 5.
