# Review GPT della sola correzione del cancello scaling

**APPROVATO, nessun blocco residuo nella correzione.** Il rilievo sulla lettura
fallita della coda e' chiuso. Claude puo' inviare i sei aggregati gia' rivisti
dopo `state` e `check` freschi, sei posti liberi e gli altri controlli del piano,
una riga alla volta, fermandosi al primo esito diverso da `sottomesso`.
Non serve un altro consenso di Simone. Nessun rilancio implicito.

## Perimetro e identita'

Commit esaminato: `1c831896ec85a20388b906404c88d425315bbe68`, padre
`37632d18e0078069c54d131c9ba235891712c1b5`, ramo Claude di esecuzione.
Diff: quattro righe aggiunte a `final_stages.py` e il selftest locale nuovo.
Helper SHA256 `b222474b4fad9fe1a810df115ebc5eb4d8d9246408e14e833abad8b8ba61924e`.

Controllata fino alla punta dichiarata `4b567d52755d59bcb7ffa5923de8436afd82a8b5`
l'invarianza dell'helper, del piano e dei comandi. I due commit successivi
aggiungono soltanto tre file; **comparison_stage.py e il suo selftest non sono
esaminati ne' approvati da questa review**. Nessun manifest o invio del confronto
e' coperto. Il checkout metadata residuo e il confronto restano nel secondo giro
di review gia' previsto.

## Esito delle tre verifiche richieste

1. In testa a `checks`, entrambi gli exit di `sacct` e `squeue_serial` devono
   essere interi esattamente uguali a zero. Il rifiuto precede l'accesso agli
   stdout, la lettura dei metadati tramite `fits()` e qualsiasi conteggio.
   `cmd_check` e `cmd_plan` passano entrambi da questa funzione.
2. `squeue exit=0` con stdout vuoto resta legittimo. Valori mancanti, booleani,
   float e stringhe non vengono confusi con l'intero zero.
3. Nessun altro cambiamento del cancello o dei sei argv rispetto alla consegna
   rivista. Piano, righe d'invio e due dry-run identici da `37632d1` a `4b567d5`;
   nessuna modifica a `scripts/`, `src/`, `tests/`. Numerica frozen `aef665b` intatta.

| Artefatto invariato | SHA256 |
|---|---|
| Piano aggregati | `abc489613dfbfddc5e33e2150854245bcd3b69292806953bafa622b91a22941c` |
| Sei righe d'invio | `bf9823e956bacdaba024bd062be8dc4a8f84f3b20aaf13cd472979d9e6727b3f` |

## Prove locali eseguite

- Rieseguito `python -I final_stages_selftest.py` da copie esatte dei blob Git,
  in un albero temporaneo separato: **10/10 passati**.
- Stesso selftest sul vecchio helper `9533c82f...`: **sei casi falliscono**,
  compreso il falso PASS con squeue fallito e vuoto. Regressione significativa.
- **14/14 prove indipendenti** sulle sole funzioni estratte via AST: errori,
  exit mancanti, booleani e tipi non interi per entrambi i comandi; stdout assenti
  e `fits()` vietato. Tutti rifiutati prima di raggiungere parsing o metadati.
- Rivalidato offline lo snapshot storico di `a4cded0`, delle **12:49:51**:
  SHA `08767e9cacb921d6e533e72990f743ef7d6605870c0c7ab04b181965bc2d288d`,
  squeue riuscito/vuoto, sacct riuscito, cancello superato.
  **Non e' un controllo corrente e non sostituisce quello prima dell'invio.**
- Seconda review indipendente concorde sul diff, sull'ordine delle guardie e sui pin.

La riproduzione storica GPT in `scaling_aggregate_queue_guard_20261106_check.py`
rimane legata al vecchio SHA: non la si ripunta al nuovo helper senza adeguarne
l'aspettativa sullo stdout vuoto. L'IndexError di quel solo adattamento non e'
un difetto del nuovo cancello; le prove qui registrate gestiscono il rifiuto
anticipato senza presumere righe stampate.

Output dei test, hash e controlli in
`results/infra/scaling_300m_20261104/queue_guard_review_20261106.json`.
Solo letture Git e prove locali: nessun SSH, deploy, invio, cancellazione,
runner, modifica al worktree Claude o al codice operativo da parte di GPT.
Restano 310 core-ore cumulative, S8 e perimetro approvati; nessun seguito a 19.618.
