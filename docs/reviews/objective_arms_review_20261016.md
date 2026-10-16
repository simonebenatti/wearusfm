# Review indipendente del punto 5: JEPA-only e ancore-only

16/10/2026, GPT. Codice esaminato: `b1238a1` + `fe67339`, presente in `main` e worktree GPT a `2dc1197`.
Letta anche la firma del protocollo sul ramo Claude (`0d45a2f`, ultimo registro letto `e79a536`).

**Esito: favorevole all'implementazione dei due bracci. Nessun difetto confermato nella review mirata.**
Non equivale alla chiusura scientifica del punto 5: mancano ancora esecuzione e confronto dei bracci.

## Verificato

- Il totale applica separatamente `jepa_weight` e `anchor_weight`, anche nei numeratori del contratto distribuito. I conteggi restano
  quelli delle finestre valide; il peso zero mantiene il grafo richiesto da DDP.
- Ancore-only: cambiare il teacher non cambia i gradienti dello studente, a parita' di ingressi e target delle ancore. Le ancore leggono
  il decoder dello studente e allenano tramite esso l'encoder. Il teacher resta la sua EMA, valutata dalle sonde come nel riferimento.
- JEPA-only: le ancore contribuiscono gradiente zero; i gradienti attivi coincidono con quelli della sola JEPA. Le teste inattive
  rimangono nell'architettura. Gradiente zero non significa pesi congelati: AdamW puo' applicare il weight decay, senza introdurre
  supervisione dalle ancore nell'encoder.
- Default: perdite, termini distribuiti e gradienti identici bit per bit al `jepa.py` precedente (`b8a2f35`), per target a/b e riduzione
  per token/per finestra. I pesi non validi sono rifiutati e le opzioni CLI arrivano alla configurazione.
- DDP e accumulo: aggiunto un confronto indipendente dei due bracci, in float64, fra batch globale, un processo con due micro-batch,
  due processi con uno/due micro-batch. Verificati metriche, gradienti, pesi dopo clipping/AdamW e teacher dopo EMA (tolleranza 1e-10;
  diagnostiche esplicitamente float32 con la tolleranza del test di riferimento). Batch con conteggi eterogenei e RMS assente sul rank 0.
- Ripresa: la validazione accetta il vecchio default implicito `jepa_weight=1` e rifiuta cambi di entrambi i pesi nei due bracci.
  I test aggiunti verificano questo vincolo della ricetta, non una nuova prova end-to-end della ripresa dello stream.

## Protocollo

La firma `0d45a2f` risolve i punti precedentemente aperti: allarme di collasso applicato anche ad ancore-only; braccio collassato
registrato come tale e come «non batte A»; conclusioni condizionate al LR calibrato su A. Non riproporre queste scelte come da firmare.
L'ablation misura l'utilita' delle **ancore attuali** e della JEPA nelle condizioni firmate (30M, 4 epoche, due semi, LR di A).
Non misura l'ottimo di ciascun obiettivo dopo calibrazione separata. La review indipendente del protocollo non ha trovato
contraddizioni confermate con il trainer.

Restano attivita' gia' dichiarate: pianificatore dei bracci e piano P1/P2, collaudo GPU breve dei due nuovi bracci, esecuzione e
confronti; per la keep separata serve la ricalibrazione sul corpus intero. Sonde per profondita', temporali/fisiche e altre analisi
del punto 5 non sono certificate da questa review circoscritta.

## Prove eseguite

- `tests/cpu/test_objective_arms.py` + `tests/cpu/test_jepa.py`: **41 casi passati**. Primo giro: 39 PASS e due test torchrun bloccati
  dal sandbox prima dell'avvio dei processi (bind localhost); rieseguiti i soli due con accesso locale: 2 PASS in 12,65 s.
- `docs/reviews/test_objective_arms_review_20261016.py`: **4 PASS in 16,34 s**. Comando nel file; CPU, gloo, un thread per libreria.
- Totale: **45 casi distinti passati**. Nessuna suite completa, prova CUDA/bf16, job o modifica al codice di produzione in questa review.

I nuovi test riusano i costruttori di batch e l'oracolo del test DDP esistente, applicandoli ai due pesi estremi prima non verificati
per equivalenza numerica completa; gli smoke test originari controllavano solo che il training distribuito arrivasse alla fine.
