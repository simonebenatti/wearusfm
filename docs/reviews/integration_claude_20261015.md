# Review finale per l'integrazione di Claude (15/10/2026)

## Stato aggiornato: P2 chiuso e base integrata

Correzione dell'autore `a7c87b5`, isolata dalla base `59b8c5b`, accettata dopo
review indipendente e verifica locale. Merge `a719c2c` conservando gli SHA;
main locale aggiornato da `5030a19`. Nessun push, deploy o nuovo job.
Il prefetch non era incluso nel merge a719c2c: la sua successiva review
separata e' conclusa, con esito e verifiche riportati in fondo al documento.

Verifiche GPT sul candidato con il fix:

- **48 PASS in 159,49 s**: due regressioni originali senza monkeypatch in
  memoria, copie in `tests/cpu`, trainer DDP, sicurezza numerica, ripresa,
  equivalenza e monitor. Overflow su un solo rank e chiavi mancanti coperti
  con gloo e accumulo 1/2; ultimo checkpoint e stati preservati.
- **21 PASS in 3,28 s**: baseline dell'encoder casuale. Totale **69 PASS**.
- Codice del trainer e nuovi test identici alla consegna di Claude; unico
  conflitto nel registro, risolto senza cancellare voci di nessun genitore.
- Nessuna nuova esecuzione della suite completa o su CUDA/NCCL/bf16.

Le sezioni successive conservano la review iniziale e la riproduzione del
difetto sulla base precedente. I due FAIL descritti sotto sono storici:
con il fix committato i test passano senza `WEARUSFM_REVIEW_IN_MEMORY_FIX`.

Fonti: main `5030a19`, base Claude congelata `59b8c5b`, valutazioni GPT `5a25a5c`.
Candidato isolato `add5359` (merge dei due rami, nessuno squash/rebase).
Il prefetch che Claude sta sviluppando non e' incluso. Main non e' aggiornato.

## Esito: un P2 da chiudere prima dell'integrazione

`src/wearusfm/training/run.py`, righe 506-515 della base `59b8c5b`:
il controllo della finitezza dei soli numeratori non sostituisce quello della
loss effettivamente costruita da `LossTerms.backward_loss`.

In floating point il prodotto per il peso o per il numero di processi puo'
andare in overflow prima della divisione. Una loss `Inf` puo' avere gradienti
finiti; quindi il controllo della norma prima di AdamW non garantisce lo stop.
Le metriche ridotte possono risultare finite e nascondere il problema.

Riproduzione sul trainer e sul modello reali, con dati sintetici CPU, keep
calibrata, batch globale 4, accumulo 2, `keep_weight=20` (accettato dalla
configurazione) e bias finito della testa pari a `1e37`:

- numeratori finiti, loss usate nel backward entrambe `Inf`;
- norma del gradiente finita, circa 79,54;
- aggiornamento 1, AdamW/EMA eseguiti, checkpoint salvato, nessuno STOP;
- totale registrato finito, circa `2e38`.

E' un caso limite del contratto di sicurezza, non un errore osservato nei job
reali o una modifica suggerita ai loro iperparametri. Non invalida di per se'
le valutazioni gia' concluse. La guardia sulla loss effettiva esisteva prima di
`fccdc1e`: non va persa quando si anticipa il controllo delle chiavi.

## Correzione minima da affidare a Claude

Mantenere il controllo locale della struttura PRIMA di chiamare
`backward_loss`; se la struttura e' valida, costruire la loss e verificarne la
finitezza nello stesso arresto coordinato, prima di `backward()`.
Non ricreare il precedente problema delle chiavi mancanti su un solo rank.

```diff
-            finite = mismatch or all(bool(torch.isfinite(v).all()) for v in terms.num.values())
+            loss = None if mismatch else terms.backward_loss(global_den, world_size=ctx.world_size)
+            finite = mismatch or (all(bool(torch.isfinite(v).all()) for v in terms.num.values())
+                                  and bool(torch.isfinite(loss).all()))
             flags = _all_max(ctx, device, [float(mismatch), float(not finite), float(h != first_hash), h, -h])
 ...
-            loss = terms.backward_loss(global_den, world_size=ctx.world_size)
             loss.backward()
```

Patch proposta, NON applicata a `run.py`. SHA di Claude conservati.
Per non includere il prefetch in questa integrazione, la correzione deve essere
consegnata come delta separato dalla sua nuova ottimizzazione.

## Regressione riproducibile

`docs/reviews/repro_backward_loss_overflow_20261015.py` contiene due test con
aspettative sane: stop prima di backward/optimizer/EMA, nessun checkpoint
nuovo, stato invariato e STOP rispettato da una ripresa successiva.

Da un checkout che contiene il trainer esaminato:

```sh
PYTHONPATH=src:tests/cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /Users/simonebenatti/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider docs/reviews/repro_backward_loss_overflow_20261015.py
```

Attesi **2 FAIL sulla base attuale**. Anteponendo
`WEARUSFM_REVIEW_IN_MEMORY_FIX=1`, il test applica soltanto una monkeypatch in
memoria della guardia proposta: **2 PASS**, senza modificare i sorgenti.
I test sono in `docs/reviews`, non nella normale raccolta `tests/cpu`.

## Altre verifiche

- Review indipendenti di trainer e rami/sonde: nessun altro rilievo concreto
  nel perimetro esaminato; micro-batch locale senza target JEPA verificato.
- Keep raw separata dai rami, rami spenti per default, identita' delle cache,
  provenienza e allineamento ordinato degli ingressi ricontrollati.
- Sintassi dei launcher shell nuovi/modificati verificata con `bash -n`.
- Conflitti limitati ai due registri: unione senza cancellazioni rispetto a
  ciascun genitore (`git diff --numstat`: zero righe rimosse nei due file).
- Suite CPU generale interrotta dopo 623 PASS, 1 SKIP, 6 FAIL e 5 ERROR:
  cinque FAIL e cinque ERROR sono legati a socket localhost/memoria condivisa
  negati dal sandbox; un test con worker e' rimasto in attesa. Il rerun dei sei
  file interessati fuori sandbox e' terminato con **65 PASS in 187,67 s**, senza
  GPU o accesso al cluster. Non e' un PASS della suite completa del candidato.
- L'altro FAIL era un'assunzione del test GPT sulla baseline: imponeva che
  il nuovo checkout fosse ancora byte-identico alla pipeline storica. Dopo
  il merge dei rami questo non e' vero e la guardia GIUSTAMENTE lo rifiuta.
  Aggiornati soltanto i test GPT: fixture dei sorgenti storici accettata,
  encoder e pooling modificati rifiutati. **21 test della baseline PASS**;
  guardia di produzione, SHA iniziali e campagna chiusa invariati.
- Restano solo le due regressioni numeriche intenzionalmente fallite nel
  perimetro esaminato; non e' stata applicata la correzione al trainer.
- Nessun job, accesso SSH, GPU, push o modifica al worktree di Claude.

## Review separata del prefetch: accettato (15/10/2026)

Fonte Claude `8cc6cbf`: prefetch `0230775`, esiti GPU `85ac845`, fix numerico
`a7c87b5` gia' ricevuto nel merge `66b5a0f`. Nessuna modifica al codice di
Claude nella risoluzione del candidato: conflitti soltanto nei due registri.
Due review indipendenti senza rilievi bloccanti su stream/eventi CUDA,
`record_stream`, lifetime dei buffer, replay/RNG, accumulo e checkpoint WSD.

Verifiche GPT:

- **58 PASS in 376,99 s**: prefetch, trainer DDP, ripresa dello stream, WSD,
  backward-loss finita, ricetta della ripresa e pannello diagnostico.
- **6 controlli aggiuntivi PASS**, persistenti in
  `docs/reviews/repro_prefetch_wsd_20261015.py`: WSD con prefetch prima/al
  confine/dentro il decay, rami stable/decay/continuazione e ripresa DDP con
  accumulo 1/2; identita' di optimizer e generatori salvati oltre ai pesi.
- Totale **64 test CPU PASS**. Il primo comando supplementare ha selezionato
  anche i due test DDP (il nome del file contiene WSD): 4 PASS e 2 errori
  di socket nel sandbox; i due casi DDP sono passati fuori sandbox in
  27,96 s. Non sono difetti di codice e non sono una nuova prova NCCL/bf16.
- Prova GPU precedente gia' PASS: 200/200 audit input su 4 processi,
  attesa 30M 0,673%/0,649%, tempi entro i limiti preregistrati. Non rieseguita.

Limiti: suite completa non rieseguita da GPT; 946 PASS e' il risultato di
Claude sul suo ramo. Non esercitati guasti CUDA/OOM, fetch bloccato o
cleanup con traceback trattenuto; nessuna regressione di campagna dimostrata
da questi casi. La chiusura esplicita e' sul percorso normale, mentre un
errore propagato alla CLI termina il processo. Nessun nuovo job o budget.
