# Review indipendente — risultati fine-tuning EPN, 05/11/2026

Revisore: subagent `consolidation_critic`, sola lettura. Non è la review esterna
di Claude. Nessuna modifica al repository, inferenza, SSH o sottomissione da parte
del revisore. Verdetto: **nessun difetto bloccante nell'aggregatore**.

Prima del confronto con l'aggregatore, il revisore ha ricalcolato da zero i
punteggi e il bootstrap dalle 12 OOF, verificando 84 file contro manifest,
hash e dimensioni. Il ricalcolo e `scripts/consolidation_ft_report.py`
coincidono entro 1e-12 nei tre punteggi, nei rispettivi IC95% e nei due
contrasti appaiati. Evidenza numerica in
`results/infra/consolidation_20261104/final_20261105/independent_review.json`.

Controlli: provenienze C0→C1 e source_receipt, selezione sulla sola validation,
regole dei pareggi, ruoli dei 314 soggetti da riferimento storico pinnato,
copertura delle 47.100 righe, pairing OOF, stringhe native, 24 hash checkpoint,
macrorecall pooled e applicazione di F-R2. Bootstrap per soggetto appaiato:
2.000 campionamenti, seed 20261102, nessuno scarto e nessuna classe mancante
nei campionamenti di questi dati. Tutti i 24 candidati completano cinque epoche.

Due limiti richiesti dal revisore, recepiti nell'analisi:

- F-R2 scatta con scratch [4,5,5,5]: nessuna narrazione di ottimo raggiunto e
  nessuna estensione automatica. Vantaggio FT−scratch +1,403 pp [0,930;1,837]
  riferito alle cinque epoche.
- Frozen 70,575%, con migliore epoca 5 in tutti i fold, è circa 12 punti sotto
  la logistica storica 82,577%. Il +23,191 pp FT−frozen riguarda questo lettore
  e questa dose, non il miglior risultato disponibile a encoder congelato.

Il revisore ha eseguito l'aggregatore sul reale in un output temporaneo;
l'esecuzione dell'owner produce un JSON identico. I 15 test della nuova
consegna sono stati rieseguiti dall'owner: tutti passati in 1,13 s.
