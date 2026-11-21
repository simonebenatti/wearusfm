# Verifica avversaria di R4-I1 (test di compare deboli)

Esito: il rilievo REGGE nei fatti; gravita' giusta "importante" (non bloccante). Una parte della motivazione va ridimensionata.

## Cosa ho eseguito
- Rieseguito `agents/r4-confronto/mutate.py` con la mia cartella di lavoro
  (`agents/verifica-r4-confronto-R4-I1-1/mut/`), venv wearusfm-tok, un thread, 64 s.
  Il modulo mutato e' davvero quello importato: il test sostituisce la propria `sys.path.insert` con la cartella mutata
  (posizione 0, davanti a PYTHONPATH) e 10 mutazioni vengono uccise (M01, M02, M11, M12, M14, M16, M19, M21, M22, M23).
- Risultato identico al revisore: SURVIVED M03-M10, M13, M15, M17, M18, M20, M24-M30 (20 su 30), ciascuna "13 passed".
- Letto `tests/cpu/test_cn_restart_compare.py`: l'unico test che chiama `compare.main()` (riga 72) lo fa su directory
  inesistenti e verifica solo che l'esito sia INCONCLUSIVE; `compare()` non arriva mai a `classify` con osservazioni sintetiche.
  Gli altri test esercitano funzioni singole (classify, compare_vector, checkpoint_differences, layouts, branch_traces,
  check_boundary, post_update_hashes) su archivi veri.

## Cosa deduco dal codice (non eseguito)
- Le guardie sopravvissute sono presenti e, alla lettura, corrette nel codice candidato
  (`scripts/diagnostics/contrastive_cn_restart_compare.py:214-304`): soglia `> LIMIT` (r.283), `any_local` sugli elementi
  diversi (r.281), gate di fedelta' dell'ordinaria (r.245-246), metadati della riscaldata (r.247), placement (r.296-298),
  uguaglianza fra rank (r.262-266), first_exact (r.299). La mutazione dimostra assenza di copertura, non un difetto attuale.
  Nessun controesempio di run vero reso INCONCLUSIVE o mal classificato deriva da R4-I1 in se'.
- Mitigazione: `compare` e' un passo CPU che legge solo file persistiti (`root/ordinary`, `root/warm`, checkpoint,
  osservazioni). `contrastive_cn_restart.py:166-176` lo lancia dopo le due celle e nessuno script del diagnostico cancella
  le uscite (grep di rmtree/unlink/os.remove: nessun risultato). Un errore di compare scoperto dopo e' quindi rimediabile
  con una nuova esecuzione CPU del confronto (con codice corretto e review), senza consumare il tentativo GPU.
  Per questo non e' bloccante.
- Ridimensionamento: la frase "R4-B1 e' proprio il tipo di difetto che un test end-to-end avrebbe trovato" non e' verificata
  qui. In particolare il difetto B-1 (torch.load mmap con Path) NON sarebbe emerso da un test end-to-end sul Mac: torch 2.14
  accetta Path con mmap=True; servirebbe lo shim del torch 2.2.

## Perche' resta importante
Il diagnostico ha un solo tentativo e l'esito (SUPPORTED/PARTIAL/UNSUPPORTED/INCONCLUSIVE) lo decide interamente compare().
Venti guardie che determinano esito o inconclusivita' (soglia, gate di fedelta', join del confine) possono essere
alterate da una modifica futura senza che la suite se ne accorga; una regressione introdotta prima del pin uscirebbe in un
report sbagliato, recuperabile solo a posteriori.

## Correzione minima proposta
Un test di `compare()` su directory sintetiche (4 rank, pochi parametri, monkeypatch di ARCHIVE/ENDPOINT/PARAMETERS e delle
impronte storiche) con un caso per esito: SUPPORTED, PARTIAL_SUPPORT, HYPOTHESIS_UNSUPPORTED (locale diverso, scarto
ridotto > 1e-6 applicato dentro compare, nessuno scarto), INCONCLUSIVE per fedelta' dell'ordinaria. Rieseguire mutate.py
come criterio: almeno M03-M07, M15, M17 devono essere uccise. Facoltativo, con shim torch 2.2 per torch.load(mmap=True).
