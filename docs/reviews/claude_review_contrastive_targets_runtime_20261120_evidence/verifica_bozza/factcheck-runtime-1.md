# Fact-check bozza runtime (verificatore 1)

In corso. Fonti: clone a 0efa4d45, workflow_result.json, reports, leonardo_torch_api.out.txt.

## Conteggi workflow (verificati)
- 54 rilievi: 1 BLOCCANTE, 18 MINORI sopravvissuti (incl. G-2 per gravita originale), 26 NOTE sopravvissute, 9 non sopravvissuti (R1-M2, R1-N1, R1-N3, R1-N4, R3-M2, R4-M1 voto diviso, R4-N2, R4-N6, R6-4).
- 77 voti di scettici + 7 revisori + 1 critico = 85 agenti.

## Verificati corretti (sintesi)
- Perimetro: 46 file 438c8d65..fbfde797; 0efa4d45 aggiunge solo delivery_pin.json.
- Tabella: schema, reviewed_commit, source 40d39f56... (ricalcolato, 125 file), plan b03c0bc6... 14862 B, preflight e4b39591... 9082 B, identici ai due commit.
- Citazioni faa4a8d7, prompt, decisioni 19/11, 17/11: letterali (a meno degli apostrofi).
- Tetto 18/169,6, inviluppo 21,6/202,4; storico 6,086667/57,827778; cumulato 24,086667/227,427778; 17 allocazioni, 7,9 h.
- P-4: default checkout 1 CPU/10 min; caso 4 CPU x 10 min x 6 = 4,0 coreh -> 172,0. Override 4/2G/00:04:00.
- P-6: 6,4 h in serie (0,4 checkout + 2 + 0,5 + 0,5 + 1 + 1 + 1), 5 rilasci intermedi, dbg max 2 job.
- torch.load con mmap: 10 chiamate, tutte str(...).
- Blob review 94f5094 (c25d4f6f) e faa4a8d7 (5bdb0d15) identici nel ramo.
- 85 agenti = 7 + 77 + 1; G-1 3/3; conteggi 18/26/9 con la gravita originale.

## Problemi trovati (bozza di lavoro)
1. Intestazione "Rilievi minori (correzioni di codice: cambiano il digest)" ed Esito r.52: M-c (test) e M-e (CLAUDE.md) non cambiano source_digest (inventario: src/wearusfm/*.py, scripts/contrastive_*.py, scripts/slurm/contrastive*.sbatch, train_jepa.py, lib_code_dir.sh). G-3 stesso: "I test sono fuori dal digest"; scettici G-2: "CLAUDE.md e' fuori da source_digest".
2. M-e: entrambi gli scettici di G-2 declassano a NOTA e negano "stessa ambiguita' di d9020c20" e "file letto da tutti gli agenti".
3. R3-N1 nelle Note: la correzione (ancorare + rifiutare duplicati) non copre il caso della review non favorevole con esempio APPROVED (scettico R3-N1).
4. Lacune del critico: non tutte in P-1...P-8 (gap 2, 5, 6, 13, 15 assenti dalla bozza; 8, 10 solo come note R3-N3/R6-7; 9 in G-1).
5. Riga 7: "non ASCII potrebbe far fallire ogni job con un locale non UTF-8": con Latin-1 o C/POSIX (Python 3.11) non fallisce (scettico R3-N2).
6. P-7: "si ricostruiscono alla seconda iterazione" vs dati: ricostruzione dalla terza update (iteration 2 a base 0), coerente con 545-546.
7. "9 rilievi confutati dagli scettici": R4-M1 ha voto diviso 1-1.

## Aggiornamento (controlli completati)
Altri punti verificati corretti: quote del prompt (r.14-16), R-1..R-4/N-1..N-7 in 94f5094, TORCH_DIST_INIT_BARRIER (default 0, "Update 04/2023"), CUDA oracolo a batch 8, GPU su boost_qos_dbg, build_command senza variabili SLURM, G-4 (code_sha_from cerca solo EXPECTED_SHA=), cancello sacct nel codice, make_job accetta 6 modi con lo stesso binding, 6 chiavi esatte nel blocco, bozza ASCII (0 righe non ASCII), cartelle di prove esistenti.

Problemi aggiuntivi:
8. P-1: il rank 0 non esce da init prima che tutti si connettano (TCPStore, torch 2.2.1 --standalone); corsa solo fra rank 1-3; nessuna corsa nei reload (scettici R2-M1).
9. r.149-150: expected_job_id e' passato dai test (test_contrastive_release.py:111,114); manca solo nei chiamanti di produzione.
10. r.38: M-1 in 94f5094 e' "Rilievo minore", non una nota.
11. r.212: sonda di Leonardo solo per le API del profilo; inoltre il ramo is_autocast_enabled("cuda") e' gestito con except TypeError.

## Esito finale del verificatore 1
Circa 150 affermazioni controllate. Nessun errore su SHA, impronte, byte, tetti, conteggi di agenti, valori del blocco o P-3/P-4/P-6/P-8.
Errori e imprecisioni da correggere (dettaglio nell'uscita strutturata):
- ERRORE: "Rilievi minori (correzioni di codice: cambiano il digest)" / r.52 / prompt r.283: M-c (test) e M-e (CLAUDE.md) sono fuori da source_digest.
- IMPRECISIONE: M-e/G-2 (declassato a NOTA da entrambi gli scettici; non e' l'ambiguita' di d9020c20; blocco solo sul ramo codex; il titolo H1 limita alla linea 3).
- IMPRECISIONE: "9 rilievi confutati": R4-M1 voto 1-1.
- IMPRECISIONE: R3-N1, la correzione proposta non copre la review non favorevole con esempio.
- IMPRECISIONE: "Lacune del critico ... in P-1...P-8": 2, 5, 6, 13, 15 assenti; 8 e 10 solo come note; P-3/P-4 non vengono dal critico.
- IMPRECISIONE: API torch di Leonardo: sonda solo per il profilo; ramo TypeError.
- IMPRECISIONE: r.7 locale non UTF-8 (Latin-1 e C/POSIX non falliscono).
- IMPRECISIONE: P-1 sincronizzazione (rank 0 aspetta tutti; reload senza corsa).
- IMPRECISIONE: P-7 "seconda iterazione" (ricostruzione dalla terza, 547).
- IMPRECISIONE: expected_job_id passato dai test.
- IMPRECISIONE: M-1 e' un rilievo minore, non una nota.
- IMPRECISIONE: "corrispondono al codice e ai test" vs G-3/G-6.
