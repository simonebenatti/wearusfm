# Factcheck n. 2 della bozza claude_review_contrastive_targets_runtime_20261120.md

Stato: COMPLETO. Circa 165 affermazioni controllate con comandi propri (git show/rev-parse sulla copia a 0efa4d45,
ricalcolo di source_digest dagli oggetti git, lettura di workflow_result.json, rapporti e sonda torch).

## Controllati e corretti (sintesi)
- 54 rilievi; sopravvissuti 1 BLOCCANTE (G-1), 18 MINORE, 26 NOTA; 9 non sopravvissuti
  (R1-M2, R1-N1, R1-N3, R1-N4, R3-M2, R4-M1, R4-N2, R4-N6, R6-4). Nessun id confutato citato nella bozza.
- 85 agenti = 7 revisori + 77 voti di scettici (21 MINORE x2, 32 NOTA x1, G-1 x3) + 1 critico; 85 file in reports/.
- fbfde797 = 46 file; 0efa4d45 aggiunge solo delivery_pin.json.
- source_digest ricalcolato: 40d39f56...cde0 (125 file) su entrambi i commit; plan b03c0bc6...021a 14862 B;
  preflight e4b39591...6dd92 9082 B; preflight.source_sha256 e science_plan.sha256 coincidono.
- Schema contrastive-target-release-review-v1; sei chiavi esatte; status solo APPROVED (release.py:23, 95-97).
- Citazioni del prompt GPT (r.39, 41-43), di Simone (decisioni 8278-8279, 8301-8302, 7846) e di faa4a8d7 (r.3, r.52,
  voce decisioni "Proposta per Simone"): letterali a meno di apostrofi ASCII.
- Risorse 18/169,6; 21,6/202,4; 6,086667/57,827778; 24,086667/227,427778 "training escluso"; 14/129,333333; 7,9 h;
  17 allocazioni; override 4 CPU/2G/4 min; gate sacct 7200 s/8 CPU/120 min; billing = CPU (prontuario r.94).
- P-1 (TORCH_DIST_INIT_BARRIER, Update 04/2023), P-3 (MinJobAge 30 s, 17/11), P-4 (172 coreh), P-6 (6,4 h, 5 rilasci,
  2 job dbg, 23/11 08:00-25/11 08:00), P-7 (545-546), P-8 (+7%) coerenti con i rilievi e con il critico.
- M-b 24 tracce/4 start.json; M-c 25/39 e 30/40; ricevuta 465 in 102,84 s contro 102,59 s.
- torch.load con mmap=True: tutti con str(...). Copie byte per byte: 94f5094 (blob c25d4f6f), faa4a8d7 (blob 5bdb0d15).
- Cartelle delle prove esistono. Bozza ASCII pura; nessun recinto contrastive-release-review.

## Errori e imprecisioni (finale)

1. ERRORE - Esito "Gli altri chiedono modifiche al codice, che cambiano il digest dei sorgenti", titolo
   "Rilievi minori (correzioni di codice: cambiano il digest)", prompt GPT "Correzioni di codice ... cambia il digest".
   tests/ e CLAUDE.md sono fuori dall'inventario di source_digest (contrastive_prepare.py:192-201). M-c (test) e M-e
   (CLAUDE.md) non cambiano il digest (G-3: "I test sono fuori dal digest, quindi la correzione non richiede una nuova
   review"; scettici G-2: CLAUDE.md fuori da source_digest). Molte note sono solo documentali (R4-N1, R4-N4, R4-N5, G-4,
   G-5, G-7, R1-N6).
2. IMPRECISIONE - "alle note M-1 e N-1...N-7": M-1 in 94f5094 e' un rilievo minore.
3. IMPRECISIONE - M-e/G-2: i due scettici danno NOTA e respingono "stesso tipo di ambiguita' di d9020c20" e "letto da
   tutti gli agenti"; il blocco ha titolo H1 "Linea3, incarico del19/11/2026"; il conteggio 18 minori include G-2.
4. IMPRECISIONE - "le API usate esistono nel torch 2.2.0a0": la sonda copre solo il profilo numerico (gap 14 del
   critico: fork_rng(devices=[]), patch AdamW.__init__, _get_ddp_logging_data, torch.load mmap non sondati).
5. IMPRECISIONE - "Lacune del critico. Sono riportate in P-1...P-8": assenti i gap 1, 4, 5, 7, 12, 14.
6. IMPRECISIONE - R6-8 (NOTA sopravvissuta) non citata: 25 note citate contro 26 dichiarate.
7. IMPRECISIONE - "le affermazioni della risposta di GPT corrispondono al codice": eccezione G-6 (R-4 "ID e TSV sacct").
8. IMPRECISIONE - "Nessuno porta a un esito attribuito male": R2-N1 (esito per coppia attribuito male) e R3-N1
   (review non favorevole presa per approvazione, come la bozza stessa scrive nelle Note).
9. IMPRECISIONE (minore) - R3-M3 "nessun chiamante": il test lo passa; "nessun chiamante di produzione".
10. IMPRECISIONE (minore) - "con un locale non UTF-8": con ISO8859-1 rc 0; fallisce con locale ASCII non C/POSIX.
