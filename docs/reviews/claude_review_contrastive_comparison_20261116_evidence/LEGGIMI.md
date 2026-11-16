# Prove della review Claude del pacchetto contrastivo `db5ac41` (16/11/2026)

Accompagna `docs/reviews/claude_review_contrastive_comparison_20261116.md`. Nessun dato da wearable: solo script, uscite numeriche,
test su tensori sintetici e conteggi dagli indici del flusso.

## Come preparare gli alberi

```bash
git archive db5ac41 | tar -x -C <albero_db5ac41>
git archive 924e19d | tar -x -C <albero_924e19d>
```

Ambiente usato: Mac, CPU, `~/.venvs/wearusfm-tok` (torch 2.14.0), sempre con
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`.

## `mie/`: controlli eseguiti da Claude

Le uscite hanno estensione `.out.txt`, perché il repository ignora i file `.out`.

| File | Che cosa fa | Ingressi fuori dal repository |
| --- | --- | --- |
| `tests_local.log`, `tests_distributed.log` | I 153 test locali di GPT e i sei distribuiti (5 passati, [CR-4] fallito) | nessuno |
| `diag_ddp.py`, `diag_ddp.out.txt` | Diagnosi di [CR-4]: lo scarto è nella sola metrica `pose` | nessuno |
| `nce_coverage_real.py`, `nce_coverage_real_4s.out.txt`, `_1s.out.txt` | Copertura InfoNCE sull'intero flusso, per dataset e per blocco | log del controllo 30M, manifest v1.1 |
| `panel_and_clip.py`, `panel_and_clip.out.txt` | Finestre per dataset nel pannello di calibrazione, nell'audit e nel primo blocco; update senza clipping; aritmetica float32 del peso delle ancore | gli stessi |
| `ratio_from_log.py`, `ratio_from_log.out.txt` | Norma del gradiente lungo il run e rapporto ρ che ne segue | log del controllo 30M |
| `rule_real_widths.py`, `rule_real_widths.out.txt` | Regola di lettura sui quattro bracci storici col bootstrap congelato | nessuno: usa i rapporti nel repository |
| `time_memory.py`, `time_memory.out.txt` | Costo CPU della memoria alle taglie vere | log del controllo 30M, manifest v1.1 |
| `suite_fail_924.txt`, `suite_fail_db5.txt` | Test falliti della suite completa sui due alberi (12 e 23) | nessuno |
| `w_trainer_stop.out.txt` | Arresto del cancello di copertura nel trainer vero (I-4): estratto dell'uscita di `script_agenti/w_trainer_stop.py` | nessuno |
| `test_proposti_run.log` | Corsa finale dei 203 test proposti: 200 passati, 1 saltato, 2 `xfail` | nessuno |
| `first_grad_norms.out.txt` | Primi otto valori della norma del gradiente nel run storico | log del controllo 30M |
| `leonardo_lettura.txt` | Limite dei file aperti e propagazione dei limiti, letti sul nodo di login | nessuno |

Gli ingressi fuori dal repository sono `~/wearusfm_local/logs/review6_remote/control30m/metrics.jsonl` (solo indici: campo `origin`,
`pose_windows`, `grad_norm`) e `~/wearusfm_local/manifests_derived_20261015/derived_final/manifest_v1_1_nested_1.json.gz`.

## `test_proposti/`: 13 file, 203 test, da lanciare su un albero a `db5ac41`

Sono un suggerimento da adattare, non una modifica della ricetta. Il nome `prop_*.py` li tiene fuori dalla raccolta automatica.

```bash
cd <questa cartella>/test_proposti
PYTHONPATH=<albero_db5ac41>/src:<albero_db5ac41>/tests/cpu:<albero_db5ac41>/scripts:$PWD \
  python -m pytest -q -p no:cacheprovider --basetemp=<cartella nuova> prop_*.py
```

Esito sul codice consegnato, eseguiti da Claude: 200 passati, 1 saltato, 2 marcati come difetto atteso (`xfail`, rilievo I-9a).
Il test saltato è `test_constants_match_protocol_and_plan` di `prop_compare_extra.py`: cerca `plan.json` due cartelle sopra il
file. Copiato in `tests/cpu` di un albero a `db5ac41` passa (eseguito da Claude).

| File | Test | Origine | Che cosa proteggono |
| --- | ---: | --- | --- |
| `prop_rnc_extra.py` | 9 | analisi RnC | Distanze oltre 256 ancore, ancore non contate, guardie |
| `prop_rnc_attach_oracle.py` (con `rnc_oracle_common.py`) | 4 | analisi RnC | Valore di `contrast_rnc` nel punto di chiamata contro un oracolo |
| `prop_rnc_strong_v.py` | 58 | verifica RnC | Oracolo numpy con ancore non contate, pareggi oltre il blocco da 256 |
| `prop_rnc_callsite_v.py` (con `v_common.py`) | 13 | verifica RnC | Campi dell'identità, cancello, log, 400 ancore |
| `prop_nce_killers.py` | 23 | analisi InfoNCE | FIFO, trial, proiettore, soglia di norma, stati corrotti |
| `prop_nce_killers_v.py` | 7 | verifica InfoNCE | Margine prima e dopo, sessione e soggetto nell'identità, scadenza per età |
| `prop_compare_extra.py` | 53 | analisi comparatore | Pesi, soglie, quantili, sospensioni del verdetto, costanti di `plan.json` |
| `prop_compare_tradeoff_xfail.py` | 2 | analisi comparatore | Pareggio nei due versi senza vincitore: oggi fallisce (I-9a) |
| `prop_compare_vc.py` | 2 | verifica comparatore | Scala relativa delle repliche di E_dyn |
| `prop_distributed_independent.py` | 10 | analisi distribuito | Lunghezze diverse, processi senza posa, calibrazione e ripresa sotto DDP |
| `prop_calibration_reader_extra.py` | 14 | analisi calibrazione | Mediana, limiti inclusivi, soglie del cancello nel lettore |
| `prop_calibration_adversarial.py` | 5 | analisi calibrazione | Pesi fermi, optimizer mai usato, config a pesi nulli |
| `prop_calibration_4rank_ckpt.py` | 3 | analisi calibrazione | Norme misurate = norme applicate a 4 processi con activation checkpointing |

## `patch/`: varianti provate dagli agenti, non applicate

- `optR/`: correzione di I-1, come diff in formato normale fra la consegna (`<`) e la variante provata (`>`), cinque file.
- `correzione_minima.diff`: correzione del test [CR-4] (I-2), una riga modificata e cinque aggiunte.
- `F1_nan_coordinated_stop/`: arresto coordinato al posto dell'eccezione (I-3), 10 righe in `attach`.

## `script_agenti/`

- `PYTHONPATH=<albero_db5ac41>/src python w_trainer_stop.py <albero_db5ac41>`, da una cartella di lavoro vuota (vi scrive
  `w_trainer_stop.json`): fa scattare il ramo d'arresto del cancello nel trainer vero. Base per il test di I-4.

## `rapporti_agenti/`: i venti rapporti

Per ciascuna delle dieci dimensioni un rapporto d'analisi e uno di verifica avversaria: rilievi con file e riga, caso, prova,
comando per rifare e correzione proposta. Sono il testo degli agenti: vale la review per ciò che ne ho tratto. Dove analisi e
verifica divergono vale la verifica; la sezione 9 della review elenca le tre correzioni che cambiano il senso. I comandi citano `<archivio>`: la cartella di lavoro completa degli agenti
(script, uscite, mutanti) è conservata sul Mac, fuori dal repository, in
`~/wearusfm_local/reviews/contrastive_review_20261116/agents/`. Negli script il prefisso della cartella temporanea originale va
sostituito con quello dell'archivio.
