# Verifica scettica di OPS-1 (lente ops_costs) — 22/11/2026

Albero letto in sola lettura: /Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244 (commit e8c98244). Nessun ssh, nessuna rete.
Difetto dichiarato: «margine di quota WORK (368.816.128 byte oltre i 12 GiB) inferiore alle predizioni FP32 dei diagnostici (451.577.600 / 541.893.120):
un diagnostico che le salvasse su WORK violerebbe l'aritmetica del cancello di quota usato per A1, salvo lettura nuova».

## V1 — i numeri archiviati (DIMOSTRATO, riletti)

- `results/infra/pose_temponet_execution_20261121_A1/launcher/pending_writes_reassessment_20261121.json`:
  work_free_bytes 36876038144, pending_write_bytes 23622320128, unchanged_minimum_free_plus_pending_bytes 12884901888,
  free_after_pending_gib 12.343486785888672, observed_at 2026-11-21T15:08:55+01:00, quota_passes true,
  scaling_archive_reserved_bytes 19327352832 (18 GiB), other_metadata_operational_reserved_bytes_unchanged 4294967296 (4 GiB),
  source_policy «Current pinned plans only; recompute if another archive/storage plan is released.»
- Aritmetica: 36.876.038.144 - 23.622.320.128 = 13.253.718.016; - 12.884.901.888 = 368.816.128. Confermato.
- La lettura grezza coincide: `operator_before_controls_quota_20261121.txt` kbytes 1037730068 su quota 1073741824 -> (1073741824-1037730068)*1024 = 36.876.038.144.
  All'invio dei controlli (`controls_quota_at_submission.txt`): kbytes 1037734136 -> 36.871.872.512 liberi (4.165.632 byte in meno).

## V2 — che cosa confronta il cancello (DIMOSTRATO dal codice e dal protocollo)

- Codice: `scripts/pose_temponet/__main__.py:72-74`: `work_free_bytes >= 12*1024**3 + pending_write_bytes` («WORK quota gate fails»);
  `scripts/pose_temponet/resources.py:252-253`: `"quota": available >= 12*GIB+pending` e, SEPARATO, `"persistent_output": predicted_bytes <= output <= 8*GIB`.
- Protocollo congelato `docs/pose_temponet_frozen_protocol_20261117.md:202`: «quota WORK >= **12GiB+scritture pendenti delle altre linee** (8output+4margine)»;
  riga 198: «max 8 GiB persistenti (15 pred FP32 = 3,31664 GiB)»; riga 217: «massimo 8 GiB output».
- Quindi: le uscite PROPRIE della campagna non entrano in `pending_write_bytes` (che sono le scritture delle ALTRE linee: qui 18 GiB archivio scaling + 4 GiB metadati);
  stanno dentro i 12 GiB (8 di uscite + 4 di margine) e hanno un tetto proprio di 8 GiB.

## V3 — prova eseguita: il cancello di A1 con il codice congelato (DIMOSTRATO)

Script: `/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/ops1_work/gate_probe.py`; uscita `.../ops1_work/gate_probe.out.json`.
Comando (dalla cartella ops1_work): `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<tree>/scripts:<tree>/src ~/.venvs/wearusfm-tok/bin/python gate_probe.py`.
Chiama `resources.gate_and_pack` vero con le due ricevute di benchmark archiviate e i tetti `budget_amendment.effective_caps(plan_controls.json)`.

| scenario | free | pending | uscite proprie | quota | persistent_output | all_passed |
|---|---|---|---|---|---|---|
| A1 come rilasciato (15:08) | 36.876.038.144 | 23.622.320.128 | 4.667.458.721 | True | True | True |
| uscite proprie + 20 predizioni | idem | idem | 5.119.036.321 | True | True | True |
| uscite proprie + 24 predizioni | idem | idem | 5.209.351.841 | True | True | True |
| 20 predizioni contate come pendenti altrui | idem | 24.073.897.728 | 4.667.458.721 | False | True | False |
| 24 predizioni contate come pendenti altrui | idem | 24.164.213.248 | 4.667.458.721 | False | True | False |
| cancello riletto DOPO la scrittura (20) | 36.424.460.544 | 23.622.320.128 | 4.667.458.721 | False | True | False |
| cancello riletto DOPO la scrittura (24) | 36.334.145.024 | 23.622.320.128 | 4.667.458.721 | False | True | False |

- Il primo scenario coincide con l'archivio: `plan_controls.json["resource_gate"]["gates"]` = quota True, persistent_output True;
  `plan_controls.json["operator_checks"]` = work_free_bytes 36876038144, pending_write_bytes 23622320128, observed_at 2026-11-21T15:08:55+01:00, stage controls.
- Riduzione all'assurdo: le uscite proprie proiettate di A1 (`TCNH-gpu-benchmark-20261120.json` details.benchmark.persistent_output_bytes = 4.667.458.721,
  di cui prediction_payload_bytes 3.561.216.000) sono 12,655 volte il margine di 368.816.128 byte, e il cancello di A1 e' passato.
  Se le uscite proprie dovessero stare nel margine oltre i 12 GiB, A1 stesso non sarebbe mai passato. Quindi «l'aritmetica del cancello usato per A1»
  NON confronta le uscite proprie con il margine oltre il minimo.
- Numeri del difetto riconfermati: 1907*148*20*4 = 22.578.880 byte per stato; x20 = 451.577.600; x24 = 541.893.120 (forma (N,148,20) float32: `workflow.py:498`).
  Deficit solo nella lettura «pendenti altrui» o «dopo la scrittura»: 82.761.472 (20 stati) e 173.076.992 (24 stati).
- Tetto delle uscite proprie 8 GiB = 8.589.934.592: anche con A1 intero (che non verra' scritto) + 24 predizioni restano 3.380.582.751 byte.

## V4 — A1 e' fermo e ha scritto poco (DIMOSTRATO dalle ricevute)

- `positive_controls_gate_20261121.json`: gate.status = recipe_not_admitted (temporal_head_gate.status recipe_not_admitted): gli stadi successivi
  (selezioni reali, rifit, inferenza: i 3.561.216.000 byte di predizioni) non verranno scritti da A1.
- Artefatti dei sei controlli scritti DOPO la lettura delle 15:08 (somma delle dimensioni nelle ricevute: selection.json + selection_states.pt + combined_gate.json):
  3.081.188 + 18.668.270 + 18.554.008 + 2.450.232 + 14.879.277 + 18.639.609 = 76.272.584 byte (piu' 6 ricevute 44.053 byte e 6 log 867.002 byte).
  Anche A1 ha quindi «consumato» il margine oltre i 12 GiB dopo la lettura, senza che cio' sia mai stato una violazione: conferma che il cancello e' una
  condizione PRIMA delle scritture proprie, non un invariante da conservare dopo.

## V5 — necessita' del salvataggio (in parte DIMOSTRATO, in parte osservazione)

- La diagnosi di GPT (`docs/reviews/pose_temponet_temporal_diagnosis_20261122.md:38`) chiede di «rigiocare i checkpoint selezionati», «riportare fitting e held
  separatamente», «registrare i numeri»: non chiede di conservare le predizioni di 20 stati. Il difetto riguarda una scelta ipotetica di disegno del driver, non la diagnosi.
- Osservazione (file NON committati, letti alle 11:4x del 22/11, possono cambiare): la bozza del driver nel worktree `claude-tcn-diag`
  (`scripts/tcn_temporal_diag/__main__.py`, `core.py`) scrive solo `replay.json`, `parity.json`, `clean_fit.json`, una ricevuta e `clean_fit_states.pt`
  (stati, ~0,9 MB l'uno); nessun `np.save`/`savez` di predizioni. L'ipotesi del difetto oggi non si realizza.
- Alternative se servissero le predizioni: statistiche sufficienti per finestra (KB), solo i 2 stati scelti (45.157.760 byte), FP16 (225.788.800 / 270.946.560).

## V6 — che cosa NON e' dimostrato

- Lo stato della quota del 22/11: non letto da nessuno (nemmeno da me: niente ssh). La lettura e' vecchia di ~21 ore su una quota condivisa da 11 utenti,
  con un archivio da 18 GiB in corso: il margine odierno puo' essere maggiore o minore. Una lettura nuova e' comunque obbligatoria prima di ogni rilascio
  (protocollo riga 202; `docs/pose_temponet_execution_A1_20261121.md:31`; passo 0 della sequenza dello stesso revisore ops_costs).
- Resto vero del difetto: il margine archiviato e' sottile (368.816.128; ~291,6 MB dopo le scritture dei controlli di A1, a parita' del resto), quindi un
  PROSSIMO cancello dello stesso tipo (candidato residuale: `resource-plan` -> `_operator_checks`) potrebbe fallire se le pendenti restano 22 GiB;
  ma questo vale con o senza le predizioni dei diagnostici e si decide sulla lettura nuova.

## Verdetto

CONFUTATO come difetto «importante»: i tre numeri sono esatti, la conclusione («violerebbe l'aritmetica del cancello usato per A1») no.
Il cancello chiede `free >= 12 GiB + pendenti delle ALTRE linee`; i 12 GiB sono «8 output + 4 margine» e le uscite proprie hanno il tetto separato di 8 GiB.
451.577.600 / 541.893.120 byte di uscite proprie sono il 5,26% / 6,31% degli 8 GiB. Gravita' residua: NOTA (lettura nuova della quota prima dell'invio,
gia' prevista; niente predizioni massive su WORK perche' non servono).

## V7 — composizione delle «pendenti» e rischio fisico (DIMOSTRATO sui numeri archiviati)

- pending_write_bytes 23.622.320.128 = scaling_archive_reserved_bytes 19.327.352.832 (18 GiB, linea 1) + other_metadata_operational_reserved_bytes_unchanged 4.294.967.296 (4 GiB):
  nessuna uscita propria di A1 dentro le pendenti (`controls_admission_record.json`: minimum_quota_margin_GiB 12, pending_writes_GiB 22).
- Dentro le pendenti ci sono 6.063.634.500 byte di sola prudenza (archive_padding_bytes 1.768.667.204 + riserva metadati 4.294.967.296).
- Sui numeri del 21/11: dopo il massimo formale dell'archivio (17.558.685.628) e 24 predizioni (541.893.120) resterebbero 18.775.459.396 byte liberi;
  contando l'intera riserva di 22 GiB, 12.711.824.896 byte (>= 4 GiB di margine del protocollo). Nessun rischio di EDQUOT attribuibile a questa scrittura, a parita' del resto.
- Rilascio di A1 delle 12:07: 39.557.316.608 >= 12.884.901.888 + 25.769.803.776 (margine 902.610.944); alle 15:08 le pendenti sono state ristimate da 24 a 22 GiB
  (`status: measured_operator_estimate_no_gate_or_source_change`): il «margine» e' il residuo di una stima prudente, non una misura di spazio realmente scarso.
