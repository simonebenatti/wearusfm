# r4-risorse — Punto 4: risorse e piani

Inizio: 2026-11-19 14:03:03
Copia: rev_targets/clone, HEAD 0efa4d45; diff 438c8d65..fbfde797

## a) Rigenerazione e impronte (14:10)
- `contrastive_targets_plan.py --check` alla punta 0efa4d45: exit 0, source 40d39f56..., 17 allocazioni.
- Clone temporaneo a fbfde797, generazione senza --check: git status vuoto; cmp identico per plan.json e preflight_plan.json.
- Clone con plan.json di 438c8d65 rimesso al suo posto e generatore di fbfde797: output identico byte per byte a fbfde797 (tutto il diff del piano viene dal generatore).
- sha256 plan b03c0bc6... (14862 B), preflight e4b39591... (9082 B), validation_final 36d1d4d4... (9629 B): identici a fbfde797 e 0efa4d45; delivery_pin coincide. fbfde797..0efa4d45 aggiunge solo delivery_pin.json.
- Nota: `--check` confronta il piano con science_plan(piano stesso): prova l'idempotenza dei campi sovrascritti, non i campi ereditati.

## b/c) Ricalcolo (14:20)
- Celle: audit 8x7200=16 coreh; 9 GPU x (4x1800=2 GPUh, 32x1800=16 coreh) = 18 GPUh/144; lettura 8x3600=8; checkout 6x4x240=1,6 -> 18 GPUh / 169,6 coreh; walltime 27000+1440 s = 7,9 h.
- Storico: catena 4756 s GPU-nodo (774+67+127+141+3647) = 5,284444 GPUh; core 51,371111 ricostruito per incrementi (15,458889 -> 16,128889 -> 17,297778 -> 18,61 -> 51,027778 -> 51,371111). Diagnostici 4x722/3600=0,802222; (4x35+32x722)/3600=6,456667. Totali 6,086667 / 57,827778. Residuo 7,913333 / 71,505556; incremento 4 / 32,533333; cumulato 24,086667 / 227,427778; margine 3,6 / 32,8; inviluppo 21,6 / 202,4. Tutto coerente fra piano, prontuario, decisioni, risposta, delivery_pin.
- RILIEVO (governance): a 438c8d65 required_before_first_submission conteneva 'publish and present resource proposal to Simone'; a fbfde797 e' stato tolto, mentre il preventivo sale a 18 GPUh (oltre il tetto registrato 14 e il residuo 7,91).

## Fonti dei consuntivi (14:35)
- Ricostruito 51,371111 coreh da righe sacct grezze: GPU 4756 s x32 = 152192; audit 64652559 3787x8 = 30296; lettura 64775571 124x8 = 992; checkout 64652044/64652383/64701182/64705989/64730738/64746634/64775502 = 30+11+106+67+36+53+61 = 364 s x4 = 1456 -> 184936 core-s = 51,371111. GPU (774+67+127+141+661+683+769+775+759)x4 = 19024 = 5,284444.
- historical_diagnostics_accounting.txt: 64835236 35 s/4 CPU, 64835286 154, 64835297 289, 64835304 279 (32 CPU, 4 GPU, billing = CPU). Il generatore hardcoda 4*(154+289+279) e 4*35: coerenti col file (non letto dal codice).
- Ricevuta validation_final: --verify ok (465 passati). Non contiene il controllo `contrastive_targets_plan.py --check`; nessun test confronta i byte su disco col generatore (test_contrastive_targets_plan usa solo il piano in memoria). Affermazione "piani rigenerati identici" vera (mia verifica), ma non attestata dalla ricevuta.

## e) Dispatcher (14:40)
- Script r4_scripts/check_alloc.py: 11 celle coerenti con ALLOCATED_SECONDS, launcher (#SBATCH) e validate_allocation; audit richiede override (default CPU 3600).
- Script r4_scripts/check_dispatch.py contro il preflight REALE: 55 casi (s, s+-1, 1800/3600/7200) accettati dalla guardia solo ai secondi del piano; tipo cpu/gpu scambiato rifiutato; con ALLOCATED_SECONDS mutato a 1801 la guardia di cella rifiuta; arm/repeat/mode/ID diversi rifiutati.
- 60 test (targets_plan, job, release, targets_guards) passati.

## Altre verifiche (14:55)
- Digest dei sorgenti ricalcolato dagli oggetti git (r4_scripts/digest_from_git.py): fbfde797 e 0efa4d45 = 40d39f56...; 438c8d65 = 643a8937... (coerente con la review 94f5094).
- --check rileva: secondi di una cella alterati, sorgente alterato, campo aggiunto a plan.json (via hash nel preflight). Ripristinato.
- historical_diagnostics_accounting.txt identico (sha b48fe647...) all'accounting archiviato in 8e38eb71 (common_results_20261119/accounting.txt).
- Memoria storica: audit MaxRSS 22.555.604 KiB (~21,5 GiB) su 30 GiB; calibrazione host MaxRSS 46.889.120 KiB su 240 GiB; picco GPU 13,72 GiB (summary bf16). Training storico 638-742 s su 576 update contro 1770 interni (x2,4); calibrazione 774 s (x2,3). Profilo math non misurato.

## Rilievi
1. MINORE (governance budget): tolto 'publish and present resource proposal to Simone' da required_before_first_submission (438c8d65 -> fbfde797 preflight_plan.json:315-319; generatore :114-115) mentre la richiesta sale a 18/169,6 (inviluppo 21,6/202,4) oltre tetto registrato 14/129,333 (residuo 7,913333/71,505556) e oltre la soglia di autonomia di 5 GPUh come prova unica. Nessun cancello di budget nel codice (release/dispatcher leggono solo celle, secondi, fresh_root). Prontuario 181-187 e blocco CLAUDE.md di fbfde797 condizionano la calibrazione al solo sacct dell'audit.
2. NOTA: README della cartella dei piani (results/infra/contrastive_targets_20261118/README.md:7) dice ancora 13 allocazioni, 14 GPUh/137,066667.
3. NOTA: depends delle celle senza i checkout; un solo checkout 'reloads' -> ogni reload attende entrambi i full CN e CR.
4. NOTA: '--dopo <ID_PREREQUISITO>' per i checkout successivi (prontuario:175-177) -> 'Job dependency problem' dopo MinJobAge 30 s (incidente 17/11, decisioni.md:7852-7854).
5. NOTA: "1770 s interni" non vale per la cella CUDA (nessun limite interno).
6. NOTA: limiti QoS dbg 2/4 per utente e memoria seriale 30.800 MiB per utente non riportati nel piano.
7. NOTA: CUDA e calibrazione accettata verificati solo dentro l'allocazione GPU (make_job verifica solo l'audit, docstring dice altro).
8. NOTA: README/ricevuta: 'piani rigenerati identici' non attestato dalla ricevuta validation_final (solo pytest).

Fine: 14:23:24. Cloni temporanei cancellati; restano gli script in rev_runtime/tmp/r4_scripts. Copia di sola lettura pulita.
