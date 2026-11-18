# Collaudo contrastivo CUDA bf16 superato — 18/11/2026

Checkout **64746634 COMPLETED 0:0**, 53 s. CUDA **64746651 COMPLETED 0:0**, 141 s.
Codice congelato `8905e346b03e5d173cccc124aa40032bf588a191`; source
`46938f319181d408ee852cfaacfdc92cc5e44eaa1cfa770825ad96956eec9320`.

Rapporto, probe, quattro attestazioni dei worker, stdout/stderr, log del job,
accounting e hash remoti archiviati byte per byte in
`results/infra/contrastive_comparison_20261116/cuda_completed_bf16_20261118/`.
`readout.json` conserva i valori esatti e i nove hash. Rieseguiti i lettori congelati
`_cuda_evidence` e `validate_receipt`; provenienza, inizializzazione, input e pin coerenti.
Review indipendente concorde, conservata in `independent_review.txt`.

| Controllo | Risultato | Lettura |
| --- | --- | --- |
| Confronti numerici, quattro rank | 320/320 nei limiti | Superato |
| FP32, loss e norme/gradiente | massimo 6,394e-7, limite 1e-5 | Superato |
| FP32, spostamenti studente/teacher | massimo 0,0007423, limite 0,01 | Superato |
| bf16, tre norme diagnostiche | massimo 0,0018955, limite almeno 0,01 | Superato |
| STOP copertura/non finiti | positivo su tutti i quattro rank | Superato |
| Uscita intenzionale di un rank | 12,32 s; zero processi residui | Superato |

Il massimo rapporto errore/limite è 0,6124, sul teacher bf16: errore 0,0348451,
limite 0,0568986. Questo limite deriva dal rumore del riferimento ed era già congelato;
non va descritto come errore inferiore all'1%. Rimane il limite metodologico dichiarato
nella review bf16. Nessun criterio è cambiato dopo questo CUDA. I precedenti tentativi
64706211 e 64730941 restano falliti e inclusi nel consuntivo.

Questa è una validazione tecnica, non un risultato sull'efficacia di InfoNCE o RnC.
È soddisfatto il gate per inviare le cinque celle già autorizzate: tre benchmark C0/CN/CR
a 576 update e due prove di ripresa CN/CR a 544+32. Fonte e ricette restano congelate;
comandi esatti in `releases/bf16_20261118/submission_plan.json`. Il training scientifico
lungo non è rilasciato. Nessun monitor in background.

Costo di questo CUDA: 0,156667 GPU-ore / 1,253333 core-ore; checkout 0,058889 core-ore.
Totale consumato della campagna: **1,232222 GPU-ore / 18,610000 core-ore**.
Cinque celle residue: tetto 10 GPU-ore / 80 core-ore; mantenendo anche checkout finale
0,266667 core-ore e lettura CPU 8 core-ore, massimo complessivo
**11,232222 GPU-ore / 106,876667 core-ore**, sotto 14 / 129,333333.
QoS riletta: MaxJobsPU 2 e MaxSubmitPU 4. Invii a gruppi contando tutte le linee.
