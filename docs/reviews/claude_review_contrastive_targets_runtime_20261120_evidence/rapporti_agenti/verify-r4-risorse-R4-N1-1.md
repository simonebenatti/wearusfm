# Verifica scettica n. 1 — R4-N1 (README dei piani con il vecchio preventivo)

Esito: **NON CONFUTATO**. Gravita' reale: **NOTA** (come dichiarata).

## Riproduzione (comandi miei, copia a 0efa4d45, nessuna modifica)
- `sed -n 7p results/infra/contrastive_targets_20261118/README.md` -> "`preflight_plan.json`: 13 allocazioni proposte, massimo 14 GPUh / 137,066667 coreh; nessuna release."
- `git log --oneline -- .../README.md` -> solo `438c8d65`; `git diff 438c8d65 fbfde797 -- .../README.md` vuoto; `fbfde797..0efa4d45` aggiunge solo `delivery_pin.json`.
- `preflight_plan.json` a fbfde797 (protocollo `...-20261119-v2`): 11 celle + 6 checkout = 17 allocazioni; `resources.gpu_hours_cap` 18.0 (riga 246), `core_hours_cap` 169.6 (riga 247); somma celle 18 GPUh / 168 coreh + checkout 1,6.
- A 438c8d65 (protocollo v1): 9 celle + 4 checkout = 13; cap 14 / 137,066667 = esattamente le cifre del README.
- Lo stesso file nuovo chiama 14/137,066667 `old_unexecuted_target_proposal` (riga 270).

## Motivi di confutazione cercati e scartati
- Nessun codice o test legge il README (grep in scripts/, src/, tests/cpu): nessun effetto sulla catena -> conferma solo che e' NOTA.
- Il README non e' pinnato (delivery_pin, validation_final/local_validation.json; `historical_artifacts_changed_from_base` copre solo `contrastive_comparison_20261116`): nulla impediva di aggiornarlo o di intestarlo.
- Mitiganti, non confutanti: titolo "preparazione del 18/11"; prontuario 19/11 riga 9 ("prevalgono sui prontuari operativi del 16–18/11") e righe 99-101 (vecchio preventivo 14/137,066667 "sostituito"); il documento a cui il README rimanda (`docs/contrastive_targets_amendment_20261118.md:1`) ha l'intestazione 14->18 GPUh; delivery_pin e prontuario (righe 77, 88) concordano su 17 / 18 / 169,6.
- Pero' il README descrive per nome un file della stessa cartella sovrascritto in place, senza intestazione storica ne' rimando al 19/11; non cita `amendment_20261119/` e la riga 8 indica la ricevuta del 18/11 anziche' `amendment_20261119/validation_final/`. Incoerenza documentale reale.
- Precisazione: N-2 della review 94f5094 elencava solo tre prontuari (preflight_execution_20261116, handoff_20261117, prompt_3_launcher_..._20261118), tutti intestati da GPT; il README non era nel perimetro di N-2. Il rilievo e' un'osservazione nuova, non un N-2 non attuato.

## Gravita'
Solo documentazione; i job usano piani/prontuario/pin coerenti. Non BLOCCANTE ne' MINORE. NOTA.
