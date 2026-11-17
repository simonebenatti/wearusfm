# Controllo indipendente della validazione, 17/11/2026

Critico budget: CONFORME, nessun blocco. Accounting ricalcolato dal sacct originale: checkout 64655487 30 s × 2 CPU = 0,016667 core-ore; validate 64659520 42 s × 2 CPU = 0,023333 core-ore, MaxRSS 2,46 GiB. Entrambi COMPLETED 0:0. Piano 70483ba8 e codice df2de35 corretti; quattro cache posa, 40 utenti train e 15 val per taglia, target/supporto identici e utenti disgiunti. Piani misti ricostruiti identici alla consegna P1-r2. Inviluppo prudenziale 348,632778/351, con 246 ancora riservate ai fit.

Limite: validate riguarda le quattro cache posa, non le cache classificazione P1/P2; controlla hash, provenienza, geometrie, target e supporto. Non dà metriche scientifiche né verifica tutta la finitezza dei latenti. La finitezza viene verificata sulle patch lette da benchmark/fit/predict.
