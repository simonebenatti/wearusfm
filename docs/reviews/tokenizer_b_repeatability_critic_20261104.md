# Controllo storico T25-L − B19618: review indipendente del 04/11/2026

**Approvato come controllo appaiato della riproducibilità della ricetta nominale. Non è un pavimento statistico di rumore.** Nessun nuovo fit o job.

La [verifica strutturata](../../results/infra/tokenizer_evaluation_20261103/t25_b_repeatability_20261104/comparability_review.json) conserva percorsi e SHA256 delle fonti, diff dei config, impronte dei file del codice storico, controlli delle sonde e ricalcolo indipendente.

## Comparabilità

B usa il codice `24139ab901c052357a46a65bc20466f2eb1e8358`; T25-L `afb644ea10ecc2a716160b4d8a4904e805c366fb`. Le sole differenze nei config sono i default esplicitati (`identity`, raggio0), il log delle origini e la conservazione del checkpoint5000. Frontend, calibrazione sintetica scalare, architettura, loader, seme, batch, schedule, dose e obiettivi hanno la stessa definizione. Entrambi riprendono il proprio training a5000 con stato completo fino a19618.

La sonda B è la faseB convergente a12000iterazioni:140/140fit, nessun warning di convergenza. Identici hash degli ingressi e split, ricetta di estrazione e liste train/val/held dei20fold. Coincidono pool768FP32, assenza diPCA, scaler train-only, solver, grigliaC, selezione/refit e versioni numeriche. I C selezionati differiscono in12/20fold: risposta della selezione alla rappresentazione, non griglia diversa.

La comparabilità è funzionale, non una certificazione di replay numerico. B non registrava tutte le origini dei batch; manca un confronto diretto degli hash dei tensori/gain iniziali storici. Il preflight recente appaiava soltanto i tre nuovi bracci. Le versioni principali dell'ambiente coincidono, ma T25 non ha il checksum pip completo registrato daB. Non si attribuisce lo scostamento residuo a una causa specifica.

## Ricalcolo indipendente

Senza importare lo script di analisi né moduli del progetto: SHA e dimensioni delle quattro OOF verificati; pairing esatto per riga, etichetta, utente e fold; contatori soggetto/classe ricostruiti dalle OOF con bincount; bootstrap appaiato mediante indicizzazione diretta degli utenti (2000repliche, seme20261102), distinto dalla moltiplicazione per la matrice delle frequenze dello script. Ricostruiti punti, intervalli e contatori pubblicati per tutti i cinque dataset e P1/P2. Impronte delle estrazioni bootstrap identiche al rapporto tokenizer congelato; errore numerico massimo inferiore a1e-10punti percentuali.

| Contrasto T25-L − B19618 | Differenza | IC95% appaiato |
|---|---:|---:|
| EPN612 | −0,23567pp | [−0,76438; +0,27389] |
| P1 | +0,51037pp | [+0,08984; +0,92589] |
| P2 | −0,09332pp | [−1,38244; +1,06475] |

P1 mostra uno scostamento positivo fra queste due esecuzioni nominalmente uguali. Gli intervalli EPN/P2 attraversano zero: differenza non risolta, non equivalenza né assenza di variabilità. Sono risultati descrittivi post hoc sullo sviluppo, a checkpoint/sonde fissi, senza correzione di molteplicità.

Una singola coppia allo stesso seme non misura la distribuzione fra semi. Non usare questo delta o IC come soglia di adozione, limite del rumore o correzione dei contrasti architetturali. Il limite era già esplicito nell'[emendamento del03/11](risposta_claude_tokenizer_20261103.md).
