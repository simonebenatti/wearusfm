# Lettore temporale: esecuzione e preventivo dopo timing — 02/11/2026

Simone autorizza il lancio diretto di questa catena, con submitter e ledger comuni.
Codice scientifico invariato: `440af702ac3ec00feb52549acb1b4935251dce38`.
Piano con SHA256 `2c1b6f431297e603c46d01f60276038fac67bc879cd8aba00954b9aa120a3b3c`.

## Controlli completati prima del run

| Fase | Job | Stato | Tempo | CPU allocate | Costo core-ore |
|---|---:|---|---:|---:|---:|
| Checkout congelato | 63219091 | COMPLETED 0:0 | 19 s | 2 | 0,01056 |
| Hash/header/metadata quattro cache | 63227865 | COMPLETED 0:0 | 46 s | 2 | 0,02556 |
| Timing solo train B | 63228323 | COMPLETED 0:0 | 24 s | 8 | 0,05333 |
| Totale preparazione | | | | | **0,08944** |

Checkout aveva richiesto 1 CPU ma SLURM ne alloca 2: il consuntivo usa l'allocazione effettiva.
Ogni fase è stata sottomessa solo dopo aver verificato il precedente COMPLETED 0:0;
non sono state aggiunte dipendenze afterok su job già conclusi. Nessuna deroga ai controlli.
Validazione superata su quattro cache, SHA originali, codice e metadati attesi.

Il benchmark ha aperto solo il train B. Statistiche + tre eigensistemi:
istantaneo 3,072 s; temporale 10,726 s; totale F=13,798 s.
Supporto train identico: 4.944.020 elementi bersaglio validi per lettore.
MaxRSS benchmark 1,394 GiB. Nessuna metrica scientifica val calcolata.

## Risorse del confronto completo, fissate prima del lancio

Richiesta **8 CPU, 16 GiB, 30 minuti: massimo 4 core-ore, zero GPU**.
Stima prudenziale: `2 × (2F + 46 + 740)` = 1.627 s, arrotondata a 30 minuti.
2F copre il fit dei due encoder; 46 s il controllo di hash/metadata; 740 s sono una
riserva per CV, decompressione, inferenza e metriche, pari a quattro volte i 185 s
del precedente rapporto E3. Quest'ultima è una riserva, non un timing misurato della nuova val.
Il fattore finale 2 aggiunge margine; gli eigensistemi non si moltiplicano per gli alpha.

Memoria: `1,5 × max(1,394 + 4, 8,33)` = 12,495 GiB, arrotondata a 16 GiB.
Il benchmark non apre la val: 4 GiB aggiuntivi e il riferimento E3 di 8,33 GiB evitano
di dimensionare il run sulla sola memoria del train.
Tetto catena con preparazione già misurata: **4,08944 core-ore**, entro i 20,5 autorizzati.
Nessuna modifica a supporto, lettori, griglia, selezione, bootstrap o interpretazione.
Dettagli macchina: `results/infra/pose_temporal_20261102/resource_decision.json`.

## Rettifiche Claude e disponibilità degli OOF

Letto `d93e22e`: stessa griglia C, 26,77 core-ore come riferimento storico del rifit pool768,
non preventivo verificato per un nuovo controllo. Sono rettifiche già incorporate nel piano GPT.
I quattro OOF su $WORK sono stati recuperati e verificati; il bootstrap appaiato è concluso
in `pca_existing_results_20261102.md`. Le vecchie note sulla loro assenza locale sono storiche.
Il confronto misura differenze fra pipeline, non l'effetto causale isolato della PCA.
Review indipendente del critico: nessun nuovo fit richiesto da queste rettifiche;
il protocollo temporale prosegue invariato. Nessun merge del ramo Claude implicito in questa lettura.

Il primo tentativo del collector locale incontrava Python 3.6 sul login node, che non supporta
`subprocess.run(text=..., capture_output=...)`. Corretta soltanto la raccolta in sola lettura
con `universal_newlines` e pipe esplicite: nessun job o codice scientifico modificato/rilanciato.
