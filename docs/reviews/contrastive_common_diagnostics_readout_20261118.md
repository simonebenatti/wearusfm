# Come leggere le tre celle diagnostiche da stato comune

Scheda prospettica del 18 novembre per il codice `365cb1e` e il piano `35ea94b`. Risultati CUDA e riprese da inserire dopo la consegna di 3-launcher. [Protocollo e regole di interpretazione](../contrastive_common_diagnostics_20261118.md)

## CUDA a stato, batch e RNG comuni

Registrare stato della cella, profili completati, SHA-256 del rapporto e flag effettivi dei quattro rank. Il batch globale ha otto finestre. Alle due del rank 0 sono state tolte le etichette di posa. Prima dei confronti verificare le impronte per rank, il supporto delle due loss e l’accordo fra rank sul gradiente ridotto. Annotare separatamente un consumo diverso del RNG durante il backward.

| Confronto | Dato da riportare | Domanda scientifica |
|---|---|---|
| Entro ciascun profilo | Tre distanze di replica C0a/C0b, CNa/CNb e CRa/CRb. Riportare errore relativo del vettore pre-clipping e coseno, globali e per i blocchi interessati. | Quanto variano due backward dello stesso braccio? |
| C0/CN e C0/CR a pesi contrastivi zero | Contrasti nelle ripetizioni a e b, accanto alle tre distanze entro braccio dello stesso profilo. | La scelta del braccio produce uno scarto ripetibile rispetto alla variabilità osservata? |
| C0a fra `fp32_math` e `bf16_math` | Distanza con SDPA math e TF32 disattivato in entrambi. | Quanto cambia il gradiente con la precisione a backend fissato? |
| C0a fra `bf16_auto` e `bf16_math` | Distanza e flag per rank. | Si misura insieme l’effetto della selezione del backend SDPA e delle impostazioni TF32, incluso TF32 di cuDNN. |

Se le repliche differiscono anche in `bf16_math_deterministic`, indicare quali coppie differiscono, l’ampiezza e i blocchi coinvolti. Con ingressi e consumo RNG identici, la sorgente resta aperta. Se il profilo deterministico si arresta su un operatore senza implementazione deterministica, documentare l’operatore e leggere le coppie verificate dei profili completati. Le due repliche per braccio forniscono un riferimento descrittivo su questo batch.

## Ripresa da checkpoint comune

Compilare separatamente per CN e CR. Riferire SHA-256 del rapporto e del checkpoint comune a 580, stato al confine e primo passo/rank con differenza di batch, RNG o metrica tra 581 e 588. Per ciascuna coppia riportare anche momenti dell’ottimizzatore, identità e ordine della coda, e distanza delle chiavi.

| Serie da leggere a 588 | Coppie da affiancare per ciascun braccio | Denominatore della distanza relativa |
|---|---|---|
| Studente | continuo/reload_a, continuo/reload_b, reload_a/reload_b | Spostamento dello **studente** del primo membro fra 580 e 588 |
| Teacher EMA | continuo/reload_a, continuo/reload_b, reload_a/reload_b | Spostamento del **teacher** del primo membro fra 580 e 588 |

Il confronto fra i due reload misura la ripetibilità di due ripartenze. Se i reload concordano e si discostano entrambi dal continuo, il riavvio merita indagine a partire dalla prima differenza. Se le tre distanze hanno ampiezza simile, l’attribuzione resta aperta. Per il teacher leggere le tre coppie sulla sua scala, indipendentemente dalla serie dello studente. Documentare gli eventuali limiti dovuti a tracce parziali e chiudere con ciò che i rapporti consentono di decidere sui training.
