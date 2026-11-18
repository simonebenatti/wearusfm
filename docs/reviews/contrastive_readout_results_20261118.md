# Lettura finale del collaudo contrastivo — 18/11/2026

## Esito operativo

- checkout congelato: job **64775502**, COMPLETED 0:0, 61 s, 4 CPU;
- lettura CPU: job **64775571**, FAILED 2:0, 124 s, 8 CPU, 30 GiB;
- l'uscita 2 è il comportamento previsto quando il rapporto completo nega il
  rilascio, non un crash del lettore;
- stato: `FAILED_REQUIRES_REVIEW`; `training_approved=false`;
- rapporto: `preflight_readout.json`, 1.094.229 byte, SHA256
  `deb6b8178362e23512f105d95b3859c28945d9ffa5356a653f1c8cad67c929fc`;
- log: `contrastive_cpu_64775571.out`, SHA256
  `ec58c42a9e66dd6234a759060392cfe97659fe9f769ee06d48c03892b82e9a35`.

Non è stato eseguito alcun retry e non sono state modificate soglie, lambda o
ricette. Il codice congelato ha digest
`46938f319181d408ee852cfaacfdc92cc5e44eaa1cfa770825ad96956eec9320`.

## Risultati

| Controllo | CN, InfoNCE | CR, Rank-N-Contrast | Esito |
|---|---:|---:|---|
| rapporto mediano gradiente attivo | 0,2261 | 0,0834 | entrambi nei limiti |
| p95 rapporto gradiente attivo | 0,4311 | 0,2140 | entrambi nei limiti |
| rapporto mediane CN/CR | \- | 2,7106 | nel limite [0,25; 4] |
| parità loss, primi 256 update | 256/256 | 256/256 | passa |
| parità `grad_norm`, primi 256 update | 110/256 | 121/256 | fallisce |
| primo scarto `grad_norm` | update 83 | update 79 | peso contrastivo ancora zero |
| massimo scarto relativo `grad_norm` | 0,7246% | 0,3145% | oltre la tolleranza congelata |
| student continuo/ripreso | 3,248% | 6,012% | limite 1%, fallisce |
| teacher continuo/ripreso | 2,047% | 2,337% | limite 1%, fallisce |
| tensori mobili oltre limite | 1.688 | 1.375 | optimizer e coda |

Il lettore registra 3.071 motivi elementari perché elenca anche ogni tensore
oltre limite. Il numero non rappresenta 3.071 difetti indipendenti: i gate
negativi sono quattro, parità iniziale CN/CR e ripresa CN/CR.

Le stime di costo prodotte dal rapporto sono 24,15 GPU-ore per C0, 24,86 per
CN e 28,07 per CR; il tetto complessivo congelato per i training sarebbe stato
118,67 GPU-ore. Questi training non sono autorizzati dall'esito.

## Interpretazione

La calibrazione delle due nuove loss è riuscita: quando i termini diventano
attivi, la loro scala rispetto alla base è quella prevista. Il rapporto non
permette però di procedere, perché il comportamento non soddisfa i criteri di
riproducibilità fissati prima dei dati.

Il fallimento della parità a peso zero è reale nella misura di `grad_norm`, ma
non compare nella loss e una replica locale CPU FP32 è bit-identica. Non è stato
dimostrato se la causa sia il percorso CUDA/bf16, l'ordine delle riduzioni o una
differenza negli input effettivi.

Il gate di ripresa ha inoltre un confondimento metodologico: il run continuo e
quello poi ripreso sono due esecuzioni indipendenti fin dall'inizio. Le loro
norme differiscono già prima del punto 544. Gli stati finali sono dunque diversi,
ma questo confronto non attribuisce causalmente la differenza al salvataggio e
caricamento del checkpoint.

## Decisione e seguito proposto

1. Non rilasciare ora C0/CN/CR a 5.000 o 19.618 update.
2. Costruire un diagnostico CUDA piccolo da stato, batch, maschere e RNG comuni,
   ripristinando l'intero stato prima di C0/CN/CR a peso zero.
3. Isolare la ripresa partendo da un unico checkpoint al punto di separazione:
   prosecuzione in memoria contro caricamento in un processo nuovo, con una
   replica dello stesso seguito per stimare la variabilità ordinaria.
4. Conservare i criteri attuali. Un loro eventuale emendamento deve essere
   prospettico e motivato dai diagnostici, non adattato a questo risultato.

La diagnosi estesa e gli script riproducibili sono in
`contrastive_gradient_parity_diagnostic_20261118.md` e
`results/infra/contrastive_comparison_20261116/gradient_parity_diagnostic_20261118/`.
