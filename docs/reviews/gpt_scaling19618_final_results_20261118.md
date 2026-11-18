# Scaling 30M/300M a 19.618 passi — risultati finali, 18/11/2026

Il confronto 64785568 è COMPLETED 0:0 in 199 s. Sei aggregati accettati, 420 fit convergenti e controlli di integrità, supporto appaiato e convergenza tutti superati. Il confronto congelato ha verificato gli array posa, ricalcolato le statistiche e gli intervalli e controllato le provenienze prima della scrittura del report. Una verifica locale aggiuntiva ha riprodotto i punti della classificazione dai conteggi interi, i delta, la lettura congelata e l'identità del manifest.

## Primari

| Endpoint | 30M | 300M | Delta 300M − 30M, IC95 | Lettura fissata |
|---|---:|---:|---|---|
| EPN612, balanced accuracy % | 82,58 | 83,07 | +0,497 pp [−0,038; +1,038] | Irrisolto |
| Posa dinamica E_dyn, inferiore migliore | 0,838933 | 0,838694 | −0,000239 [−0,002328; +0,001793] | Irrisolto |

Riduzione relativa puntuale dell'errore dinamico 0,0285%, contro il 2% richiesto dalla definizione di miglioramento pratico. A questa dose manca un vantaggio dimostrato sui primari. Questo risultato riguarda la ricetta e i checkpoint esaminati, con un solo seme di pretraining e utenti di sviluppo.

## Classificazione

| Dataset o task | 30M % | 300M % | Delta pp, IC95 |
|---|---:|---:|---|
| NinaPro DB2 | 18,22 | 18,83 | +0,609 [−0,337; +1,533] |
| NinaPro DB3 | 3,38 | 3,01 | −0,365 [−1,366; +0,598] |
| NinaPro DB6 | 33,75 | 32,75 | −1,002 [−2,070; −0,085] |
| EPN612 | 82,58 | 83,07 | +0,497 [−0,038; +1,038] |
| UCI EMG | 78,46 | 79,83 | +1,376 [−1,807; +4,704] |
| P1, aggregato NinaPro | 17,19 | 17,32 | +0,136 [−0,525; +0,748] |
| P2, aggregato EPN/UCI | 80,52 | 81,45 | +0,936 [−0,682; +2,584] |

P1 e P2 sostengono la non inferiorità con il margine prespecificato di 1 pp. Il risultato su DB6 indica un calo nella descrittiva del singolo dataset. Intervalli marginali, senza correzione per confronti multipli.

## Dimensione pareggiata e interpretazione

Il 300M proiettato da 896 a 384 dimensioni dà EPN612 81,89%, delta contro 30M −0,686 pp [−1,227; −0,144]. La vista naturale del 300M supera quella proiettata di 1,183 pp [0,862; 1,501]. La proiezione fissa può rimuovere informazione utile, quindi questo controllo non isola un effetto causale puro del numero di parametri.

Per E_dyn la vista pareggiata dà 0,838274, delta contro 30M −0,000659 [−0,003349; +0,001799], ancora irrisolto. Nelle descrittive la posa statica migliora, con riduzione relativa 0,70% nella vista naturale e 1,93% nella vista pareggiata. Il movimento resta il punto debole dello scaling osservato con questa lettura.

Il guadagno puntuale sui gesti è contenuto e variabile tra dataset. Per la posa dinamica i due modelli sono molto vicini, con incertezza che attraversa zero. La lettura non identifica un guadagno pratico del 300M a questa dose. Eventuali decisioni su altro training spettano a Simone alla luce del confronto scientifico tra le linee.

## Artefatti e risorse

Report completo in `results/infra/scaling_300m_20261104/continuation19618_20261106/evidence/compare/report.json`. Ricevute, sacct grezzo, accounting e log sono affiancati per ciascuno stadio. Il manifest comparison_inputs è fissato nel metadata b7f01477. I tre checkout documentali sono stati conservati, con codice numerico df2de35.

Consuntivo CPU-only cumulativo del controller 94,732778 core-ore su 351. Comprende la riserva storica prudente 4,792778. Tutti gli stadi residui ora sono contabilizzati come completati. Costi GPU del pretraining e delle estrazioni restano nei rispettivi consuntivi precedenti.

Il controllo locale aggiuntivo non ha rieseguito indipendentemente l'intero bootstrap sugli array. La verifica completa è stata svolta dal comparatore congelato sul cluster. Riproduttore e verbale locale in submissions/p1_reader_release_20261117/check_final_report.py e final_report_check.json.
