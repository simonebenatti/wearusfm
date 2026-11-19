# Diagnostici contrastivi da stato comune · lettura scientifica del 19 novembre

**Decisione proposta a Simone.** Chiudere la diagnosi sulla ricetta storica 25/50. Il calcolo introduce differenze misurabili tra esecuzioni a ingressi identici. Nei casi studiati la scelta della loss a peso zero e il caricamento del checkpoint non mostrano un eccesso di scarto distinguibile dalla variabilità osservata ripetendo lo stesso braccio. Restano aperte la riproducibilità al batch di produzione e la convenienza dei nuovi obiettivi sul downstream. Propongo di adattare i due controlli nella coorte RMS100/fast125 prima dei training.

Evidenze al commit `8e38eb71`, esecuzione `365cb1e`, piano `35ea94b`, criteri di lettura `db66b507`. Ho verificato i 121 SHA-256 del manifest locale, i tre rapporti, i confronti per blocco e le 192 righe delle tracce di input/RNG (due bracci × tre percorsi × quattro rank × otto passi). Le copie provengono da 3-launcher. Quattro job completati, 0,802222 GPU-ore e 6,456667 core-ore aggiuntive. [Dati e contabilità](../../results/infra/contrastive_comparison_20261116/common_results_20261119/) · [Scheda fissata prima dei dati](contrastive_common_diagnostics_readout_20261118.md)

## Il gradiente a pesi contrastivi zero

Stato iniziale, coda, batch, microbatch e RNG coincidono per rank. Anche il consumo finale del RNG e il gradiente ridotto tra i quattro rank concordano. Otto finestre emg2pose, due sul rank 0 senza etichette di posa. Entrambi gli obiettivi hanno supporto. Il test simula l’update 255 senza aggiornare i pesi.

| Scarto relativo del vettore pre-clipping, bf16 automatico | Ripetizione a o singola coppia | Ripetizione b |
|---|---:|---:|
| C0a/C0b, repliche controllo | 0,029617% | — |
| CNa/CNb, repliche InfoNCE | 0,028261% | — |
| CRa/CRb, repliche Rank-N-Contrast | 0,028834% | — |
| Controllo/InfoNCE | 0,029281% | 0,029434% |
| Controllo/Rank-N-Contrast | 0,029482% | 0,028859% |

Tutti e quattro i confronti tra bracci rientrano nell’intervallo delle tre distanze entro braccio. I coseni globali sono almeno 0,99999995. Il tokenizer ha il maggiore scarto relativo per blocco (0,052–0,057% sulle sette coppie). Nei profili bf16 math, FP32 math e bf16 math deterministico i sei backward hanno un solo hash del gradiente per profilo e scarto zero in ciascun blocco. Due ripetizioni per braccio danno un confronto descrittivo su questo batch.

A backend math fissato, FP32/bf16 dà **0,569780%** di differenza del vettore e coseno **0,99998402**. È una differenza ripetibile fra precisioni. A precisione bf16 fissata, automatico/math dà **0,068807%**. I quattro rank passano da SDPA con più backend ammessi e TF32 di cuDNN attivo a solo math e TF32 di cuDNN spento. TF32 per matmul è spento in entrambi. Il kernel effettivamente scelto resta ignoto. Math e math deterministico danno lo stesso gradiente su questo caso.

**Collegamento con il vecchio fallimento.** Fra le sette coppie bf16 automatiche il massimo scarto della *norma scalare registrata dal trainer* è 0,000557%. La regola storica sulla norma era circa 0,01% e veniva applicata a ogni passo di traiettorie già evolute. Il singolo backward isola una sorgente di variabilità. Non quantifica quanto di quella perturbazione si amplifichi fino agli scarti storici dello 0,725% per CN e 0,315% per CR. [Diagnostica storica delle norme](contrastive_gradient_parity_diagnostic_20261118.md)

## Ripresa con le loss attive

Per ogni braccio il continuo salva a 580 e prosegue in memoria. Due gruppi nuovi ripartono da quei medesimi byte. Prima del primo forward coincidono studente, teacher, stato contrastivo, momenti e gruppi di Adam al learning rate previsto. Per ogni rank coincidono batch effettivi, microbatch e RNG prima del forward per tutti gli otto aggiornamenti. CN ha 31–32 query ammissibili per passo, CR ha 1.190–43.056 coppie. La prima differenza nelle metriche, anche nella loss totale tra i due reload, compare già a 581 in entrambi i bracci.

| Braccio e coppia | Studente | Teacher EMA | Adam primo momento | Adam secondo momento | Chiavi della coda |
|---|---:|---:|---:|---:|---:|
| CN continuo/reload A | 2,4691% | 0,10389% | 0,2684% | 0,002019% | 0,10692% |
| CN continuo/reload B | 2,4521% | 0,10388% | 0,2761% | 0,002092% | 0,10603% |
| CN reload A/reload B | 2,5161% | 0,10723% | 0,2594% | 0,001953% | 0,10574% |
| CR continuo/reload A | 3,1234% | 0,14185% | 0,2056% | 0,001948% | 0,07759% |
| CR continuo/reload B | 2,9990% | 0,13848% | 0,2116% | 0,002109% | 0,07670% |
| CR reload A/reload B | 3,3980% | 0,14986% | 0,2153% | 0,001968% | 0,07711% |

Le distanze sono al passo 588. Per studente e teacher il denominatore è il movimento 580→588 **del rispettivo modello** nel primo membro della coppia. Le tre distanze del teacher vanno lette all’interno della sua colonna e dello stesso braccio. Per i momenti e le chiavi il denominatore è la norma aggregata dei tensori del primo membro al passo 588. Contatori e gruppi dell’ottimizzatore, metadati, stream, RNG finale e identità e ordine della coda coincidono in tutte le coppie.

Il continuo e le riprese differiscono nell’avvio delle riduzioni distribuite. A 581 il continuo riporta i bucket DDP già ricostruiti. Nei reload la ricostruzione compare nelle tracce a 583, con dimensioni e ordine da quel punto uguali al continuo. Le due riprese hanno le stesse tracce dei bucket in ogni rank ma divergono già a 581. In entrambi i bracci la coppia reload/reload presenta la maggiore distanza finale dello studente e, nella sua scala, del teacher. Un difetto che perda lo stato controllato al caricamento è contraddetto dalle impronte. Resta possibile un effetto numerico più piccolo del riavvio o comune ai due reload. Un solo checkpoint e due reload per braccio su otto passi non danno un limite di equivalenza.

Il gate storico a 576 misurava i parametri dello studente e del teacher rispetto al movimento dall’inizializzazione, controllava separatamente i buffer e applicava l’1% **a ciascun tensore** mobile di ottimizzatore e coda. Le percentuali della tabella rispondono a domande diverse. Il rapporto storico resta `FAILED_REQUIRES_REVIEW`. Tutti e tre i nuovi rapporti mantengono `training_approved=false`. Le prove sono tecniche e non ordinano CN e CR per beneficio sulla rappresentazione. [Rapporto del gate storico](contrastive_readout_results_20261118.md)

## Proposta per Simone e per la review della nuova coorte

Proposta prospettica per RMS100/fast125, informata dai dati 25/50. La review della preparazione richiede una scelta esplicita su parità e ripresa prima della release (R-1). Suggerisco **bf16 math con TF32 spento**, fissato e registrato su tutti i rank, per i tre bracci dalla calibrazione ai training se è sostenibile. Il profilo ordinario math ha già ripetuto il singolo backward esattamente. Tempo e memoria al batch misto globale di 32 e costo dei benchmark restano da misurare. Una difficoltà di memoria, tempo o ripetibilità riporta il piano a Simone prima di qualunque cambio di backend. [Preparazione RMS100/fast125](https://github.com/simonebenatti/wearusfm/blob/438c8d65/docs/contrastive_targets_amendment_20261118.md) · [Review di Claude, R-1–R-4](https://github.com/simonebenatti/wearusfm/blob/94f50942/docs/reviews/claude_review_contrastive_targets_amendment_20261119.md)

1. **Parità a peso zero.** Integrare nel collaudo su quattro GPU due stati reali del controllo, subito prima degli aggiornamenti 128 e 256, con batch misti di 32 e coda corrispondente. Fissare questi indici prima di aprire i dati della coorte. Da ciascuno stato calcolare due backward per C0, CN e CR ripristinando pesi, coda, batch, maschere e RNG per rank. Richiedere nel profilo scelto identità dei gradienti ridotti per byte entro e tra i bracci, oltre agli invarianti e al supporto di entrambe le loss. Un confronto senza supporto o privo del profilo richiesto resta inconclusivo. Per le 256 norme e loss di traiettorie indipendenti si conserva la lettura longitudinale con le vecchie tolleranze (relativa 1e−4, assoluta 1e−6). Questa lettura descrive come si accumula la deriva con la dose. Gli invarianti del flusso e del peso zero restano obbligatori. **Cambia la prova decisiva** per la neutralità a peso zero.
2. **Ripresa.** Nei benchmark attivi di CN e CR, salvare una sola volta al passo 544 per braccio, proseguire in memoria fino a 576 e fare due ripartenze indipendenti dagli stessi byte per i passi 545–576. Verificare lo stato esatto al confine, i batch e le impronte RNG CPU e CUDA separate per rank e passo, il supporto attivo, i contatori e le identità ordinate. Applicare a tutte e tre le coppie gli stessi limiti numerici storici con la stessa definizione (1% sui parametri rispetto all’intero spostamento dall’inizializzazione del primo membro, buffer esatti, 1% per singolo tensore mobile di Adam e coda). Affiancare distanze dal punto di biforcazione e scarti per blocco con il relativo denominatore. **Cambia il punto di partenza del confronto**, così la differenza preesistente fra traiettorie non entra nella misura. Una violazione mantiene il blocco. Un eccesso ripetibile continuo/reload, con reload concordi, richiede localizzazione prima di promuovere il profilo.
3. **Costo e uscita.** Usare l’audit, la calibrazione train-only con nuove λ e i tre benchmark già necessari per 100/125. Misurare rapporti dei gradienti a pesi attivi, memoria e throughput nella configurazione candidata. Prima degli invii della rispettiva fase, fissare SHA di sorgenti, piani, review e input, tetti comprensivi delle ripartenze e delle verifiche e criteri di arresto. L’audit nuovo deve completare con exit code zero entro 7.200 s. Il tetto proposto in precedenza di 14 GPU-ore per la preparazione va ricalcolato per la matrice modificata. Con ricevute complete e favorevoli, presentare il costo dei tre training a 19.618 (lettura descrittiva a 5.000) e una release distinta. Un rapporto mancante lascia inconclusi tutti i confronti che lo richiedono.

Lo stato attuale è **proposta da discutere prima degli invii**. La qualifica dei target 100/125 richiederà evidenze della nuova coorte e un preventivo dedicato. La decisione scientifica su InfoNCE e Rank-N-Contrast dipenderà dai contrasti appaiati su E_dyn, P1 ed EPN612. 3-launcher conserva la responsabilità operativa dopo l’eventuale release.
