# TCN A1: controlli conclusi, gate temporale non superato

Stato operativo: **stopped_at_positive_controls / recipe_not_admitted**. Sei controlli COMPLETED 0:0, nessuna selezione reale, rifit o inferenza held inviata. Non e un problema di autorizzazione: e l'arresto prescritto dal protocollo davanti a un gate negativo.

## Provenienza e verifica

Codice rivisto a1ae9607681c1a181034d7f72770537054af419d; checkout immutabile edebfb6c290cf661989c5cff45916bc6a93e607e; 247 sorgenti 9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea, invariati. Review originale Claude 2f759ebbf275e2aae3d63b986b48e035590c82a1. Piano controls b334ece870bc85b804708f92e4c240f4999750ef8d089ca7df1bd040d2ee04e3 al commit 6a20054ad2ffdb3d754e6106775a159a0933d99a. Tutte le ricevute dei controlli hanno il contesto gpu8 12cb2a0b3207b66e260d3ca490db93ee0c029bb8f03d766df89a9edba6b7eb3f.

Ricevute e sei rapporti di selezione copiati per byte e hash originali sul Mac, insieme ai sei log e al consuntivo. Il controller congelato ha verificato ricorsivamente gli undici predecessori e i loro artefatti, letto entrambi i calibration.pt originali, e applicato runner._control_gate/workflow.qualify_controls. Esecuzione operatore in sola lettura, 13,76 s, nessuna nuova allocazione, fit, cache di segnale o dato di sviluppo esterno letto, nessuna scrittura remota. I pannelli chiamati held nei controlli sono i fold esterni interni al train.

Prova completa: results/infra/pose_temponet_execution_20261121_A1/launcher/positive_controls_gate_20261121.json. Gate SHA256 interno: 9e91c8a237cc305f6e8c47bacff170536e4b5f0c1c1599f542e078c22e133858. Sul Mac: ricevute ricorsive, sei lock e statistiche di selezione verificati nuovamente; formule del recupero temporale ricalcolate anche direttamente. Esito in controls_final_verification_20261121.json. Questa verifica non sostituisce i componenti delle calibrazioni con le loro sintesi JSON.

## Esito dei controlli

| Controllo | SLURM | Tempo | Esito |
|---|---:|---:|---|
| lineare L0 | 65035622 | 24m11s | passa |
| lineare MLP | 65035623 | 21m27s | passa |
| lineare TCN_post | 65035626 | 61m50s | passa |
| temporale L0 | 65035628 | 18m25s | descrittivo |
| temporale MLP | 65035629 | 14m20s | descrittivo |
| temporale TCN_post | 65035631 | 73m47s | non passa |

I riferimenti WLS erano diagnostic in entrambi i mondi. Nel controllo lineare L0, MLP e TCN recuperano almeno il 97,46% del vantaggio della ridge rispetto alla costante, in tutti i pannelli E_dyn e MSE: soglia 90% rispettata. L0 segnala minimum_in_last_five_maximum_epochs, ma supera ugualmente il criterio congelato; non e un'autorizzazione a estendere la dose.

Nel controllo temporale la soglia richiede di recuperare almeno il 50% del vantaggio dell'oracolo rispetto al riferimento lineare clean_ls in ciascun fold e nel pooled:

| Pannello | clean_ls E_dyn | Oracolo E_dyn | TCN E_dyn | Recupero |
|---|---:|---:|---:|---:|
| fold0 | 0,710431 | 0,653451 | 0,696657 | 24,17% |
| fold1 | 0,749920 | 0,691103 | 0,738818 | 18,88% |
| pooled | 0,728479 | 0,670659 | 0,715926 | 21,71% |

Recupero = (clean_ls - TCN) / (clean_ls - oracolo). I tre pannelli sono sotto il 50%. Nessun optimization_alarm sulla TCN temporale: il risultato prescritto e recipe_not_admitted, non optimization_unresolved. Selezionati LR 0,003 ed esposizione 31 dalla griglia e dalla regola originali, dopo il completamento dell'intera ricerca. Il numero di esposizioni del candidato scelto non e il numero totale di epoche calcolate dalla ricerca.

L0 e MLP temporali ottengono pooled 0,735580 e 0,735130; sono descrittivi, non possono qualificare la lettura temporale. La TCN mostra un recupero positivo, ma insufficiente al criterio firmato: non e corretto dire che non impara nulla.

## Conclusione scientifica e operativa

La ricetta di ottimizzazione supera il controllo lineare ma non dimostra la capacita temporale richiesta sul controllo costruito. Non possiamo quindi usare un eventuale risultato nullo sul compito reale per concludere che la dinamica manca nelle rappresentazioni del FM. Questa campagna non ha ancora misurato il confronto reale fra teste, ne la loro generalizzazione sui 15 utenti esclusi dal pretraining.

Fase successiva ferma senza invii, retry, soglie o sorgenti cambiati. Prima di una ripartenza serve una diagnosi del controllo/ricetta e, se si propone un cambiamento, una consegna distinta e revisionata. Nessun nuovo esperimento preparato o lanciato da questa nota. Nessun monitor in background.

## Consuntivo e incidenti operatore

Sei controlli: 3,566667 GPUh / 28,533333 coreh. Totale delle tre campagne, inclusi i due inventari CPU originali, il vecchio preflight fallito, il checkout A1 e il censimento GPU: **16 allocazioni, 3,681111 GPUh / 31,975000 coreh**, di cui 2,526111 coreh CPU-only. Tetti A1 invariati: 53,85 GPUh neurali, 56,10 GPUh totali, 115,50 coreh CPU-only, 564,30 coreh complessive; massimo 29 allocazioni. Fonte sacct -X, senza conteggio doppio di batch/extern; picchi RSS dei controlli 2,91-3,08 GiB. Ledger: un solo sottomesso per ciascuno dei 16 ID; nessuna prenotazione operativa sospesa e nessun ID della fase reale inviato.

Due tentativi operatore di sola lettura del gate sul login hanno fallito prima del calcolo per numpy non importabile: l'operatore aveva sostituito PYTHONPATH anziche conservare i percorsi del modulo cineca-ai. Risolto riproducendo l'append del launcher congelato. Log dei due errori conservati; zero job aggiunti, nessuna modifica al venv o al checkout, nessun retry di allocazioni. Comando e script operatore finali versionati nelle prove, fuori dal manifest dei sorgenti scientifici.

Ricevute, rapporti, log e consuntivo sono nel repository; calibration.pt e selection_states.pt restano su WORK ai percorsi pinnati. Nessun rinfresco SCRATCH eseguito. I piani congelati e i riferimenti deploy esistenti non sono modificati dalla chiusura al gate.
