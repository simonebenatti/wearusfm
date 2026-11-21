# TCN A1 — aumento dei soli tetti autorizzato da Simone, 21/11/2026

Stato: budget autorizzato; emendamento del runtime da preparare e revisionare. Nessun fit rilasciato.

## Decisione

Dopo la comunicazione del gate negativo, Simone ha scritto: «va bene se aumenti il budget» e «dagli quello che serve moltiplicato per 1.5».
La base e il preventivo inclusivo misurato dai due benchmark reali, con fattore temporale 2 e 180 secondi per job gia compresi. Il moltiplicatore 1,5 riguarda i tetti, non un secondo aumento dei walltime, della dose o del numero di modelli.

| Tetto | Fabbisogno misurato | Nuova autorizzazione, ×1,5 |
| --- | ---: | ---: |
| Fit neurali | 35,90 GPUh | 53,85 GPUh |
| Campagna GPU totale | 37,40 GPUh | 56,10 GPUh |
| Sola CPU | 77,00 coreh | 115,50 coreh |
| Campagna totale | 376,20 coreh | 564,30 coreh |

Restano inclusi nei tetti anche i costi operativi precedenti; il consuntivo e molto sotto il margine. Non aumentare i limiti di memoria, i walltime per job, i massimi di LR/epoche o il numero di celle: passano gia i gate. Nessun punteggio di nuovi lettori e stato consultato.

## Prove pubblicate dall'operatore

Checkout immutabile in uso: `/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose_temponet_20261120`, SHA `3205763d1b3e29799231aa1f9453819f6182e4ee`. Sorgenti `37985d28c75bf82a6a784106d414825b727b9ba78575b675dd2dac1d4ccfc8de`, codice rivisto `5d6fb2dc855bd3c12788bb6d3c3a66043c34f1ba`.
Piano rilasciato `results/infra/pose_temponet_execution_20261120/plan_released.json`, SHA256 `84b82384a2eafb60f11480910d33ebc85832cb75063d3479d42e243351ea23b0`.

Copie originali, log, sacct e pin in `results/infra/pose_temponet_execution_20261120/launcher/`:

- TCNH-preflight-20261120 = 65016127, COMPLETED 0:0, ricevuta SHA256 `0a3314138a98ea3b9e6a8475297c38d5fed575dc112eb52dc4ae4d14d51ee16b`.
- TCNH-cpu-benchmark-20261120 = 65016438, COMPLETED 0:0 in 244 s, ricevuta SHA256 `d6fca52ab2d719003d641a24f1067e25c7c31f9400d928c1d0e0d3757fc33ce8`.
- TCNH-gpu-benchmark-20261120 = 65016447, COMPLETED 0:0 in 392 s, ricevuta SHA256 `8228db8c6bed7df2eb45eeeedb2203f002ff4e64f625adca010c53f1b367f306`.

`benchmark_verification.json`: payload, provenienza, input, contesti numerici e artefatti verificati. `resource_gate_failed_20261121.json`: proiezione integrale, falliscono soltanto neural_gpu_hours, gpu_total, core_total. `budget_amendment_A1_authorization_20261121.json`: decisione e cifre esatte. `operator_checks_resources_20261121.json`: lettura nuova dopo i benchmark.

Consuntivo cumulato (due inventari seriali, primo preflight fallito, nuovo preflight e due benchmark): 6 allocazioni, 0,108889 GPUh e 1,522778 coreh. Nessun job WLS, controllo neurale, selezione, rifit o inferenza inviato. Nessuna prenotazione aperta senza esito nel ledger. Primo fallimento e tre ricevute valide restano intatti.

## Consegna al responsabile del codice scientifico

Preparare un emendamento esclusivamente di risorse che riconosca i quattro tetti sopra in entrambi i controlli (`resources.gate_and_pack` e `execution.apply_training_pack`) e nel piano. Il prontuario del 19/11, sezione Identita, impone: «Ogni modifica ai sorgenti invalida il piano e richiede review». La responsabilita di codice/review resta distinta dall'operatore unico: questo commit non modifica il runtime congelato.

Prima del rilascio, risolvere esplicitamente la transizione di source identity: i tre predecessori validi sono prodotti dal codice congelato precedente. Non basta cambiare due numeri: `validate_plan`, `verified_receipt`, `_ancestors`, `resource-plan` e il runtime rifiuterebbero il nuovo source_sha256 o execution_sha256. Conservare le tre ricevute con pin esatti e una regola ristretta e verificabile di compatibilita per questi predecessori; non accettare genericamente hash diversi e non reinterpretare la provenienza storica. Fissare i sorgenti permessi dal diff, identita scientifiche identiche, nuovo checkout/riferimento e come si collegano gli ID gia completati ai restanti. Nessun retry dei tre job riusciti per risolvere soltanto una difficolta di provenienza.

Il pacchetto da rileggere deve comprendere codice, test (rifiuto di ricevute non pinnate o alterate, ricetta/input cambiati, vecchi tetti non scavalcabili senza A1), piano rigenerato, stessa proiezione sulle stesse misure e prontuario incrementale. Nessuna nuova autorizzazione finanziaria necessaria entro i tetti qui registrati. Contatto con Claude tramite Simone come da prontuario; nessun messaggio inviato ad altre chat.

Prima degli invii GPU dipendenti resta il confronto hardware su un secondo nodo: il benchmark su lrdn0059 misura Xeon Platinum 8358, NVIDIA A100-SXM-64GB, capability [8,0], 68.099.571.712 byte. Il pin numerico gpu8 e `12cb2a0b3207b66e260d3ca490db93ee0c029bb8f03d766df89a9edba6b7eb3f`; cpu8 `21e70b1c1a297b71f430fc4036bf9afdf61a5068a7e59c02c05b8012dc1e24d7`. Non allentare i pin. Nessun refresh SCRATCH durante uno stadio.

## Seguito: implementazione completata

Implementato nel codice a1ae9607681c1a181034d7f72770537054af419d. Per la review e il seguito operativo vale docs/pose_temponet_execution_A1_20261121.md; il presente documento conserva la richiesta originaria. Stato awaiting_claude_rereview, nessun deploy o job A1.
