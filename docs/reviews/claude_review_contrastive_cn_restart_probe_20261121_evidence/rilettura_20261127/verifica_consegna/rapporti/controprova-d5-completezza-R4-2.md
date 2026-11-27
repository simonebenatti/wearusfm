# Controprova R4: impronte dei checkpoint a 576

Esito: regge, gravita' importante.

## Verifiche fatte
- Sequenza, righe 187-188: dice che le impronte dei .pt sono "gia' nelle ricevute e nel report".
- observe.py righe 122-126: la ricevuta observation.rank*.json registra file, bytes e sha256 dei soli vettori al 545.
- compare.py riga 316: endpoint_hashes[mode] = file_sha(actual) del checkpoint.pt a 576. Le righe 326 e 386 lo scrivono
  come actual_endpoint_sha256, ma solo dentro il report.
- grep su leg.py, contrastive_run.py, contrastive_job.py: nessun altro file registra lo sha256 di checkpoint.pt a 576.
- contrastive_cn_restart.py righe 175-184: execution_failure.json contiene fase, fasi completate, elapsed e causa. Nessuna impronta.

## Tentativi di confutazione, falliti
- L'ordinaria ha uno SHA atteso fisso (compare.py riga 22, ENDPOINT). Copre pero' solo la gamba ordinaria, non la riscaldata,
  ed e' un valore atteso, non misurato.
- I .pt restano su WORK, quindi lo SHA si potrebbe calcolare piu' tardi. Ma un calcolo fatto al momento del recupero non prova
  che i byte non siano cambiati dopo il job.
- Il caso conta davvero. Il protocollo (riga 25) prevede il recupero del solo lettore proprio quando il confronto si
  interrompe per un errore operativo, cioe' quando report.json manca. In quel caso chiede di conservare le "impronte degli input"
  e i "relativi SHA verificati". Seguendo la sequenza l'operatore crederebbe di averle gia' e non le calcolerebbe.
- Non brucia un ID e non cambia l'esito del job. Per questo la gravita' e' importante e non bloccante.

## Correzione
Si accoglie quella proposta: se manca report.json, calcolare e archiviare sha256, dimensione e percorso di ogni .pt della
cartella di uscita (checkpoint delle due gambe e vettori). In alternativa, correggere la frase sui .pt.
