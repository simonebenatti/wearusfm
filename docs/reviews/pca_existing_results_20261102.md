# Pipeline pool128 e pool768: confronto appaiato — 02/11/2026

**La pipeline PCA128 perde rispetto alla pipeline storica pool768 in entrambi gli encoder; in P2 la penalizzazione è maggiore per B.** Il divario B−R3c di PSD-3 va quindi letto tenendo conto anche della pipeline di lettura. Il confronto non isola causalmente la sola PCA, perché cambiano anche precisione del preprocessing e numero di thread. B resta sotto R3c nelle stime puntuali anche senza PCA.

Protocollo e codice fissati nel commit `440af702ac3ec00feb52549acb1b4935251dce38`, dopo approvazione di Simone e review del critico. Nessun nuovo fit, nessuna estrazione o GPU; analisi locale degli output già esistenti. Gli score puntuali erano noti prima di questo bootstrap: lettura esplorativa, non un contrasto pre-registrato rispetto alla prima osservazione dei dati.

| Contrasto, punti percentuali | P1, IC95 appaiato | P2, IC95 appaiato |
|---|---:|---:|
| **pool128 − pool768, B** | **−0,497 [−0,749; −0,253]** | **−2,859 [−3,209; −2,513]** |
| **pool128 − pool768, R3c** | **−0,453 [−0,818; −0,078]** | **−1,699 [−2,085; −1,196]** |
| B − R3c, pool768 | −0,553 [−1,105; −0,023] | −1,172 [−3,054; +0,613] |
| B − R3c, pool128 | −0,597 [−1,138; −0,088] | −2,332 [−4,043; −0,813] |
| **Variazione del contrasto B−R3c passando a pool128** | **−0,044 [−0,474; +0,365]** | **−1,160 [−1,740; −0,660]** |

Per EPN612, la pipeline pool128 perde **5,480 pp** per B [−6,094; −4,885] e **3,945 pp** per R3c [−4,420; −3,469]. Il divario B−R3c è **−2,459 pp** [−3,026; −1,858] senza PCA e **−3,994 pp** [−4,698; −3,291] nella pipeline PCA128. Parte dell'ampliamento del divario P2 si vede quindi già su EPN612. P2 continua a pesare EPN612 e UCI alla pari: non si ricalcola usando il numero di soggetti.

Conseguenze per le decisioni:

1. **Limitare le conclusioni di PSD-3 alle viste compresse provate.** Non è corretto attribuire tutto il peggioramento P2 di B al pretraining: l'interazione fra rappresentazione e pipeline contribuisce al confronto osservato. Non è nemmeno corretto dire che la PCA spieghi tutto: lo svantaggio puntuale di B esiste anche con pool768.
2. **Il confronto MLP−logistica dentro PSD-3 resta informativo**, perché usa le stesse feature compresse per ciascuna vista. Non è però una prova che una MLP su tutte le feature sarebbe inutile. Questo rapporto non valuta decoder senza PCA o MLP768.
3. **Non cambiare il lettore temporale in base a questi risultati.** Il protocollo approvato usa già latenti temporali384 senza PCA; il controllo risponde a un'altra domanda e può procedere indipendentemente.
4. **Un'ablazione strettamente causale della PCA sarebbe un nuovo rifit**, con implementazione/precisione/thread invariati e solo la riduzione disattivata. È motivata dalla perdita osservata se vogliamo attribuirne il meccanismo; non è stata eseguita o accodata con questo bootstrap. Riferimento storico per quattro sonde768: circa27 core-ore, non una nuova stima misurata.

Verifiche: quattro NPZ originali raccolti da Leonardo in sola lettura, tutti con SHA atteso; dieci shard PSD-3 già disponibili. Identità esatta di checkpoint, ingressi, righe, target, soggetti e fold; score ricostruiti uguali ai report pinnati entro1e−12. EPN612 conserva etichette testuali native. Supporti: DB2 38.544 righe/32 utenti, DB3 9.834/8, DB6 33.961/8, EPN612 47.100/314, UCI364/19.

Bootstrap: 2.000 campioni per soggetto entro dataset, seme20261102, medesimo campionamento per le quattro celle, pesi P1/P2 congelati. Nessuna replica con classi mancanti o singola classe. Gli IC sono descrittivi, senza correzione di molteplicità, condizionati a lettori/checkpoint fissati: non includono variabilità di refit o pretraining. Piccole differenze dagli IC storici sono attese per il nuovo seme e non cambiano i punti stimati.

Materiali: [rapporto compatto e provenienze](../../results/infra/pca_20261102/paired_report.json), [ricevuta di raccolta](../../results/infra/pca_20261102/collection_receipt.json), [script](../../scripts/pca_existing_oof_compare.py). Il rapporto completo locale ha SHA256 `4f72b363b97ca223f3d8492deca5fc09487899b300ce7e15777d4c7bf2e1a956`; il compatto conserva tutti i risultati e omette soltanto gli array di conteggi soggetto×classe. Predizioni e feature restano fuori dal repository. Il piano del temporale e i suoi comandi sono separati; questo risultato non attesta alcun lancio Leonardo.
