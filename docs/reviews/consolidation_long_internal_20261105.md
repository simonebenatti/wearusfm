# Valutazione lunga: review interna prima di Claude

Review indipendente in sola lettura dei moduli scientifici, controller,
piano concreto e manifest. Non sostituisce la review esterna richiesta a Claude.
Nessun job, SSH o modifica degli agenti revisori.

Esito finale: nessun blocco residuo rilevato. Controllati:

-82jobcalcolo+4checkout,82ID/output unici,40celle classificazione e48job fisici,
copertura esatta dei4fold inclusi i16shard DB2.
-Tetti9GPUh e1711,666667coreh ricalcolati; contesto storico e preventivi
misurati distinti dalle allocazioni massime.
-Tre endpoint legati alle ricevute reali, configurazione/stato del checkpoint,
gain e tensori teacher restaurati esattamente, immutabilità postestrazione.
-Posa dal loader originale, mean_K FP16 e testa FP32 separati; pairing E2
prima del crop6 e valutazione sul supporto comune; CV sul solo train.
-Classificazione con raw fingerprint/ordine/ruoli appaiati, griglia e
convergenza12000, audit persistente, OOF e manifest verificati.
-Fonte/ricetta del piano ricalcolate, pin locali verificati, tutti82argomenti
CLI analizzati con il parser reale.

Rilievi corretti durante la preparazione: legame della copia19k al
persistent_source, hash/checkpoint e stato del modello dopo estrazione,
misura delle scritture/hash delle predizioni nel benchmark posa,
risorse DB2 in due shard fissati prima della misura, controllo di protocol/kind
dei report della sintesi. Un sospetto importjson mancante era una vista
superata durante la lettura: il file corrente contiene l'import, nessun
difetto confermato su quel punto.

Da convergere esplicitamente con Claude: i due pannelli primari posa senza
un unico riferimento globale; baseline storica dichiarata quando la parità
non è bit-identica; trasferibilità del benchmark fold0 ai nuovi checkpoint;
inviluppo45min delle estrazioni e politica di quota condivisa.
Questi punti sono visibili nel protocollo e nel prompt; nessuna review esterna
è dichiarata prima della risposta effettiva.
