# Rilascio del controller dei lettori dopo P1-r2 — 17/11/2026

**APPROVATO**, nessun rilievo bloccante. Review indipendente del critico e 277 test superati in 33,09 s.

Il profilo esplicito `--p1-reader-recovery` collega le cache accettate di P1-r2 (64620434) alle estrazioni posa/P2 già accettate. I piani reali sono ricostruiti dalle tre ricevute e confrontati con la consegna verificata. Il controllo 30M resta identico.

Il controller e i due helper sono locali al Mac. I 103 sorgenti scientifici e i sei file numerici/dispatcher/submitter controllati restano invariati rispetto al codice congelato df2de35. I checkout documentali discendono da quel commit e aggiungono soltanto documenti: nessuna eccezione ai controlli di rilascio.

I cancelli vincolano la generazione delle righe agli hash del controller, alle evidenze, ai documenti, ai checkout e al ledger. La validazione deve essere accettata prima dei benchmark; tutti i benchmark devono essere accettati prima dei fit. Il budget mantiene la prenotazione piena per job senza accounting accettato e sostituisce una sola volta i costi verificati. Massimo iniziale 349,959445 su 351 core-ore; tre checkout residui già inclusi.

Limiti: il gate non prenota atomicamente la QoS; un rifiuto impone lo stop. La nuova tranche attende COMPLETED 0:0 della precedente. Il riconoscimento GPT sugli ID CPU registrati è esteso anche al lettore storico del ledger, senza cambiare operazioni numeriche. Nessuna deduzione causale su Lustre: P1-r2 non ha esercitato transizioni di timestamp.

Audit riproducibile, hash dei file e dettagli dei test sono in `results/infra/scaling_300m_20261104/continuation19618_20261106/submissions/p1_reader_release_20261117/independent_review.json`; riproduttore affiancato. Autorizzazione operativa: richiesta di Simone del 17/11, riportata nel piano e nelle due sezioni del registro. Questo rilascio non attesta risultati di benchmark non ancora eseguiti.
