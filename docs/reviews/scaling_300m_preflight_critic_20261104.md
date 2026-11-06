# Collaudo 30M/300M: review scientifica e algoritmica del 04/11/2026

**Verdetto: favorevole alla consegna del collaudo a Claude, con i tre rilievi chiusi.** Questa review riguarda il controllo tecnico da 128 aggiornamenti per taglia. Non certifica la campagna scientifica a 5.000 passi e non dispone alcun lancio. Simone ha richiesto la review di Claude prima delle sottomissioni.

File esaminati: [protocollo](../scaling_300m_protocol_20261104.md), [script](../../scripts/scaling_preflight.py), [launcher](../../scripts/slurm/scaling_preflight.sbatch), [piano](../../results/infra/scaling_300m_20261104/preflight_plan.json), [risorse](../../results/infra/scaling_300m_20261104/resource_policy.json) e [test](../../tests/cpu/test_scaling_preflight.py). Riletti anche aggiornamento, conteggi distribuiti, calibrazione e diagnostiche richiamati dal trainer e da `tokenizer_preflight.py`.

## Disegno e significato

Il disegno è adeguato alla domanda tecnica: due modelli nuovi, 30M e 300M con T100-G, eseguiti in sequenza sullo stesso prefisso di 128 batch globali da 32. Cambiano soltanto le dimensioni e profondità del preset LADDER; finestra, passo, obiettivo, dati, maschere, batch e schedule nominale restano fissati. Il confronto misura fattibilità e costo della ricetta attuale su due taglie, non una legge generale di scaling.

I primi 64 aggiornamenti usano il warmup nativo. I successivi 64 impongono il picco di LR sullo stesso modello e sullo stesso AdamW, conservando i momenti e il teacher EMA. È correttamente dichiarato uno stress anticipato, non la simulazione di uno stato allenato fino a 1.962 passi. Il run scientifico eventuale ripartirà da zero con warmup completo; il collaudo non scrive checkpoint riutilizzabili.

Il banco del tokenizer e il gain iniziali sono verificati uguali; non si pretende l'uguaglianza dei pesi di forme diverse o del consumo RNG del modello. Gli hash appaiano invece i batch effettivamente consegnati, i segnali, le origini, le maschere, i target e i supporti effettivi di query/posa. Il loader esclude validation ed EPN/UCI. La calibrazione sintetica esistente non viene adattata a uno score downstream.

L'aggiornamento richiama il calcolo di produzione, con denominatori distribuiti, clipping, AdamW ed EMA. Il fattore di clipping usa la norma globale pre-clip; l'update relativo è la norma della variazione effettiva, comprensiva degli effetti dell'ottimizzatore e del weight decay, divisa per la norma precedente dei parametri osservati. Non è una norma del solo gradiente né una prova che due taglie abbiano ottimizzazione equivalente.

## Rilievi e correzioni

| Rilievo | Correzione verificata | Stato |
|---|---|---|
| Il gruppo storico `backbone_first` conteneva soltanto `norm_t.weight`, poco informativo per la dinamica delle grandi matrici del 300M. | Aggiunte matrici QKV temporali e prima proiezione MLP del primo e dell'ultimo blocco; nomi esatti registrati nell'identità iniziale. Restano gruppi frontend, proiezione e posa. | Chiuso |
| Il pairing usava `zip(records)` senza imporre il numero dei record, potendo accettare una lista troncata. | Richiesti esattamente 128 record per braccio, step consecutivi e sequenza di fasi 64/64; test negativi per record mancanti, step e fase errati. | Chiuso |
| Il preventivo provvisorio riportava un tempo T100-G5000 diverso dal summary congelato. | Input corretto a **3.883,029760837555 s**, chiamato esplicitamente tempo del trainer. Proxy ricalcolato a **15.208,075541384995 s**, verificato indipendentemente contro il summary e il rapporto storico. | Chiuso |

Inoltre è stato aggiunto l'oracolo della transizione nativa/stress: due aggiornamenti consecutivi confrontano pesi, teacher, gruppi AdamW e tutti i momenti con la sequenza esplicita di produzione. Questo copre la conservazione dello stato fra fasi, oltre ai test delle singole fasi.

## Gate, misure e risorse

I due modelli non restano residenti contemporaneamente. Il riferimento della cattura del decoder viene spezzato prima della liberazione; segue la raccolta degli oggetti e la pulizia della cache prima della taglia successiva. La memoria è verificata sui massimi di ciascun braccio e dei rank, con tetto allocato del 90% della capacità rilevata. Le copie diagnostiche dei gruppi sono incluse: il valore è conservativo per gli aggiornamenti osservati. Memoria riservata e allocata sono riportate distintamente.

I tempi sommano i massimi fra rank delle componenti forward/backward e clip/AdamW/EMA; sincronizzazioni e diagnostiche sono separate. I primi otto aggiornamenti di ogni fase non entrano nei riassunti. Il protocollo riconosce che loader, pannelli e checkpoint incidono sul training completo: il throughput misurato non viene presentato come walltime totale garantito.

Output nuovo fuori dai checkout, identità del codice e dei file controllate, nessun resume implicito. Dose completa su quattro rank, valori finiti, gradienti/update osservati non nulli e parità degli input sono gate tecnici. Un fallimento non produce una ricevuta finale completa e non autorizza retry ciechi o cambi di batch. Allocazione massima: **4 GPU, 32 CPU, 480G, 30 minuti**, ossia **2 GPU-ore e 16 core-ore allocate**; scadenza interna a 25 minuti. Checkout separato: 0,333 core-ore.

Il clipping frequente non è da solo una diagnosi di instabilità con AdamW. Uno stress finito di 64 passi al picco non dimostra che LR e clipping siano ottimali né copre tutte le forme del corpus. Questi limiti sono espliciti e non costituiscono un motivo per trasformare il collaudo in una ricerca di iperparametri sullo sviluppo.

## Verifiche e perimetro residuo

L'autore ha eseguito la suite aggiornata: **23 test passati in 12,99 s**, comprendendo `test_scaling_preflight.py` e la suite storica `test_tokenizer_preflight.py`; sintassi Bash e `git diff --check` puliti. La review ha controllato direttamente le correzioni, il nuovo oracolo e il calcolo del preventivo. Non è stata eseguita una prova GPU durante questa review.

Restano intenzionalmente fuori dalla consegna eseguibile: training scientifico a 5.000 passi, controller delle estrazioni D896/pool1792, benchmark CPU, letture appaiate contro il 30M a 5.000 e rispettivi preventivi misurati. Le guardie D384/pool768 delle vecchie campagne non vanno aggirate. Questi lavori saranno necessari prima del run scientifico; **non devono essere completati per far revisionare adesso a Claude il collaudo**.

Nessun nuovo job, allocazione, commit o push effettuato dal revisore.
