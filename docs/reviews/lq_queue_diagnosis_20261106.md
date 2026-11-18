# Diagnosi della coda LQ19618 — 6 novembre 2026

Su richiesta diretta di Simone: motivo della coda e possibilita' di far partire
il training approvato. Letture SLURM del mattino (job08:22:55CET), nessuna mutazione.
Prove strutturate: `results/infra/coordination_20261106/lq_queue_diagnosis.json`.

Job63593128: PENDING/Priority, eleggibile dal05/11ore16:38, nessuna dipendenza,
Nice0, nessun nodo imposto, nessun vincolo aggiuntivo, StartTimeUnknown.
Chiede1nodo con4GPU/32CPU/480G per4h. PartizioneUP; al campionamento nessun nodo
idle,2892allocated e384mixed, piu' nodi in completamento/pianificati/altri stati.
Un nodo parzialmente libero non soddisfa la richiesta del nodo intero.

Priorita'142879: QOSnormal120000punti, fair-share20996, eta'1874, taglia8.
Fattore fair-share0,839856: il consumo non produce un motivo di blocco quota
in questo job. Nella coda campionata1242job hanno priorita' piu' alta;120 sono
normal a nodo singolo con4GPU. Non e' una posizione FIFO, ne' una previsione
che tutti debbano partire prima: valgono compatibilita' e finestre scheduler.

## Alternative verificate

- La stessa associazioneutente/account ammette normal,bprod,dbg,lprod.
- bprod richiede almeno65nodi: non e' un'alternativa per1nodo.
- dbg ha priorita' maggiore ma30min massimi: non copre questa continuazione.
- lprod ha la stessa priorita'40 di normal: non e' una corsia piu' rapida garantita.
- Le manutenzioni Booster grandi iniziano il10/11 e non spiegano questa attesa.
- Non c'e' una partenza stimata dallo scheduler. Un nuovo invio non rimuove
  la contesa e perde l'eta' della prenotazione originale.

Conferma esterna dei limiti QoS: documentazione ufficiale CINECA
https://docs.hpc.cineca.it/hpc/leonardo.html, letta il06/11.

Il walltime non si puo' abbassare come semplice aggiustamento del job:
`logsquare_train.py::validate_allocations` impone14400s per questa fase e
il launcher legge il TimeLimit reale e lo verifica contro il piano. Un cambio
isolato farebbe fallire la guardia. L'esecuzione attesa e'circa3h06da preventivo,
ma non dimostra un limite piu' corto sicuro. Qualunque diversa segmentazione o
cap richiede emendamento operativo, pin e review; nessuno eseguito qui.

Conclusione: attesa di risorse e priorita', senza un blocco tecnico da correggere.
Nessuna accelerazione equivalente e garantita individuata. Manteniamo la
prenotazione sana; l'owner della chat3 e' stato informato. Nessuna cancellazione,
nuova sottomissione, allocazione interattiva, cambioQoS o modifica del checkpoint.
