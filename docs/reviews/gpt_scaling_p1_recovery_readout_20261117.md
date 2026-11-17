# P1-r2 a 19.618 — raccolta e accettazione, 17/11/2026

**ACCETTATO.** P1-r2 64620434 e' completo e integro. Le sue tre cache possono
essere unite alle due di posa e alle due P2 gia' accettate, con provenienza
distinta per produttore. Possiamo procedere al rilascio dei piani aggiornati,
alla validazione completa degli ingressi e ai benchmark dei lettori congelati.
I fit seguono i benchmark e il calcolo delle risorse, come previsto.

La raccolta non ha inviato job, rieseguito estrazioni, modificato checkpoint,
cache, ricevute o guardie dei lettori. Non contiene nuove misure di qualita'
del FM: e' la chiusura tecnica dell'estrazione necessaria per ottenerle.

## Esito e integrita'

| Controllo | Evidenza | Esito |
|---|---|---|
| Scheduler | 64620434 COMPLETED 0:0, 16/11 22:23:29–22:53:01 CET, 1.772 s | Conforme |
| Ricevuta | Completa; job/commit/manifest/checkpoint/passo/teacher esatti; nove file attesi nella cartella nuova | Conforme |
| Nuova guardia | Sei letture complete, dodici eventi, audit identici in ricevuta e sidecar; validatore eseguito fra le due letture | Conforme |
| Stabilizzazione | Zero riletture aggiuntive, zero transizioni mtime/ctime, zero errori; variazioni soltanto di atime | Nessuna tolleranza attivata |
| Rilettura indipendente | Tre flussi integrali via SSH, SHA-256 sul Mac, 4.619.980.998 byte; stesso oggetto/percorso e attributi protetti prima/dopo | Tre hash identici alle ricevute |
| Ricetta e ingressi | 103 sorgenti scientifici invariati, checkpoint d7ebe9d8 a 19.618, gain salvato, input verificati dal wrapper; runtime uguale a P2 | Conforme |
| Posa/P2 preservate | Ricevute identiche a quelle accettate; sette cache 300M composte senza riscriverle | Conforme |
| Controllo 30M | Piani completi con le sette cache e i riferimenti congelati originali | Schemi dei lettori validi |

| Cache P1 | Utenti | Finestre | Pool | SHA-256 |
|---|---:|---:|---|---|
| NinaPro DB2 | 32 | 38.544 | 38.544 × 1.792 | 6fd9b33473cff76848166b6ce089cb8521331c77775cc4b9db25d11efa988d6b |
| NinaPro DB3 | 8 | 9.834 | 9.834 × 1.792 | e69bc54dccaec1d1913f28df39f90f266944dc4c304eb9f83a9334bcf4c96db6 |
| NinaPro DB6 | 8 | 33.961 | 33.961 × 1.792 | fd8656bc18131fb77ed55db685440c7f0eddbd0452e14b11c0b8e4b8e9622847 |

Il trasferimento aggiuntivo e' in sola lettura e non conserva copie grandi sul
Mac. Il nodo di login trasferisce i byte e legge gli attributi; non calcola
hash o feature. Ogni SHA locale viene confrontato con quello prodotto sul
nodo GPU; stat sul percorso e sul descriptor prima/dopo il trasferimento
devono coincidere con la fine dell'audit del job, escluso atime.

La nuova rilettura integrale riguarda P1. Per posa/P2 sono state raccolte le
ricevute e le provenienze originali e verificati i metadati correnti; la loro
rilettura completa e quella del controllo 30M restano ai validatori congelati
della fase successiva. Non si confondono queste due coperture.

## Cosa sappiamo del problema precedente

Il recupero e' riuscito, ma in questo run non si e' osservata l'instabilita'
dei timestamp. Il ramo di stabilizzazione non e' stato esercitato. Non e'
quindi dimostrato che Lustre fosse la causa dei due fallimenti, ne' che una
tolleranza abbia risolto il problema storico. La nuova guardia offre controlli
di contenuto/oggetto e osservazioni esplicite, mantenendo il rifiuto delle
vere modifiche. Non possiamo ricostruire a posteriori l'esito che avrebbe
avuto la vecchia sequenza di lettura su questo run.

La diagnostica resta NOT_REPRODUCED. I job 64572000 e 64596915 restano FAILED;
non sono stati adottati i loro file. Il P1 qui accettato e' soltanto 64620434.

## Prosecuzione dei lettori

Piani e provenienza sono in
`results/infra/scaling_300m_20261104/p1_recovery_readout_20261117/reader_inputs/`.
Il join esplicito mantiene P1 da `e4ebdbfa` e posa/P2 da `df2de35`, lo stesso
checkpoint, ricetta, supporti, fold e ancore del confronto. I costruttori
`pose_plan`/`probe_plan` e i validatori numerici congelati accettano i piani.
Cinque alterazioni intenzionali della provenienza sono rifiutate.

Resta un raccordo operativo da applicare prima dei job: il vecchio
`build_reader_plans()` indica P1-r1 e assume un solo commit. Deve consumare il
join e i piani qui pubblicati, senza cambiare evidenze storiche o sorgenti
numerici. La presente consegna non finge che quel controller sia gia'
aggiornato, ne' che questi documenti siano gia' presenti in un nuovo checkout.

Poi valgono i controlli operativi previsti (pin, checkout, ledger, uscite,
coda e quota), validazione degli ingressi e benchmark train-only; budget e
distribuzione dei fit derivano da quei tempi/RSS. Le guardie dei lettori
rimangono invariate. Nessun nuovo consenso e' richiesto da questa review per
operazioni gia' autorizzate; qui non e' stato avviato nessun lettore.

## Consuntivo e prove

P1-r2: **0,492222 GPU-ore e 3,937778 core-ore**; MaxRSS 25,095 GiB su 100 GiB.
P1 cumulativo, inclusi i due fallimenti: 1,513056 GPU-ore su 3,10.
Stadio estrazioni e diagnostica: 5,478056 GPU-ore su 7,30 e 43,824444 core-ore
allegate su 58,40. I due checkout aggiuntivi valgono 0,041667 core-ore.
Inviluppo CPU-only, inclusi lettori futuri e checkout effettivi:
349,959445 su 351 core-ore; non e' il consumo gia' avvenuto dei lettori.
Il margine numerico residuo non autorizza un ulteriore P1.

In `p1_recovery_readout_20261117/`: `acceptance.json`, ricevute/sidecar/log
integrali, eventi, snapshot remoto, `sacct.txt` testuale originale,
`payload_stream_verification.json`, accounting ricalcolato, tre audit
indipendenti con riproduttori, piani e provenienza mista. Nessun punteggio
scientifico e' stato aperto o stimato in questa fase.
