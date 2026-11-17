# Audit contrastivo 64652559: copertura valida, timeout operativo

Riferimento: checkout `f09bf857eee24f02aff25fa743853d60fcab7f03`, digest esecutivo
`ab2d6460bfe090e4adba8cfd480b726077005c4ed6ae10db7ebbd5909ddf808b`.
Simone chiede di lanciare la calibrazione se l'audit passa, oppure leggere e spiegare il timeout.
**Nessun job nuovo sottomesso.** Nessuna modifica al codice, alla ricetta o ai criteri.

## Esito verificato

| Evidenza | Risultato |
|---|---|
| Job principale SLURM | `TIMEOUT`, ExitCode e DerivedExitCode `0:0` |
| Sottopasso `.batch` | `COMPLETED 0:0` |
| Intervallo registrato, Europe/Rome | 17/11, 08:07:49–09:10:56 |
| Durata / limite richiesto | 3.787 s (63:07) / 3.600 s |
| Rapporto | JSON completo, 20.275.571 byte, scritto alle 09:10:55 |
| SHA256 rapporto | `5c720fadbcb12fd5458bc0e19cd03f28fec30ad5973ca72739f01bf1b3d7547b` |
| Replay | 1.024 passi consecutivi, 32.768 finestre, geometria 4 rank × 6 worker |
| Pannello 256–511 | InfoNCE 8.179 / 8.192 = **99,8413%**; RnC attivo 256 / 256 |
| Intervallo attivo 256–1.023 | InfoNCE 24.527 / 24.576 = **99,8006%**; RnC attivo 768 / 768 |
| Pin | Sorgenti, configurazione canonica e quattro input corrispondenti |
| Descrittori | Limite conservativo 800, disponibilità 131.072 |

Il rapporto e il log sono completi: lo script stampa il riepilogo soltanto dopo la scrittura/chiusura
del JSON e il controllo finale di integrità di input, configurazione e sorgenti. Nessuna eccezione
Python nel log. Il sottopasso è terminato con codice zero. Questo non cambia lo stato `TIMEOUT` del job principale.

Nel pannello, CapgMyo ha 13 query e zero ammissibili, esente dal solo requisito per dataset come deciso
prima del job. Rimane nel denominatore globale. Gli altri 16 dataset hanno copertura 100%, minimo
58 query contro la soglia 32. Nell'intervallo attivo completo, CapgMyo è 0 / 45 e DB3 233 / 237;
gli altri dataset hanno copertura 100%. Non sono attesi blocchi completi di 1.024 update attivi:
256 passi sono di riempimento e 768 attivi.

Ricalcolo locale riproducibile con `audit_completed_20261117/verify_report.py` e ricalcolo indipendente
del subagente `audit_timeout_review`: conteggi, frazioni, esenzione e verdetti corrispondenti;
anche il lettore `_coverage` passa. Il controllo non riesegue il processamento dei segnali grezzi.
Input reali ricalcolati via SHA sul cluster dopo il job e checkout ancora pulito. Nessuna review
di Claude nuova viene attribuita a queste verifiche.

## Perché il job ha superato il tempo

Il preventivo di 60 minuti era insufficiente. L'audit riproduce **24 flussi logici in serie**:
gli 8 core richiesti non sono otto processi di caricamento paralleli. Inoltre usa il `PretrainLoader`
completo, che legge/filtra i segnali e costruisce anche ancore, codici e vicini poi non usati
dal controllo di candidati/supporto. OMP, OpenBLAS e MKL erano limitati a un thread.

SLURM registra 35:38,449 di CPU effettiva, picco RAM **21,51 GiB su 30**, e `MaxDiskRead=36187.38M`.
Nessuna evidenza di memoria esaurita. Il codice non emette progressi o tempi per fase, quindi non
si può attribuire quantitativamente il ritardo al disco, alla costruzione delle ancore o ad altro.
L'audit usa più lavoro del necessario e non sfrutta i core allocati in parallelo: è un problema
del disegno/preventivo operativo, non evidenza contro InfoNCE o RnC.

La configurazione letta da SLURM riporta `OverTimeLimit=3 min`, `KillWait=180 sec`. La tolleranza
spiega perché l'esecuzione ha potuto proseguire oltre i 60 minuti; l'accounting da solo non identifica
l'istante dei segnali o perché la conclusione sia avvenuta sette secondi oltre quei tre minuti.
Non trattare questa tolleranza come una futura risorsa garantita.

## Consuntivo e seguito

Audit: **0 GPU-ore, 8,415556 core-ore allocate**, contro tetto 8; CPU effettiva 0,594014 core-ore.
Con i due checkout già chiusi: **8,461111 core-ore allocate**. Il massimo residuo del vecchio piano
porterebbe il totale a 129,461111 contro 129,333333: lo scarto va contabilizzato, non nascosto.

**Gate di contenuto superato; gate operativo non superato.** Il piano e l'handoff congelati richiedono
conclusione regolare e arresto delle dipendenze in caso di fallimento. Il runtime verifica il contenuto
del rapporto, ma non lo stato SLURM: questa distinzione non autorizza a eludere il contratto esecutivo.
Calibrazione e training rimangono non rilasciati in questa consegna.

Proposta operativa, **non applicata**: recuperare questo rapporto completo mediante una rettifica
puntuale, con hash pinnato e review, mantenendo visibile `TIMEOUT` e senza ripetere l'audit. Per mantenere
il tetto globale si possono ridurre da 300 a 240 secondi i tre checkout ancora da fare: massimo totale
129,261111 core-ore. I due checkout già misurati hanno richiesto 30 e 11 secondi; la riduzione riguarda
soltanto il deployment e non i job scientifici. La rettifica dovrebbe precedere la release della
calibrazione; nessuna soglia, λ, selezione o durata GPU verrebbe cambiata. Non è una nuova prova scientifica.

Evidenze in `results/infra/contrastive_comparison_20261116/audit_completed_20261117/`:
rapporto originale compresso senza perdita, log, accounting/configurazione SLURM, verifica dei pin,
verificatore e `verification.json`. Nessun rilancio, cancellazione o monitor in background.
