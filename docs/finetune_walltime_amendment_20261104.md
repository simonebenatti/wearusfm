# Fine-tuning EPN: misura delle risorse e adeguamento dei tempi

Il benchmark63525812 è COMPLETED0:0,457s: **0,126944 GPU-ore e1,015556 core-ore**.
Report originale e misure in `results/infra/consolidation_20261104/benchmark/`;
SHA del report `a5eada1ab8cc0b265c3c7b2ba03a0b7f43ca9b1e819db2011362a00971f80c72`.
Verificati codice17dd455, piano, sorgenti, ricetta, input e checkpoint. Nessun
checkpoint scientifico né score downstream prodotto dal benchmark.

| Regime | Secondi per update | Proiezione un candidato | Due candidati | Esito col limite originale14.100s |
|---|---:|---:|---:|---|
| Frozen | 0,18940 | 7.959,88s (2h13) | 14.666,55s | split |
| Fine-tuning | 0,62991 | 23.558,54s (6h33) | 45.863,87s | bloccato |
| Scratch | 0,62924 | 23.535,00s (6h32) | 45.816,80s | bloccato |

Il gate memoria passa: proiezione host circa32,39GiB su100, CUDA riservata massima
circa6,58GiB su63,42. Il limite è il tempo richiesto per cinque epoche; non è un
risultato negativo sull'utilità del fine-tuning. I tempi includono i margini
preregistrati (training×1,5, validation/held addebitati come training, costi fissi).
Non sono durate effettive osservate di un fit completo. Nessun fit sottomesso,
incluso frozen, in attesa di rendere eseguibile l'intero confronto.

## Emendamento operativo autorizzato

Simone: «poi se devi alzare limiti di tempo vai tranquillo», dopo aver chiesto
di stampare il prompt per Claude. La review riguarda questa implementazione;
non occorre una nuova autorizzazione all'aumento del walltime dopo convergenza.

- Frozen:4h, gate14.100s. Finetune e scratch:7h, gate24.900s,300s di riserva.
- UnaGPU,8CPU,100GiB; stessi dati, cinque epoche, batch8, seed0, griglie testa
  ed encoder, inizializzazioni, scelta validation-only e held una volta.
- Policy joint/split invariata, applicata ai limiti per regime; se anche il
  singolo supera il nuovo limite si ferma, senza cambiare dose/batch/LR.
- Proiezione prudenziale attuale24candidati:122,3409GPUh. Tetto atteso24job:
  **144GPUh e1152coreh** (8frozen×4h +16FT/scratch×7h). Se un regime rientra con
  due candidati si raggruppa secondo la stessa regola, pubblicando il costo minore.
  Tetto ipotetico tutto joint72GPUh; i due tetti sono alternativi.
- Partition boost_usr_prod osservata MaxTime24h, QoS normal senza MaxWall proprio;
 7h è compatibile con questi limiti osservati. La coda non ha un tempo garantito.

L'emendamento è successivo al benchmark di risorse, prima di qualunque fit e
di qualsiasi score del confronto. Non modifica F-R2/V-R1 o le soglie di lettura.
Continuazione63525803→63525805→63525809 indipendente, checkout17dd455 invariato.

## Codice, pin e prossimo benchmark

`fit_packaging` deve usare il limite per regime; piano e launcher devono imporre
la stessa allocazione. Entrambe le policy saranno nella ricevuta benchmark e
verificate prima dei fit, oltre alla provenienza già esistente.

Il controller fa parte di `source_identity`: la modifica invalida il vecchio pin
per l'esecuzione dei fit, anche se la misurazione scientifica non cambia.
Conserviamo63525812 come evidenza, **non aggiriamo il controllo dei sorgenti**.
Dopo review: nuovo benchmark train-only sul codice corretto, massimo0,5GPUh/4coreh,
con checkout nuovo (1CPU richiesta,2GiB,10min, massimo allocato osservato2CPU:
0,333333coreh). Nessun riuso o modifica del checkout della continuazione.

Il prossimo report stabilirà joint/split e costi effettivi. Piani e pin pubblicati
prima degli invii; submitter/ledger comune, namespace FT dedicato, niente monitor.
Questa consegna prepara l'emendamento e non sottomette il nuovo benchmark o i fit.

Verifiche:44test mirati passati dall'autore e dal critico indipendente, nessun
rilievo bloccante; [review interna](reviews/finetune_walltime_critic_20261104.md).
Questa verifica non sostituisce la rilettura Claude richiesta da Simone.
