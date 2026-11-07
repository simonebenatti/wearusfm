# T100-LQ: controllo degli archivi e dei pin CPU del 07/11/2026

**Esito del controllo circoscritto: APPROVATO, nessuna discrepanza trovata.**
Questa nota verifica gli archivi locali e la derivazione dei tre piani CPU;
non sostituisce la review globale della catena, non rilascia la ridge e non
contiene una valutazione scientifica del candidato.

Controllo eseguito con un programma separato dal generatore, usando soltanto
`gzip`, `hashlib`, `json` e `pathlib`; nessuna importazione del generatore né dei
lettori scientifici. Nessun nuovo test di training o fit, nessun accesso remoto.

## Oggetti effettivamente controllati

Piani in `results/infra/logsquare_evaluation_20261107/cpu_readout/`:

| File | Byte | SHA-256 |
| --- | ---: | --- |
| pose_readout_plan_5000.json | 7048 | `54958f4ee7dbfa0ab6d1e997752b1a6cc5f73bbf3be33d0136474bb3ace9b0c7` |
| pose_readout_plan_19618.json | 7057 | `b020d671ff55e3680e9cf3aca4f2a63bdb0cb5d534b5013fc4ec37402aff5b8a` |
| probe_plan_19618.json | 36681 | `5fd487d182d8438ac484d8a0d6d87d157dc8a64d14e241642a854b60c7f23fb1` |
| resources_probe.json | 2284 | `6ebfb14883f757dc4b3a4f3bd8b87f0f4ed4b4446fd4c8d3b6b8267950f292d5` |

Byte e hash ricalcolati coincidono con `generation_validation.json`.
Il pin della policy nel piano sonde risolve esattamente l'ultimo file della tabella.

## Ricevute e provenance compresse

Sono state decompresse integralmente le quattro `receipt.json.gz` nelle cartelle
`completed_pose5000`, `completed_pose`, `completed_p1` e `completed_p2` del 07/11.
Per ciascuna, SHA-256 e numero di byte **decompressi** coincidono con il pin
originale in `verification.json`; l'hash di ciascun `verification.json` coincide
anche con il pin registrato dal generatore.

| Cartella | Job registrato | Byte originali della ricevuta |
| --- | ---: | ---: |
| completed_pose5000 | 63615952 | 3793503 |
| completed_pose | 63772648 | 3793883 |
| completed_p1 | 63772649 | 178860 |
| completed_p2 | 63772650 | 268388 |

Le ricevute riportano `complete: true` e `checkpoint_unchanged: true`; le
verifiche archiviate riportano `COMPLETED 0:0`. Non è stato interrogato nuovamente
SLURM per questa nota.

Sono state decompresse anche tutte le nove provenance. Il loro SHA-256 originale
coincide, per ogni cache, fra archivio decompresso, output del receipt, verifica
raccolta e piano CPU. Il JSON decompresso coincide integralmente con la copia
inclusa in `verification.json`.

| Dose/task | Cache | Byte della cache nei pin concordanti | Byte della provenance decompressa |
| --- | --- | ---: | ---: |
| 5000 | train | 257774193 | 2523 |
| 5000 | val | 2485995330 | 2513 |
| 19618 | train | 257554387 | 2525 |
| 19618 | val | 2484248223 | 2515 |
| P1 | ninapro_db2 | 829630230 | 2637 |
| P1 | ninapro_db3 | 211675902 | 2634 |
| P1 | ninapro_db6 | 939641978 | 2636 |
| P2 | epn612 | 729002734 | 2664 |
| P2 | uci_emg | 5606358 | 2659 |

I nove percorsi delle cache sono distinti. Per tutti e nove coincidono percorso,
SHA-256 e byte della cache, percorso e hash della provenance, percorso e hash
del receipt, hash del checkpoint e SHA del codice di estrazione. Le identità
sono teacher/T100-LQ, alla dose prevista, con configurazione del modello identica
fra receipt e provenance. Per P1/P2 coincidono inoltre hash del piano di estrazione,
`input_sha256`, `splits_sha256` ed `extraction_recipe`, anche contro l'identità
del dataset nel piano sonde.

Il checkpoint a 5000 è `6233b0705e8db2caaf3c84068812053878ec2291887be88740302fe6c844b0a9`.
Posa, P1 e P2 a 19618 condividono
`316fa96fb10a941a1e3be230e2738b19bbfb83f691dd940e4d32e089ebb2c214`.
Il codice di estrazione a 5000 resta `9720fc8f7510bd8e7640164630da30515a8b3210`;
quello dei tre job finali è `c1963236a08690998c812c86278bdefbf745ccf1`.

Le NPZ remote non sono state rilette da questo controllo locale: i loro hash e
byte sono confrontati con la misura remota archiviata. Il controllo effettivo
dei file, del supporto, dei fold e del contesto numerico resta nei lettori runtime.

## Differenze dai preparatori del 04/11

Confronto strutturale completo dei JSON, non limitato a una lista di chiavi scelte:

| File | Sole chiavi di primo livello diverse |
| --- | --- |
| pose_readout_plan_5000.json | `status`, `caches` |
| pose_readout_plan_19618.json | `status`, `caches` |
| probe_plan_19618.json | `status`, `caches`, `resources`, `release`, `preparation_note` |
| resources_probe.json | `status` |

Le differenze sono la compilazione dei pin effettivi, lo stato di disponibilità,
il pin della policy rilasciata e il testo/evidenza della consegna. Ricette, sorgenti
pinnati, dati di riferimento, controlli storici, contrasti, fold e tetti restano
identici ai preparatori. In entrambi i piani posa `fit_release` è ancora nullo:
il checkout A non rilascia anticipatamente i fit ridge.

## Verifica indipendente finale della catena

**APPROVATO per preparatori posa e sonde.** Controllati manifest, percorsi,
dipendenze e risorse: 16 job scientifici e 2 checkout condivisi, 51 core-ore
massime, zero GPU. Gli SHA dei piani per checkout A coincidono; i quattro job
posa su checkout B conservano SHA futuri nulli e richiedono le release successive
ai benchmark. Dipendenze complete: quattro shard P1 → aggregato P1; shard P2 →
aggregato P2; entrambi gli aggregati → confronto. Output distinti fuori dal codice.
Confermati pin delle nove cache, originali delle ricevute/provenienze, riferimenti
storici e identità numeriche. Ricalcolata la contabilità: quattro estrazioni
1,739444 GPU-ore / 13,915556 core-ore; tre finali 1,221111 / 9,768889, checkout
esclusi. Nessuna ridge rilasciata dal solo manifest: restano obbligatori gate
misurati, release pinnata e nuovo checkout B. Nessun blocco trovato.

## Verifica indipendente delle release ridge e del manifest B

**APPROVATO il rilascio dei due fit posa dopo pubblicazione e checkout B.**
Decompressi e ricalcolati SHA/byte dei quattro rapporti preparatori; identita'
di job, accounting, piano, codice A `61f0021753fedf170eb1a7d40fde3fa40a638861`,
cache, riferimenti e ricetta coincidono. Tutti i job sono registrati COMPLETED
0:0; validate usa 2 CPU, benchmark 8 CPU, entrambi 30G e nessuna GPU. Il contesto
numerico dei benchmark coincide esattamente con quello storico a otto thread.
I cinque sorgenti restano invariati.

Ricalcolo separato dal generatore e dalla funzione del gate:

| Dose | V, accounting validate (s) | S (s) | E (s) | Formula prima dell'arrotondamento (s) | Richiesta arrotondata (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5.000 | 18 | 3,076266342 | 0,089982333 | 226,513050127 | 240 |
| 19.618 | 19 | 2,925895695 | 0,090236562 | 227,036891927 | 240 |

Formula invariata: `ceil_a_60s[1,5*(2V+11S+13E)+120]`; entrambe le richieste
sono entro il tetto congelato di 1.800 s. E' un proxy prudente, non una durata
garantita. Nei piani sotto `cpu_fit_release` cambia soltanto `fit_release`;
identita' scientifica e byte dei piani A coincidono con il commit congelato.
SHA dei piani B: 5.000 `ce86036be16aa3396d904ed97c6a545ce0da0a24a7009e73df99fc35144da9d1`;
19.618 `4b9afef400c7c18122e34aeed5a94478f5cb90ffada7895b5d331ed93c9f6919`.

Il manifest B risolve solo percorso e SHA dei quattro job gia' previsti;
ID, risorse, output e gate restano quelli pubblicati. Totale CPU invariato a
51 core-ore; i confronti richiedono ancora il rapporto ridge completo e il suo
SHA effettivo. Nessuna modifica alla catena sonde sul checkout A. Controllo
locale in sola lettura degli artefatti; nessun job, fit o accesso remoto.

## Verifica dei rapporti ridge e degli ultimi invii

APPROVATO per i due confronti. Report originali verificati per byte/SHA,
accounting COMPLETED 0:0 (55 e 61 s), 8 CPU / 30G / zero GPU, ID del ledger,
piani e codice B, sorgenti e contesto numerico, fit_release e pin di coefficienti
e predizioni coerenti. Argomenti del piano concreto corretti: 2 CPU / 30G /
30 min ciascuno, zero GPU, 2 core-ore totali già previste. Omissione di afterok
accettata: i fit sono già conclusi e verificati, tutti i gate runtime restano
attivi. Nessuna interpretazione delle metriche e nessuna nuova scelta scientifica.
