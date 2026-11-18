# Review indipendente del completamento 300M/5.000 e del piano estrazioni

**VERDETTO: APPROVATO** il completamento tecnico del job **63532932** e il
passaggio alle **cinque estrazioni già previste**, con i pin effettivi esaminati
qui. Nessun blocco rilevato nelle prove raccolte. Restano obbligatorie le guardie
integrali del controller sul nodo, il checkout congelato pubblicato e le verifiche
operative di destinazioni nuove e ledger. Questa review non è un risultato
scientifico sul vantaggio del 300M e non dispone una continuazione a 19.618.

Review svolta il **5 novembre 2026**, in sola lettura dei sorgenti e degli
artefatti raccolti. Training runtime
`dc96559405b5fff971a3be06a6ca59c992d342dc`; piano training SHA256
`66aac4e8cfaf7bb85f36ce78a1f3454fe1fb696a17f0b2fc5a7d3a3b86fb4c66`.
Contratto scientifico SHA256
`14b044de1df7be3e55c836685342d94ae48f9d072dc6784b33cca197dab47df1`.

## Completamento verificato

Le quattro copie in
`results/infra/scaling_300m_20261104/scientific5000_completion_20261105/`
sono state rihashate localmente e confrontate con la nuova lettura remota:

| File | SHA256 |
|---|---|
| `receipt.json` | `0ac5fe8130da2759113fe14af33c7326c42a20d2e7df5426da96ba9e7f415aa8` |
| `config.json` | `6120b82627160df67fc5ef7b7a9760d5f2def03175dd7e9677213c20ee2b76e0` |
| `summary.json` | `f0975c4de0f810b0b5782984e699b50c41c1c67ad930aa9dd0bd650979182af9` |
| `launch.json` | `f1521ce7464bb7e43ef5e1f1ae31e20db226c4464284c824e566b39058bd23e9` |

Eseguiti `check_source_contract` e `validate_plan` sui file locali reali:
**103 sorgenti verificati**, configurazione completa e comando effettivo uguali
alla ricetta ricostruita. Rihashati anche i **9 file congelati** del manifest
scientifico, senza differenze. Confrontati i quattro gate, i pin di input,
sorgenti scientifici/esecutivi e controllo storico in launch, receipt e piano.
La ricevuta finale lega correttamente job, dose, codice, piano, checkpoint e
contratto; dichiara endpoint scientifico completo e nessuna continuazione
automatica. Il wrapper la scrive soltanto dopo la verifica finale dei checkpoint
e l'invariabilità degli ingressi.

`summary.json` conferma **5.000 update**, `stopped=max_steps`, **292.054.847
parametri**, quattro rank, otto finestre per rank, accumulo 1, prefetch attivo.
La configurazione resta T100-G/100 ms, passo 25 ms, D896, 14 teste, 3 blocchi
locali, 18 backbone e 2 decoder; seme 0, LR 1e-4, warmup 1.962, WSD 49.046,
ancore 0,2, posa 1, JEPA 0. Il comando è fresco, senza inizializzazione dal
checkpoint S3, resume o branch. Questi sono controlli della ricetta effettiva,
non nuove condizioni scientifiche introdotte dopo aver visto le loss.

`sacct.txt` registra job principale, batch ed extern **COMPLETED, 0:0**:
07:16:12–11:22:13 CET, **14.761 secondi**, quattro GPU, 32 CPU, 480G.
Consuntivo ricalcolato: **16,401111 GPU-ore** e **131,208889 core-ore**, entro
22/176; checkout contabilizzato separatamente. Il trainer riporta
14.665,053557 secondi e **37,751831 GiB** di massimo CUDA fra i rank. MaxRSS
batch 93.553.760 KiB riguarda la memoria host, distinta da quella CUDA.
Python 3.11.6, Torch 2.2.0a0+git6c8c5ad, CUDA 12.1 e lista moduli coincidono
esattamente con la summary storica 30M/5.000. `pip_freeze` è assente in entrambe:
non si dichiara quindi verificata l'identità dell'intero ambiente Python.

L'audit remoto `diagnostics.json`, prodotto dallo script pubblicato
`inspect_training_logs.py`, è coerente con i tre SHA dei log raccolti. Conferma
**5.000 righe contigue, 160.000 esposizioni di finestre**, batch globale 32,
LR previsto su tutti i passi, campi essenziali finiti, nessun STOP, fixed eval
e pannello ai passi **0,500,…,5.000**, con identificativo del pannello stabile.
Lo script è stato riletto: la review verifica output, hash e logica dell'audit,
non pretende di aver ricalcolato localmente gli interi log o il supporto del
pannello, che non sono stati trasferiti. Non usa l'ultima loss come soglia di
accettazione o come misura del trasferimento.

Prove aggregate rihashate:

- `remote_sha256.txt`: `e2d1853ad29f834ec3d22aa49842703528adbe9317c81c15d8c1f51ba54f925d`;
- `diagnostics.json`: `6f398e1a36cbb864129db9c3879d2dcada5841448c90d9393d7460b0f6c70d0c`;
- `inspect_training_logs.py`: `8b8368c9308c4f9eab50b052723f5bd5e35684f4449a36e1a79badf121b9eec8`;
- `sacct.txt`: `f78408892fce8f99ec398d721124d03a973951ce72825e2f0f3c45bea0b74bfa`.

## Checkpoint effettivi e piano di estrazione

Le **13 letture SHA remote** si riconciliano con ricevute e piano. Il named
300M `checkpoint_step0005000.pt` e il finale `checkpoint.pt` hanno entrambi SHA
`37279995d0d484e241b032eb9b6b23a31c16a5947a0e2cf0c575da0cb70641aa`,
4.673.783.740 byte ciascuno. Il controllo 30M conserva SHA
`49dd518b5eea632a16bc7ff88ab0fb013f30d61458164c9521867572c83daa80`.
L'uguaglianza dei due SHA 300M è osservata in questo run; il requisito del
wrapper resta la corretta identità di entrambi e l'uguaglianza dello stato
addestrato, non un'assunzione generale sulla serializzazione.

Piano esaminato:
`results/infra/scaling_300m_20261104/extraction_plan_5000_20261105.json`, SHA256
**`8b358951182f02dc6af8fafd2621f2dc1ae2eb98939e67c5c914ecc09933528d`**.
I pin di checkpoint/config/receipt/summary coincidono con gli hash effettivi.
Per il 30M sono usate le copie immutabili **5000 in `$WORK`**, non la summary
scratch successivamente aggiornata a 19.618. Confronto, sei input, percorsi,
riferimenti di supporto posa/classificazione sono identici al piano storico
`tokenizer_evaluation_20261103/extraction_plan_5000.json`; i riferimenti B19618
servono a verificare supporto, bersagli e ordine delle righe, non sostituiscono
il controllo scientifico T100-G-30M/5.000. Verificati contratto e cinque
primitive congelate di estrazione.

| Cella ammessa dal launcher | GPU | CPU | Memoria | Limite |
|---|---:|---:|---:|---:|
| 300M posa train + val | 1 | 8 | 100G | 150 min |
| 300M P1 | 1 | 8 | 100G | 120 min |
| 300M P2 | 1 | 8 | 100G | 150 min |
| 30M P1 | 1 | 8 | 100G | 30 min |
| 30M P2 | 1 | 8 | 100G | 45 min |

Generati in memoria e controllati i comandi delle cinque celle: checkpoint
corretto, sola estrazione teacher, nessun fit. Per la classificazione il teacher
è selezionato esplicitamente nelle funzioni di `channel_readout.py`, non tramite
un argomento CLI `--which`. Tetto confermato **8,25 GPU-ore / 66 core-ore
allocate con GPU**. Nessuna sesta cella di validazione o nuova estrazione posa
30M: si producono dodici cache nuove e si riusano due cache posa30M.

## Guardie e limiti da mantenere

I checkpoint non sono stati caricati su questo Mac. Il wrapper del training
ha già eseguito le verifiche strutturali complete dei due file e confrontato
student/teacher/optimizer. Ogni estrazione deve comunque rieseguire, sul nodo,
`load_context`/`load_teacher_endpoint`: hash, dose, firma dati, ricetta completa,
stream, RNG, struttura AdamW, finitezza, caricamento strict del teacher e
uguaglianza esatta dei suoi tensori e del guadagno. Nessuna ricalibrazione su
valutazione; controllo STOP e invariabilità prima/dopo; cache validate per
supporto, righe, bersagli, geometria e provenienza prima della receipt finale.
La review delle copie non sostituisce queste guardie e non certifica la
riprendibilità futura a 19.618 o l'associazione completa momenti/parametri.

La presenza dei nuovi pin rende concreto il piano GPU. Non inventa risultati
di benchmark CPU ancora da misurare: i fit restano subordinati alle cache
complete e ai benchmark, dentro il tetto CPU già pubblicato di 180 core-ore.
Restano invariati i criteri S5/S8. Questa attività non ha eseguito SSH, lanci,
commit o modifiche ai sorgenti; ha scritto soltanto questa review.
