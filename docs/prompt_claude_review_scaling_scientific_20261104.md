# Prompt per Claude — review preparazione scientifica 300M, 04/11/2026

Fai una review della nuova consegna offline di GPT per lo screening 300M a 5.000 passi. Solo review: nessun lancio, allocazione o modifica del codice; le correzioni le implementa GPT.

Codice da revisionare: **5a7b384659ec38a3c59797fc238a3d478316ebb3**, su origin, ramo `codex/scaling-300m-20261104`.
Base del diff: `6d80e8d5fff7c1abbed9425e709cdd4aa46d8603`.
Worktree GPT: `/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-scaling-300m`.
Usa il tuo checkout separato; non modificare questa worktree.

Il collaudo già approvato resta congelato a `05d4bf0`: codice, launcher, piano e risorse sono invariati. È fermo per il guasto del login node, senza job inviati. Questa review riguarda il seguito scientifico e recepisce i tuoi commit `befab4a` e `2f3207b`.

## Cosa è stato preparato

- `scripts/scaling_train.py`: wrapper 300M5000, stessa ricetta T100-G, contratto di 103 sorgenti rispetto ad `afb644e`, guardie su collaudo, memoria di produzione, valutatori pronti, risorse e conservazione del controllo30M.
- `scripts/scaling_evaluation.py`: checkpoint teacher, configurazione/gain, validatori posa D384/D896 e classificazione pool768/pool1792, driver estrazioni e pin delle primitive storiche. Regressioni contro i validatori D384, comprese le etichette testuali EPN.
- `scripts/scaling_reading.py` e `results/infra/scaling_300m_20261104/scientific_reading_policy.json`: applicazione della regola S8 ai riepiloghi già verificati.
- `tests/cpu/test_scaling_preflight.py`: F5 e F8, picco LR esplicito e RNG dell'oracolo avanzato indipendentemente. Nessuna modifica dell'update operativo.
- Protocollo aggiornato: `docs/scaling_300m_protocol_20261104.md`. Sintesi: `docs/scaling_300m_offline_preparation_20261104.md`.

La suite integrata ha **248 test passati**; comando, hash e review interna sono in `results/infra/scaling_300m_20261104/offline_verification.json` e `docs/reviews/scaling_scientific_offline_critic_20261104.md`.

## Domande della review

1. **Identità del confronto:** il wrapper conserva dati, schedule, batch32, quattro rank, loss, frontend e codice scientifico del controllo, cambiando solo il preset e i campi operativi dichiarati? I pin delle primitive di estrazione chiudono anche quel percorso?
2. **Validità delle valutazioni:** i due preset, la dose5000, il teacher, il gain e le ricevute vengono verificati senza aggirare le guardie storiche? Target/supporti/righe/soggetti/fold restano appaiati e i test D384 sono regressioni effettive?
3. **Gate del training:** componenti mancanti o report incompleti impediscono davvero l'avvio? La conservazione del controllo copre estrazioni e confronto, non soltanto il training? Il summary30M usa lo snapshot della fase5000, non il file mobile ora a19618?
4. **F5/F8:** il test del RNG è indipendente anche nella transizione warmup→stress e conserva AdamW/EMA? I limiti dei controlli strutturali optimizer/RNG rispetto a un futuro resume19k sono dichiarati correttamente?
5. **S8:** sono coerenti segni, intervalli, soglia pratica2% sulla posa e margine1pp esclusivamente suP1/P2? EPN e posa restano primari separati. Miglioramento senza peggioramenti/perdite oltre margine propone conferma19k; due primari irrisolti propongono approfondimento della dose; qualunque primario peggiore o perditaP1/P2 oltre margine impone pausa della ricetta e nuova valutazione. È una raccomandazione, non un'autorizzazione automatica.
6. **Costi e interpretazione:** formula del seguito19k fissata ora e tetti numerici separati prima di aprire gli score5000 evitano scelte opportunistiche? La pausa della ricetta non viene confusa con una bocciatura generale dello scaling?

La tua formulazione S7 sul “pavimento di rumore” è stata rettificata: una sola coppia storica T25-L/B non stima la variabilità fra run. Nessuna soglia nuova, sottrazione o dichiarazione di indistinguibilità deriva da quella coppia.

## Perimetro incompleto, dichiarato

**Il training5000 non è ancora pronto al lancio.** Mancano misure GPU reali, lettori/aggregatori della matrice a due taglie, secondario S5 a dimensione pareggiata, benchmark, pin effettivi e review dell'intera catena. Non è stata creata una readiness fittizia. F1–F4 e F6 restano facoltativi differiti; S4 è già scritto prima del collaudo. Il controllo30M in scratch va conservato fino al confronto, con scadenza13/12 da verificare sugli artefatti.

Restituisci **APPROVATO / DA CORREGGERE per questa consegna offline**, distinguendo difetti del codice consegnato, miglioramenti facoltativi e lavoro futuro già dichiarato. Per ogni blocco indica file/riga, problema riproducibile, impatto e correzione minima. Non assumere sufficiente il verdetto del critico.

Registra review e conclusioni nel tuo worktree e nel registro, committa e pusha, comunicando lo SHA a Simone. Nessun job o modifica al codice GPT.

