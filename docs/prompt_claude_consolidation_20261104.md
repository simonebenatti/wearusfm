# Review Claude — consolidamento T100-G e primo fine-tuning EPN

Ramo `origin/codex/consolidation-20261104`, base `8ee6603`. Lavora nel tuo worktree:
non modificare quello dell'autore. Questa richiesta è solo review, nessun job,
checkout remoto, training o merge. Simone ha chiesto preparazione → tua review →
correzioni/convergenza → lancio da GPT. I piani sono bloccati in awaiting_claude_review.

Questa linea possiede SOLO consolidamento30M e fine-tuning. Il300M sta nel thread
originale; log-quadratico/contrastiva in un altro. L'estrattore comune pre/post
backbone appartiene a questa linea e deve funzionare per D384 e D896.

Leggi `docs/consolidation_protocol_20261104.md` e rivedi il diff dalla base8ee6603,
in particolare:

- `scripts/consolidation_continue.py`, launcher omonimo e continuation_plan.json:
  stato completo del checkpointG19.618, WSD49.046 invariato, segmento30.000 solo
  operativo, checkpoint39.237/49.046, zero modifiche alla sorgente e ai moduli
  scientifici originali. Dati, optimizer, RNG, world4, workers6 e ricevute.
- `src/wearusfm/harness/backbone_depth.py`: hook sull'encode effettivo, FP32,
  validità comune e post uguale alla prima metà del pool storico. Per posa si
  deve chiamare depth_readouts sugli input del loader originale; extract_windows
  ha preprocessing da classificazione e non deve essere usato per la posa.
- `src/wearusfm/harness/finetuning.py`, `scripts/consolidation_finetune.py`,
  launcher e finetune_plan.json: EPN interno4fold, train/val/held separati,
  teacher19.618 frozen/fullFT contro architettura casuale, testa LN+linear identica.
  Cinque epoche complete, due LR per regime, selezione epoca/LR solo su validation.
  La logistica storica è riferimento separato. Non dichiarare convergenza scratch.
- Test `tests/cpu/test_consolidation.py`, più regressioni WSD e tokenizer_train.

Chiedo un giudizio esplicito sulle **scelte ancora da validare**, non solo sui test:

1. La continuità di schedule e stream è davvero preservata e dimostrata dalle guardie?
2. Le due profondità e il pool storico rispondono alle domande senza confondere
   pooling, larghezza, supporto temporale e LayerNorm finale?
3. Il primo FT su EPN, cinque epoche e due LR per regime è un confronto di sviluppo
   ragionevole alla dose fissata? Quali conclusioni sono sostenibili se scratch è
   ancora in miglioramento? Proponi correzioni motivate prima di qualsiasi fit.
4. Calibrazione non supervisionata per soggetto, class-weight, split, head, RNG,
   selezione e OOF evitano leakage e vantaggi accidentali per uno dei regimi?
5. Benchmark train-only: proiezione include caricamento, preprocessing, validation,
   held e margini; tetti85% memoria e4h per due candidati. Vedi un errore nei costi
   o nella costruzione del gate? Il tetto48GPUh non è una stima osservata.

Distinzione di completezza: continuazione e FT hanno controller/launcher; la libreria
di estrazione comune è disponibile. **Gli adattatori e controller della matrice
lunga P1/P2/posa e il report aggregato con IC non sono ancora consegnati.** Richiederanno
pin effettivi dei nuovi checkpoint, budget misurati e completamento tecnico prima
delle rispettive valutazioni. Non considerarli già lanciabili e non usare valutatori
hardcoded768 su cacheD. Puoi approvare separatamente continuazione, benchmarkFT,
protocollo dei fit e protocollo di valutazione, indicando cosa resta condizionato.

Restituisci APPROVATO / DA CORREGGERE per ciascuna parte, rilievi ordinati per gravità
con file/riga e una correzione verificabile. Nessun cambiamento di perimetro o lancio.
