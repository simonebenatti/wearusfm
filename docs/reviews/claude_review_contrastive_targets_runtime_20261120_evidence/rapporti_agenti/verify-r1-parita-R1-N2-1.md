# Verifica scettica n. 1 del rilievo R1-N2 (r1-parita)

Stato: in corso. Copia a 0efa4d45, codice rivisto fbfde797.

## Lettura del codice (fatta)
- `scripts/contrastive_target_parity.py:34-35` PAIRS definito; unico uso a :315 (`comparison_pairs` nel rapporto). `git grep PAIRS` a fbfde797: nessun altro uso nel modulo (il PAIRS di `contrastive_compare.py:27` e' un altro oggetto, C0/CN/CR downstream).
- `assess_point` (:155-222): raccoglie in un'unica lista `hashes` i 24 digest (6 casi x 4 rank) e a :217-218 emette un solo messaggio generico; ritorna `unique_gradient_hashes` senza caso/rank.
- Readout `scripts/contrastive_benchmark_readout.py:515-521`: failure_reasons generico, MA :516 incorpora nel risultato anche `report: parity` intero (per rank e caso: gradient_sha256, gradient_norm, loss). L'attribuzione e' quindi ricostruibile dall'uscita senza nuove misure; l'ampiezza no (gradienti non salvati).
- Diagnostica storica `scripts/diagnostics/contrastive_common_state_cuda.py:41-49,318-319`: stesse sette coppie (COMPARE) con `gradient_delta` per coppia; il nuovo cancello conserva solo l'elenco.
- Piano `plan.json` /numerical_amendment/neutrality/criterion: "reduced gradients byte-identical within and between arms and ranks"; prontuario :38-41 idem; decisione faa4a8d7 punto 1 idem. Nessun documento promette descrittivi per coppia o scarti per la parita' (a differenza della ripresa, prontuario :68).

## Riproduzione (fatta, comandi miei)
Ambiente: PYTHONPATH=clone/src:clone/tests/cpu:clone/scripts, venv wearusfm-tok, -B, thread a 1.
1. `rev_runtime/tmp/verify_r1n2_1/repro.py` (evidenza sintetica come `case()/point()` dei test), su `assess_point` a fbfde797:
   - modi di guasto: solo C0b diverso (ripetibilita' C0), solo CNa+CNb diversi fra loro uguali (neutralita' CN), solo rank 2 diverso (riduzione fra rank), misto CRb/CNa;
   - esito: tutti `FAILED`, `errors == ["reduced gradients are not byte-identical within/between arms/ranks"]`, `within_and_between_byte_identical: false`; cambia solo il numero di hash unici (2 o 3). Stampa: `IDENTICAL_GATE_OUTPUT_ACROSS_FAILURE_MODES: True`, `PAIRS_USED_IN_ASSESS_POINT: False` (co_names di assess_point senza PAIRS).
   - la scomposizione per coppia ricostruita a mano dagli stessi dati distingue i quattro modi (es. C0a/C0b false nel primo, C0a/CNa e C0b/CNb false nel secondo, accordo fra rank false nel terzo).
2. `rev_runtime/tmp/verify_r1n2_1/test_r1n2_readout.py` con `readout_fixture` dei test del readout: per "within_C0b" e "between_CN" `read_preflight` da' lo stesso `FAILED_REQUIRES_REVIEW`, la stessa failure_reason `production zero-weight gradients/invariants not byte-identical`, gli stessi errori e le stesse chiavi dell'assessment (errors, inconclusive, status, unique_gradient_hashes, within_and_between_byte_identical). 2 passed.
3. Sanita': `tests/cpu/test_contrastive_target_parity.py` 29 passed in 4,39 s; la copia resta pulita (`git status --porcelain` vuoto).

## Tentativi di confutazione
- Controllo altrove: nessuno. Unici consumatori del rapporto sono il readout (:503-521) e l'osservatore (:298, :319); nessuno usa PAIRS.
- Percorso non di produzione: no, e' il cancello decisivo della neutralita' (piano, prontuario :38-41, readout :515-521).
- Dichiarazione esplicita: il criterio dichiarato e' solo l'identita' per byte "within and between arms and ranks"; nessun documento promette attribuzione o scarti per la parita'. Questo esclude un difetto rispetto alla specifica (l'esito e' corretto), non il rilievo di chiarezza: il campo `comparison_pairs` nel rapporto suggerisce confronti per coppia che il cancello non fa.
- Attenuazione parziale: il readout incorpora il rapporto intero (`'report': parity`, readout :516) con digest, `gradient_norm` e loss per caso e rank, quindi l'attribuzione si ricostruisce offline in pochi minuti senza nuovi job. L'ampiezza no: i gradienti non sono salvati; |norma_a - norma_b| e' solo un limite inferiore di ||a-b|| (nella diagnostica storica lo scarto delle norme, max 0,000557%, era circa 50 volte sotto lo scarto vettoriale ~0,03%).
- Gravita': l'esito del cancello non cambia (qualunque disuguaglianza con evidenza completa fallisce e riporta a Simone). NOTA confermata; non MINORE perche' nessuna specifica richiede la scomposizione.

## Esito
refuted = false; gravita' NOTA. Rilievo riprodotto; unica imprecisione: "il readout dice solo ... ed elenca gli hash unici" omette che il readout incorpora anche il rapporto completo per caso e rank (readout :516).
Non verificato: costo di memoria reale della correzione proposta su A100 (stima: una copia CPU in fp32 del gradiente, ~4 byte per parametro); comportamento su CUDA/NCCL.
