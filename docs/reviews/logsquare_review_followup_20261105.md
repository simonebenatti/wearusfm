# Seguito alla review Claude 49f9438

## R-1 chiuso

Il solo test `test_review_plan_freezes_selection_aggregation_and_null_candidate_pins`
ora distingue le due fasi: REVIEW_REQUIRED richiede checkpoint nulli;
APPROVED_FOR_BENCHMARK richiede il pin completo del 5.000, collegato a piano,
release e configurazione verificati per SHA e byte. In entrambe le fasi restano fissi
selezione, aggregazione e allocazione completa (0,5 GPU-ore, 8 CPU, 100G, 30 min).
Endpoint 19.618, ricevuta di inferenza e allocazioni delle estrazioni rimangono
assenti nel piano storico del benchmark. Nessun cambio di runtime o ricetta.

Verifica del file: 27 test passati. Suite richiesta: 363 test passati nei 12 file LQ
in 113,37 s fuori sandbox. Il primo tentativo nella sandbox aveva 20 fallimenti
dovuti a socket locale DDP e tabella dei processi bloccati; non è stato usato come
verdetto sul codice. La correzione separata B19618 del coordinamento ha 15 test verdi.
Evidenza: results/infra/logsquare_evaluation_20261105/execution/review_followup_validation.json.

## Note

N-1 recepito nel consuntivo parziale: 21 s del tentativo fallito e 3.970 s del 5.000,
più la futura durata effettiva del 19.618, contro lo stesso tetto di 22 GPU-ore.
Il massimo finale possibile oggi è 20,434444 GPU-ore / 163,475556 core-ore.
Non si modifica il piano di training congelato. Le altre fasi hanno budget separati.

N-2 rimane la precisazione di processo riportata da Claude: la review approva questo
rilancio; nessuna nuova regola per rilanci futuri viene dedotta. N-3 resta il limite
sulla pulizia forzata non esercitata dal percorso normale sul cluster. N-4 recepito
nelle nuove voci con spazi fra parole e numeri.

La review originale di Claude è archiviata testualmente; il suo ambito non è esteso
al benchmark o all'estrazione successivi. Questi hanno la propria review indipendente
pubblicata. Nessun job fermato, rilanciato o modificato per correggere il test.
