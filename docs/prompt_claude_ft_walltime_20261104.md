# Claude — verifica mirata dell'adeguamento tempi FT

La review0b2623a resta valida per ricetta e lettura scientifica. Ramo
`origin/codex/consolidation-20261104`; base del nuovo diff `5b3ac37`.
Simone autorizza esplicitamente ad alzare i limiti e chiede questa verifica.

Leggi `docs/finetune_walltime_amendment_20261104.md`, report originale e
`verified_reading.json` in `results/infra/consolidation_20261104/benchmark/`.
Benchmark63525812 COMPLETED0:0,457s, memoria valida; proiezioni singolo candidato
frozen2h13, finetune6h33, scratch6h32. Il limite originale4h blocca gli ultimi due.
Nessun fit è stato lanciato. Training lungo indipendente già accodato.

Rivedi solo l'emendamento operativo:

1. Piano, controller e launcher coerenti:4h/14.100s per frozen,
   7h/24.900s per finetune e scratch. Riserva300s; stessa regola joint/split.
2. Dati,5epoche,batch8,seed,LR,inizializzazione,validation e held invariati.
   Nessuna modifica di F-R2/V-R1 o selezione dopo aver letto risultati scientifici.
3. Policy per regime presente nel nuovo report benchmark e confrontata prima
   dei fit; limite residuo non superato nemmeno dal singolo candidato.
4. Vecchio benchmark conservato come evidenza: source_identity cambia col
   controller, quindi nuovo benchmark obbligatorio; nessuna guardia aggirata.
5. Costi: proiezione prudenziale122,34GPUh;24job attesi, tetto144GPUh/1152coreh.
   Nuovo benchmark massimo0,5GPUh, checkout separato; continuazione17dd455 intatta.

Lavora nel tuo worktree e non lanciare nulla. Restituisci un verdetto mirato
all'emendamento, con eventuali soli blocchi residui; non riaprire la griglia
scientifica già convergente. Pubblica review e registro sul tuo ramo con SHA
remoto verificato. Dopo convergenza GPT procede col benchmark e poi con i fit
che passano i gate, senza richiedere a Simone un'altra autorizzazione.
