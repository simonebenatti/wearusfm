# Lettori e controllo della memoria di produzione — 4 novembre 2026

Il collaudo 30M/300M `63526191` è completo e verificato. La review Claude della
preparazione offline è approvata nel commit `5e4eb8a`. Questa consegna completa
i lettori della fase a 5.000 e prepara la misura della memoria sul trainer intero.
Non contiene risultati scientifici del 300M né dichiara già pronto il lancio a 5.000.

## Codice nuovo

| Componente | Cosa verifica o calcola | Limite |
|---|---|---|
| `scaling_projection.py` | Due matrici fisse 896×384, QR con segno deterministico, manifest e hash; compressione prima di LN/scaler | Una sola proiezione, indipendente dai dati; non dimostra equivalenza informativa |
| `scaling_pose_readout.py` | Ridge teacher su 40 utenti train e 15 val; tre viste, crop e alpha storici, statistiche appaiate | Il controllo compresso è secondario; nessun tuning sulla val |
| `scaling_probe.py` | P1/P2, griglia C e split storici, audit della convergenza, OOF anche con etichette testuali | Benchmark sul train prima dei fit; nessuno score di validazione nel benchmark |
| `scaling_compare.py` | Rivalida ricevute, shard, cache, coefficienti, predizioni, supporti e runtime; ricalcola contrasti e IC | Un seme di pretraining; convergenza irrisolta impedisce la raccomandazione S8 |
| `scaling_production_memory.py` | 500 passi da zero con il trainer immutato, picchi CUDA di tutti i rank, pannelli e diagnostiche realmente eseguiti | Collaudo tecnico; il checkpoint non si usa come inizializzazione scientifica |

I primari restano 300M naturale meno 30M naturale, entrambi T100-G a 5.000.
S5 aggiunge 300M compresso meno 30M e 300M naturale meno compresso, sul medesimo
supporto. Non sostituisce il primario se dà un risultato più favorevole.
Il controllo 30M si fitta una sola volta e si riusa nei due contrasti.

Le matrici S5 sono in `results/infra/scaling_300m_20261104/projection_s5/`:
manifest SHA256 `74d1a0f2453592cb134ab104247d05a9a43b2960e43417bc23c31426fa1b9b3d`.
Sono numeri casuali generati con la ricetta fissata, non dati wearable. Nessun
checkpoint, segnale o predizione di soggetti viene aggiunto a Git.

## Memoria: prova S3

Il passo 500 è la prima ripetizione dei pannelli con AdamW e prefetch residenti,
insieme alla diagnostica dei gradienti. Un collaudo di soli 64 o 128 passi non
coprirebbe quella fase. Si usa lo stesso trainer, senza anticipare le cadenze,
spegnere diagnostiche, cambiare batch o riprendere il checkpoint tecnico.

Dal collaudo reale: tempo prudenziale per passo 3,035413 s. La formula
`ceil_900(500 × t × 1,20 + 1200)` dà **3.600 secondi**.

| Invio | Allocazione massima | Tetto |
|---|---|---|
| Checkout S3 | 2 CPU, 4G, 10 minuti | 0 GPU-ore; 0,333334 core-ore |
| S3 produzione | 1 nodo, 4 A100, 32 CPU, 480G, 60 minuti | 4 GPU-ore; 32 core-ore |

Il piano concreto è `results/infra/scaling_300m_20261104/production_memory_plan.json`.
Launcher: `scripts/slurm/scaling_production_memory.sbatch`; submitter e ledger
comuni. Il codice viene congelato e pubblicato prima dell'invio. Checkout e output
devono essere nuovi. Prima del job GPU si verificano completamento del checkout,
SHA e pulizia; la verifica sincrona evita la dipendenza SLURM scaduta con MinJobAge
di 30 secondi già osservata. Nessun retry cieco e nessuna continuazione automatica.

Il report completo si scrive solo dopo che le prove su disco verificano 500
update, pannelli ai passi 0/500, diagnostiche al 500, quattro misure CUDA coerenti
col massimo del summary e picchi sotto il 90% della memoria effettiva. Il runtime
va confrontato col summary storico per i campi realmente disponibili; cuDNN,
assente dal vecchio summary, non può essere dichiarato uguale per deduzione.

La review interna ha trovato e fatto correggere una fixture sbagliata:
`fixed_eval_set.windows` è piatto, 256 righe di quattro valori, non 8 liste da 32.
È un difetto del nuovo validatore, individuato prima dell'uso; nessun job coinvolto.

## Review Claude recepita e lavoro restante

F-c: caso positivo della readiness e casi negativi sui campi richiesti; i pin
comprendono anche proiezione S5 e lettura S8. F-d: delta della posa calcolato dagli
stessi punti a precisione piena nel JSON. F-a/F-b: runtime e codice nelle prove S3.
F-e/F-f: i validatori di cache/checkpoint vanno eseguiti su nodi di calcolo; il
preventivo CPU deve contare D896 e il caricamento dei checkpoint. Nessuna esecuzione
pesante sul login node. F-g resta un miglioramento facoltativo separato.

S8 mantiene soglie e tabella. Precisati nel protocollo: la pausa considera entità
e dose; la tranche più lunga non misura variabilità fra pretraining; nessuna
probabilità congiunta di falsa pausa viene inventata.

Prima del training scientifico a 5.000 mancano ancora il risultato reale S3,
i piani/launcher operativi di estrazione e lettura con risorse, la verifica del
controllo 30M e della retention, e la review della catena legata ai pin finali.
I benchmark delle letture precedono i fit CPU. La formula già fissata sul
collaudo dà un tetto training di 19.800 s, **22 GPU-ore**, distinto dalle estrazioni
e dai fit: non è ancora una sottomissione. Nessuna richiesta di un nuovo consenso
per la prova già approvata da Simone.
