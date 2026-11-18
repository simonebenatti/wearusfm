# Collaudo CUDA contrastivo: limite bf16 delle norme diagnostiche — emendamento tecnico di Claude, 18/11/2026

Incarico di Simone, 18/11, dopo la lettura del collaudo fallito, citato alla lettera: «mah , allora per non fare troppi
rimpalli sistema tiu il problema, poi mi generi un prompt per cxhat gpt cosi lo teniamo aggiornato e poi lo faccio lanciare,
coswi risparmio i tuoi token per le cose piu utili». Per `CLAUDE.md` una correzione tecnica
affidata esplicitamente a Claude nella linea 3 resta soggetta alla review di GPT. Claude non ha lanciato né cancellato job:
deployment, checkout e invii restano a GPT, operatore della linea 3.

## Evidenza: collaudo 64730941

Checkout 64730738 COMPLETED 0:0 (11:00:30-11:01:06, 4 CPU). CUDA 64730941 FAILED 1:0 (11:01:08-11:03:15, 2 min 7 s,
4 GPU, 32 CPU), codice `3b5d4be`, torch `2.2.0a0+git6c8c5ad`, A100 64 GB, memoria GPU massima 7,78 GiB. Il rapporto
`contrastive_preflight_20261118_mmap/cuda.json` (sha256 `3bcae0892d7db23da1e6d47f5c5f96938cdac4211fbd5a177a666b98ac9836ba`)
esiste e dice `passed: false`, con `inputs_unchanged: true`. La correzione Path/mmap funziona: lo script arriva in fondo.

Ogni rank confronta il proprio calcolo distribuito con un riferimento a un processo calcolato sulla propria GPU: le cifre
sotto sono i massimi e gli intervalli sui quattro rank.

| Controllo | Esito |
| --- | --- |
| Probe di uscita intenzionale di un rank | superato |
| Arresti coordinati (copertura e numerico) su quattro rank | superati su 4 rank su 4 |
| FP32: perdite, norme diagnostiche, gradiente (limite 1e-5) | superati, scarto massimo 7,34e-7 |
| FP32: spostamento di studente e teacher (limite 0,01) | superati, massimo 7,42e-4 |
| bf16: perdite (limite fisso 0,01), gradiente, studente, teacher | superati; rapporto massimo scarto/limite 0,40, 0,43 e 0,62 |
| bf16: norma diagnostica `contrast_global_grad_nce`, update 2 | **fuori limite** in CN e CR su ciascun rank; sul rank 0: CN 1,388e-3 contro 1,061e-3, CR 1,394e-3 contro 1,159e-4 |

Sono 8 confronti su 320, 2 per rank. In FP32 la stessa norma coincide entro 7,3e-7: l'algebra distribuita è giusta.

## Diagnosi (lettura di Claude)

In bf16 il limite delle tre norme diagnostiche era `|bf16 − fp32| / |fp32|` del riferimento a un processo più `1e-5`.
Confronta quindi due errori di arrotondamento bf16 della stessa grandezza. Per `contrast_global_grad_nce`, sui quattro rank,
lo scarto fra quattro rank e un processo va da 0,047% a 0,177% e quello fra bf16 e FP32 a un processo da 0,0017% a 0,143%; per
le tre norme insieme, da 0,0002% a 0,177% e da 0,0017% a 0,349%. Quando il secondo è piccolo per caso, come in CR
all'update 2, il controllo fallisce senza alcun difetto. Entrambi gli scarti stanno sotto l'unità di arrotondamento
del bf16 (2^-8, circa 0,39%). Le tre norme sono diagnostiche: i pesi sono già calibrati e non entrano nel training; il pannello
513-576 le legge con limiti larghi ([0,005; 0,5] e [0,25; 4]).

## Regola emendata, post hoc

È scritta dopo aver visto il dato, quindi è un emendamento tecnico **post hoc**, non una regola fissata prima.
Per non sceglierla sul dato, il valore è quello già fissato prima per le perdite in bf16 (`bf16_loss_relative` = 0,01).

- `scripts/contrastive_cuda_check.py`: `CRITERIA` aggiunge `bf16_diagnostic_relative_floor: 0.01`; in bf16 il limite delle
  tre norme `contrast_global_grad_*` diventa `max(regola precedente, 0,01)`. Non è mai più stretto di prima.
- Invariati: tutti i limiti FP32, le perdite bf16, i limiti bf16 di gradiente, studente e teacher, gli arresti, il probe,
  `TECHNICAL_OVERRIDES`, il protocollo, i λ, le configurazioni, i dati e la ricetta.
- Un errore reale dei fattori distribuiti darebbe scarti dell'ordine dell'unità, e resterebbe visibile anche in FP32 a 1e-5.

## Test

`tests/cpu/test_contrastive_cuda_check.py` aggiunge un test con i valori osservati (riferimento bf16 a 1,06e-4 dall'FP32,
scarto di 1,39e-3): passa con la regola nuova e fallisce con quella vecchia (provato reintroducendola). Il test verifica anche
che uno scarto dell'1,1% su ciascuna norma diagnostica fallisca, e che il limite minimo non tocchi FP32, gradiente, studente e
teacher. I risultati della suite sono nella ricevuta `amendment_bf16_20261118/prelaunch_verification.json`.

## Riuso delle evidenze storiche e nuovi riferimenti

`scripts/contrastive_cuda_check.py` è uno dei cinque file ammessi dal validatore di compatibilità: il validatore non cambia.
Cambia il digest dei sorgenti, quindi il manifest `compatibility_20261118/manifest.json` è rigenerato con il generatore di GPT
(`build_manifest.py`), con gli stessi sorgenti storici, documenti congelati e artefatti. Audit 64652559 e calibrazione 64701482
restano quelli originali, non ripetuti; λ InfoNCE 0,11942686122572264 e RnC 0,33044627364850215 invariati. Coorte nuova
`calibrated_cohort_20261118_bf16/` con le dodici configurazioni byte-identiche e il digest nuovo; release nuova
`releases/bf16_20261118/` (CUDA e cinque benchmark) con uscite nuove in
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_preflight_20261118_bf16/`, generate da
`amendment_bf16_20261118/build_release.py` (lo stesso generatore di GPT con i riferimenti nuovi).

## Risorse

Consumato, con i due collaudi falliti e i loro checkout: **1,075556 GPU-ore / 17,297778 core-ore**. Con sei celle GPU residue
(nuovo CUDA e cinque benchmark, 4 GPU × 30 minuti, 32 CPU), due checkout da 4 CPU × 240 s e la lettura finale da 8 core-ore:
massimo **13,075556 GPU-ore / 121,831111 core-ore**, entro **14 / 129,333333**.

## Che cosa resta a GPT

Review di questo diff; avanzamento del proprio ramo fino a questo commit (senza riscrivere); riferimento di deploy, checkout
nuovo, verifica dei pin remoti; invio del solo CUDA con `releases/bf16_20261118/cuda.job.json`; lettura e pubblicazione; solo
dopo un CUDA positivo i cinque benchmark, rispettando i limiti per utente della QoS `boost_qos_dbg`. Nessun training scientifico.
