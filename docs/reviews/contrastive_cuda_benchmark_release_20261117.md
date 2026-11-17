# Calibrazione letta; rilascio CUDA e benchmark condizionati — 17/11/2026

Simone autorizza esplicitamente la lettura della calibrazione e, se superata, la prosecuzione con
test CUDA e benchmark già previsti. **Calibrazione superata.** GPT rilascia la cella CUDA;
i cinque benchmark sono autorizzati sotto la condizione sospensiva del suo esito positivo.
Nessun training scientifico a 5.000 o 19.618 passi è rilasciato qui.

## Evidenza della calibrazione

Job **64701482 COMPLETED 0:0**, 774 secondi, 4 GPU e 32 CPU: **0,86 GPU-ore / 6,88 core-ore**,
entro 2 / 16. 512 batch completi, 4 rank × 6 worker, zero update; fine `max_steps`.
Hash iniziale, studente finale e teacher finale identici:
`ba8a86579992a0eeb2a8072c67eef30b699465ad091671b50123c0073173fa2f`.

Ricetta e sorgenti sono quelli congelati. Metadata raccolti byte per byte e verificati contro
gli SHA remoti. Checkpoint su WORK, 234.339.066 byte, SHA256
`0756921a0e59c0d93cf5821793a8457f6becee7e55a7b68ea5ab1ec80e17d50a`.
Il collaudo CUDA ricontrollerà il checkpoint, incluse assenza di stato dell'ottimizzatore e
identità dei pesi, prima dei confronti.

Ricevuta ricalcolata da `contrastive_prepare.py`: **PASSED_FOR_REVIEW**, nessun motivo di fallimento.
Primi 32 batch congiuntamente informativi: indici 256–287, passi registrati **257–288**.
Fattore storico invariato 0,1179564123141922. Coefficienti fissati per tutti i bracci:

| Termine | λ | Gate [0,0001; 1] |
|---|---:|---|
| InfoNCE | 0,11942686122572264 | Superato |
| Rank-N-Contrast | 0,33044627364850215 | Superato |

Copertura: InfoNCE 8.179 / 8.192 = 99,8413%; RnC attivo in 256 / 256 batch.
CapgMyo 13 query e zero ammissibili: esenzione dal solo requisito per dataset, denominatore globale
conservato. Conteggi uguali al pannello dell'audit. Ricalcolo indipendente di `audit_timeout_review`
concorde; nessuna nuova review viene attribuita a Claude.

La coorte unica è generata dalla ricevuta autenticata. Restano invariati dati, inizializzazione,
ricetta, dose, loss e criteri già pubblicati. Il riferimento storico dei primi otto passi,
con componenti delle ancore e norma del gradiente, è riportato nel readout JSON come controllo
descrittivo: dal secondo passo lo storico aggiorna i pesi, questa calibrazione li mantiene fermi.
I λ sono coefficienti di bilanciamento, non prove di utilità delle loss sul downstream.

## Rilascio della cella CUDA

`CONTRAST-CUDA-v2-20261116`: 4 GPU, 32 CPU, 240 GiB, boost_qos_dbg, 30 minuti,
tetto 2 GPU-ore / 16 core-ore. Prova di uscita intenzionale di un rank e scomparsa dei processi;
poi oracolo globale/distribuito su CN e CR, FP32/bf16, due update, rank senza posa e STOP coordinati.
Fixture tecnica e tolleranze restano quelle congelate in `contrastive_cuda_check.py`.
Output: `$WORK/wearusfm_runs/results/contrastive_preflight_20261116_v2/cuda.json`, con
`cuda.json.rank_exit_probe/` e `cuda/` per le evidenze correlate. Tutti i percorsi devono essere nuovi.

Il job usa la configurazione originale della calibrazione e il checkpoint fisso pinnato; i pesi
unitari della fixture sono override tecnici dichiarati, non sostituiscono i λ misurati dei benchmark.

## Cinque benchmark: rilascio condizionato, nessun anticipo

Tre completi a 576 update: `CONTRAST-BENCH-C0-v2-20261116`, `CONTRAST-BENCH-CN-v2-20261116`,
`CONTRAST-BENCH-CR-v2-20261116`. Due riprese 544 + 32 con nuovo torchrun nello stesso job:
`CONTRAST-RESUME-CN-v2-20261116`, `CONTRAST-RESUME-CR-v2-20261116`.
Per ciascuno: 4 GPU, 32 CPU, 240 GiB, 30 minuti, tetto 2 GPU-ore / 16 core-ore.
Output nuovi `benchmark/C0`, `benchmark/CN`, `benchmark/CR`, `resume/CN`, `resume/CR` sotto la campagna.

Prima di sottomettere questi cinque job GPT deve raccogliere e pubblicare la lettura di CUDA:
COMPLETED 0:0, `passed=true`, tutti e quattro i rank, criteri/override/pin coerenti, probe di uscita
valido, STOP e confronti numerici superati. Il lettore `_cuda_evidence` e la verifica del probe
devono accettare le evidenze. Nessun solo exit code o afterok sostituisce questa lettura.
Se un gate fallisce, **nessuno dei cinque benchmark viene sottomesso**. Nessun retry automatico.

Le release e i job possono essere preparati nello stesso checkout della coorte; la condizione
sospensiva è verificata dall'operatore prima degli invii, non da un cambio di stato del runtime.
Le prove ripartono dalla stessa inizializzazione, senza riprendere ottimizzatore, RNG o coda della
calibrazione. Si mantiene il vincolo dello stato iniziale nella coorte autenticata.

## Risorse e congelamento

Consumato prima del nuovo checkout: audit e tre checkout **8,578888889 core-ore**, calibrazione
**6,88 core-ore / 0,86 GPU-ore**. Restano al massimo sei celle GPU, due checkout da 4 CPU × 240 s
e la lettura CPU da 8 core-ore. Totale massimo con consuntivi: **119,992222222 core-ore /
12,86 GPU-ore**, entro 129,333333333 / 14.

Nuovo checkout `contrastive-v2-20261117-benchmarks`, 4 CPU, 2 GiB, 240 s; codice/pubblicazione
con SHA verificato, output fuori dal checkout, submitter e ledger comuni. Sorgenti congelati
`ab2d6460bfe090e4adba8cfd480b726077005c4ed6ae10db7ebbd5909ddf808b`; `plan.json` invariato.
Non modificare sorgenti o ricevuta dopo questa calibrazione. Nessun monitor o automazione.
