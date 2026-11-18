# Review Claude: correzione Path/mmap del collaudo contrastivo (GPT `3b5d4be`) e collaudo CUDA 64730941, 18/11/2026

Richiesta: il messaggio di GPT che Simone mi ha passato il 18/11, «Per Claude: Commit della correzione:
`3b5d4be2bda36bdc601ae3c50ae9b8ace696f16e` Diff: `f67acb0..3b5d4be` Ramo: `origin/codex/tokenizer-logcontrast-20261104`
Documento: `docs/reviews/contrastive_mmap_release_20261118.md`». Poi Simone, alla lettera: «si ovvio che se trovi qualcosa
segnalamelo»; «ovvio che non possiamo fare troppi giri di review tra voi 2 , io a un certo piunto l'esperimento lo devo lanciare»;
e, dopo il collaudo fallito, «mah , allora per non fare troppi rimpalli sistema tiu il problema, poi mi generi un prompt per cxhat
gpt cosi lo teniamo aggiornato e poi lo faccio lanciare, coswi risparmio i tuoi token per le cose piu utili».

Contesto registrato da GPT in `docs/decisioni.md`: il 17/11 Simone ha dato a GPT il via al collaudo senza la mia rilettura della
revisione del 16/11 («direi che se tu sei a posto per questo giro lanci tu, ti do fiducia»). Questa review copre la correzione
`3b5d4be` e i commit operativi `afe7cff`, `7e96014`, `b7db83a`, non la revisione `c680e95`. Nessun job lanciato da me.

## Verdetti

| Che cosa | Verdetto |
| --- | --- |
| Correzione Path/mmap (`3b5d4be`) | **Approvata.** Nessun rilievo bloccante; confermata sul cluster dal collaudo 64730941, che arriva in fondo |
| Riuso di audit e calibrazione originali (validatore di compatibilità) | **Approvato.** Il manifest dice il vero; il validatore rifiuta 111 alterazioni su 111; due punti di chiamata senza test (MINORE) |
| Release, coorte, invii e accodamento del CUDA | **Approvati.** Una sola attenzione operativa: limiti per utente della QoS di debug per i cinque benchmark (MINORE) |
| Collaudo CUDA 64730941 | **Fallito per un criterio, non per un difetto.** Corretto da me su incarico di Simone nel commit `be21629`, vedi sotto |

## Che cosa ho verificato di persona

- Digest dei sorgenti di esecuzione con la regola del validatore: `ab2d6460…` ai commit dell'audit (`f09bf85`), della calibrazione
  (`4fc5152`) e a `f67acb0`; `6b421b75…` a `3b5d4be` e `b7db83a`. Fra `f67acb0` e `3b5d4be` cambiano solo i cinque file dichiarati.
- I sei `torch.load(..., mmap=True)` di `src/` e `scripts/` passano ora una stringa; gli altri caricamenti non usano mmap.
- Il collaudo 64730941 (checkout 64730738 COMPLETED 0:0 alle 11:01; CUDA 11:01:08-11:03:15, FAILED 1:0): rapporto `cuda.json`
  completo, sha256 `3bcae089…`, letto e archiviato con log e `sacct` nel mio ramo di correzione.
- La calibrazione vera: con la regola originale del 10% i λ sarebbero stati 1,01 (InfoNCE) e 2,80 (RnC), fuori da [1e-4; 1];
  con la regola scelta da Simone il 16/11 sono 0,119 e 0,330. Il primo `grad_norm` della calibrazione, 27,7938, coincide con
  quello del 30M storico: flusso e segnali sono gli stessi.

## Il collaudo 64730941 e la correzione

Sui quattro rank, 320 confronti: 312 superati. Tutti i confronti FP32 entro 7,34e-7 (limite 1e-5); spostamenti FP32 entro
7,42e-4 (limite 0,01); in bf16 perdite, gradiente, studente e teacher superati; probe di uscita e arresti coordinati superati.
Fuori limite 8 confronti, 2 per rank: `contrast_global_grad_nce` in bf16 all'update 2, CN e CR. Il limite bf16 delle tre norme
diagnostiche era l'errore bf16-FP32 del riferimento a un processo più 1e-5: confrontava due arrotondamenti della stessa grandezza,
entrambi sotto l'unità di arrotondamento del bf16 (circa 0,39%), e falliva per caso.

Correzione, su incarico di Simone, nel ramo `claude/contrastive-bf16-20261118` partito da `b7db83a`: in bf16 le tre norme
diagnostiche hanno limite `max(regola precedente, 0,01)`, con 0,01 uguale alla tolleranza delle perdite bf16 già fissata prima.
È **post hoc** e così è scritta in `docs/decisioni.md` e in `docs/reviews/claude_contrastive_bf16_amendment_20261118.md`.
Il ramo contiene anche test di regressione, manifest rigenerato, coorte e release nuove e le evidenze del job. Due agenti
indipendenti l'hanno rivista: nessun bloccante; le correzioni minori che hanno chiesto (test più stretto, cifre sui quattro rank,
citazione letterale, ricevuta delle verifiche) sono applicate. La review della correzione spetta a GPT (`CLAUDE.md`).

## Rilievi non bloccanti sulla correzione di GPT

Da correggere quando conviene, senza un nuovo giro di review. I test non entrano nel digest: aggiungerli non cambia nulla.

| N. | Gravità | Dove | Fatto | Correzione minima |
| --- | --- | --- | --- | --- |
| C-1 | MINORE | `scripts/contrastive_run.py:94-96`; `scripts/contrastive_benchmark_readout.py:317-319` | I ganci di compatibilità sull'audit non sono coperti da test: toglierli lascia passare la suite | Due test: audit con `source_sha256` diverso e un byte cambiato devono sollevare `historical evidence compatibility` |
| C-2 | MINORE | `scripts/contrastive_prepare.py:371-379` | Nel ramo storico di `authenticate_calibration` nessun test lega la ricevuta storica al suo run: un mutante che salta il ricalcolo passa la suite | Un test con il run archiviato rietichettato al digest corrente: deve sollevare `differs from authenticated recalculation` |
| C-3 | MINORE | piano d'invio dei cinque benchmark | La QoS `boost_qos_dbg` ha limiti per utente (2-4 job, contando le altre linee): inviati in blocco, alcuni verrebbero rifiutati | Inviarli a gruppi, rileggendo prima `sacctmgr show qos boost_qos_dbg` |
| C-4 | NOTA | `releases/*/cuda.job.json` | Il pin del manifest nei job è solo documentale; il manifest è comunque verificato dal validatore | Dichiararlo o leggerlo con `checked_file` |
| C-5 | NOTA | `scripts/contrastive_benchmark_readout.py:317-319` | La lettura finale salta il manifest per un audit che dichiari il digest corrente: oggi lo ferma solo il pin del job di lettura, che non esiste ancora | Pinnare l'audit nel job di lettura |
| C-6 | NOTA | `7e96014` | La review indipendente e i ricontrolli remoti della rettifica dell'accodamento sono dichiarati senza ricevuta | Archiviare le uscite grezze |

## Rischi residui per i prossimi job

1. **Limiti bf16 ancora autoreferenziali** per gradiente, studente e teacher: nel job 64730941 il rapporto scarto/limite è stato al
   massimo 0,40, 0,43 e 0,62. Un nuovo fallimento per caso è poco probabile, non escluso. Se accade: leggere il rapporto, niente
   retry cieco. Cambiarli ora sarebbe un altro emendamento post hoc senza dato.
2. **Lettura finale, confronto continuo-ripreso** (`contrastive_benchmark_readout.py:135-137, 178-180`): un solo tensore
   dell'ottimizzatore o della coda che differisca di oltre l'1% fa sollevare un'eccezione, senza ricevuta. I due run sono job GPU
   separati e non deterministici: può succedere senza difetti. **Va deciso adesso:** dopo il nuovo CUDA il codice è congelato
   nel digest, e ogni modifica obbligherebbe a rifare il CUDA. Decisione di GPT.
3. **Checkout in coda:** la quota CPU di `lrd_all_serial` è occupata dai fit della linea 1 (un fit da 8 CPU alla volta): il
   checkout nuovo aspetterà la fine del fit in corso.

## Che cosa non ho verificato

CUDA e NCCL oltre a ciò che il job 64730941 ha eseguito; il rumore GPU fra due run uguali (rischio 2); la revisione `c680e95` e i
criteri del collaudo nel loro insieme; la lettura finale, che non ha ancora un job. Le analisi degli agenti sono nella cartella
`claude_review_contrastive_mmap_fix_20261118_evidence/`: per ciascuna dimensione un'analisi e una verifica avversaria.
