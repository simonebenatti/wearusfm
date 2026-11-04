# Coordinamento delle review di Claude sulle tre linee di GPT — 04/11/2026

Simone, 04/11/2026: «tu farai da coordinamento delle 3 chat dei reviewers sul piani che ho fatto: rifai uno specchietto come questo coi
nomi delle 3 sottochat e lo stato». Coordinatore: la chat principale di Claude (`Tokenizer e embeddings pretraining`). Controparte di GPT:
`docs/coordinamento_linee_20261104.md` (f4c54eb, ramo `codex/coordination-20261104`).

Aggiornato alle 17:40, dopo la rinomina delle sottochat fatta da Simone («fatto cambiati i nomi, rimettimi qua la tabella agiornata»):
la numerazione qui sotto e' quella nuova. Le voci del registro e la versione delle 17:35 di questo file usavano la numerazione vecchia
(1 = consolidamento, 2 = log-quadratico, 3 = scaling): oggi 1 = scaling, 2 = consolidamento, 3 = log-quadratico e contrastiva.

| Chat di Claude | Responsabilita' esclusiva | Controparte GPT | Stato |
|---|---|---|---|
| `1- 30/300M` | Verifica del collaudo train-only 30M/300M (GPT 05d4bf0, prompt `docs/prompt_claude_review_scaling_300m_20261104.md`) | Proponi soluzione / scaling 300M (`codex/scaling-300m-20261104`) | **Consegnata**: APPROVATO senza blocchi, `docs/reviews/claude_review_scaling_300m_preflight_20261104.md` (review befab4a della chat principale + seguito F8/S8 2f3207b della sottochat, fusi in 4c4104a); GPT puo' sottomettere il collaudo (2 GPU-ore + 16 core-ore); la sottochat e' ferma dopo il seguito |
| `2- schedule lungo (consolida) 30M, 100 ms + GELU: E FINETUNING` | Review del consolidamento: continuazione del 30M T100-G (19.618 → 49.046), sonde a due profondita', benchmark e primo fine-tuning EPN (GPT bbca940, prompt `docs/prompt_claude_consolidation_20261104.md`) | schedule lungo e consolidamento (`codex/consolidation-20261104`) | **Consegnata**, verdetti per parte in `docs/reviews/claude_review_consolidation_20261104.md` (commit 63fd9c5): continuazione **APPROVATA** (C-M1 e C-M2 da registrare prima della sottomissione); benchmark FT **APPROVATO** dopo F-B1, con B-R1 scritta prima della misura; protocollo dei fit **DA CORREGGERE** (F-B1 bloccante: griglia della testa uguale nei tre regimi; F-R2 in `decisioni.md`); protocollo di valutazione **APPROVATO come disegno, CONDIZIONATO** (controller della matrice, pin dei checkpoint, preventivo misurato e V-R1 ancora da consegnare) |
| `3- log quadro e conrtrastive` | Review del ramo log-quadratico T100-LQ (GPT 729093b, prompt `docs/prompt_claude_logsquare_review_20261104.md`); poi la contrastiva, separata, quando GPT la presenta | log quadratico e contrastive loss (`codex/tokenizer-logcontrast-20261104`) | **Consegnata** la parte log-quadratica: APPROVATO CON CORREZIONI per il collaudo (3 correzioni piccole prima del lancio, 2 requisiti prima del training; il training non e' approvato), `docs/reviews/claude_review_tokenizer_logsquare_20261104.md`, commit 93d89e0; sottochat chiusa per questa parte, checkout temporaneo rimosso; la contrastiva non e' ancora stata presentata |
| Chat principale (questa) | Coordinamento dei reviewer; ha scritto la review scaling | coordinamento (`codex/coordination-20261104`) | Tre review consegnate, nessun lancio autorizzato da Claude; il passo successivo di ogni linea e' dell'owner GPT (correzioni, regole in `decisioni.md`, collaudi via semplicione); altre review solo su richiesta di Simone o di GPT |

## Regole comunicate alle sottochat (17:30)

Tre sessioni scrivono sullo stesso worktree e sullo stesso ramo `worktree-claude-training-engine`: prima di ogni edit `git status` e
`git log -1`, rilettura del registro appena prima di inserire una voce; solo il proprio file di review e la propria voce di registro, mai
`git add -A`; commit subito e push con verifica dello SHA remoto (ssh di GitHub intermittente il 04/11: riprovare, non forzare); nessuna
nuova sessione o subagent che scrive; a fine lavoro una riga al coordinatore con verdetto, SHA e file. Le regole sono state inviate alle
sottochat del consolidamento (oggi `2- schedule lungo ...`) e del log-quadratico (oggi `3- log quadro e conrtrastive`), che le hanno
applicate. La sottochat dello scaling (oggi `1- 30/300M`) ha lavorato su un ramo proprio (`claude/review-scaling-300m-20261104`, su
origin) e la sottochat del log-quadratico lo ha fuso nel ramo comune: esito unico, nessun doppio verdetto.

## Cosa autorizzano le review

Nessuna delle tre review autorizza lanci: i collaudi li sottomette l'owner GPT di ciascuna linea dopo le correzioni richieste; i training
scientifici restano subordinati a collaudo, preventivo misurato, criteri congelati e firma di Simone, come scritto in ogni review.
Per il consolidamento l'ordine scritto nella review e': F-B1 e F-M4 (F-M3 raccomandata) nel codice → B-R1 e F-R2 in `decisioni.md` →
benchmark FT (0,5 GPU-ora) → pin della ricevuta → fit; la continuazione e' approvata e GPT puo' portare `continuation_plan.json` a
`claude_review_converged` dopo aver registrato C-M1 e C-M2; le valutazioni della matrice vanno ripresentate con controller, pin,
preventivo misurato e V-R1.
