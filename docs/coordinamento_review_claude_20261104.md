# Coordinamento delle review di Claude sulle tre linee di GPT — 04/11/2026

Simone, 04/11/2026: «tu farai da coordinamento delle 3 chat dei reviewers sul piani che ho fatto: rifai uno specchietto come questo coi
nomi delle 3 sottochat e lo stato». Coordinatore: la chat principale di Claude (`Tokenizer e embeddings pretraining`). Controparte di GPT:
`docs/coordinamento_linee_20261104.md` (f4c54eb, ramo `codex/coordination-20261104`). Aggiornato alle 17:35.

| Chat di Claude | Responsabilita' esclusiva | Controparte GPT | Stato |
|---|---|---|---|
| `1- 30M, 100 ms + GELU: E FINETUNING` | Review del consolidamento: schedule lungo del 30M T100-G (19.618 → 49.046), sonde a due momenti/livelli, fine-tuning EPN (GPT bbca940, prompt `docs/prompt_claude_consolidation_20261104.md`) | schedule lungo e consolidamento (`codex/consolidation-20261104`) | **In corso**: checkout temporaneo di bbca940; nessuna review committata |
| `2- controlla log quadro` | Review del ramo log-quadratico T100-LQ (GPT 729093b, prompt `docs/prompt_claude_logsquare_review_20261104.md`); poi la contrastiva, separata | log quadratico e contrastive loss (`codex/tokenizer-logcontrast-20261104`) | **Consegnata**: APPROVATO CON CORREZIONI per il collaudo (3 correzioni piccole prima del lancio, 2 requisiti prima del training; il training non e' approvato), `docs/reviews/claude_review_tokenizer_logsquare_20261104.md`, commit 93d89e0 |
| `3- Review collaudo scaling 30M/300M (GPT 05d4bf0)` | Verifica del collaudo train-only 30M/300M (GPT 05d4bf0, prompt `docs/prompt_claude_review_scaling_300m_20261104.md`) | Proponi soluzione / scaling 300M (`codex/scaling-300m-20261104`) | **Consegnata**: APPROVATO senza blocchi (review befab4a della chat principale + seguito F8/S8 2f3207b della chat 3, fusi in 4c4104a); GPT puo' sottomettere il collaudo |
| Chat principale (questa) | Coordinamento dei reviewer; ha scritto la review scaling | coordinamento (`codex/coordination-20261104`) | Regole del ramo condiviso inviate alle chat 1 e 2; in attesa del verdetto della chat 1 |

## Regole comunicate alle sottochat (17:30)

Tre sessioni scrivono sullo stesso worktree e sullo stesso ramo `worktree-claude-training-engine`: prima di ogni edit `git status` e
`git log -1`, rilettura del registro appena prima di inserire una voce; solo il proprio file di review e la propria voce di registro, mai
`git add -A`; commit subito e push con verifica dello SHA remoto (ssh di GitHub intermittente il 04/11: riprovare, non forzare); nessuna
nuova sessione o subagent che scrive; a fine lavoro una riga al coordinatore con verdetto, SHA e file. La chat 3 ha lavorato su un ramo
proprio (`claude/review-scaling-300m-20261104`, su origin) e la chat 2 lo ha fuso nel ramo comune: esito unico, nessun doppio verdetto.

## Cosa autorizzano le review

Nessuna delle tre review autorizza lanci: i collaudi li sottomette l'owner GPT di ciascuna linea dopo le correzioni richieste; i training
scientifici restano subordinati a collaudo, preventivo misurato, criteri congelati e firma di Simone, come scritto in ogni review.
