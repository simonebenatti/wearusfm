# Verifica scettica n. 1 del rilievo P1-N1 (validate_recipe accetta un config senza target_version)

## Passo 1: riproduzione indipendente (438c8d65)
Script mio: /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_targets/tmp/v1n1/scripts/probe.py
(validate_recipe(raw, "calibration") su review_configs/calibration.config.json e varianti).
Esito:
- A file pubblicato: ACCEPTED v3 100/125, canonico == atteso
- B senza target_version: ACCEPTED v3 100/125, canonico != atteso
- C senza finestre / C1 senza rms_window_ms / D target_version v1 / E 25/50 senza versione: REJECTED
- F/G senza readout_version, amp_bf16, stream_segment, accum_steps, keep_calibration, loader.pose_unprompted: ACCEPTED
Il fatto e' riprodotto. Ma non e' specifico di target_version: validate_recipe (scripts/contrastive_run.py:47-50)
confronta per costruzione la RunConfig ricostruita, quindi tollera l'assenza di QUALSIASI campo con default uguale all'atteso.

## Passo 2: tentativi di confutazione
1. Preesistente: blob di scripts/contrastive_run.py identico a e388a1de, 0efdc29c e 438c8d65 (d56d51c3...);
   RunConfig.target_version = TARGET_VERSION (run.py:114) e TARGET_VERSION v3 (spectral_targets.py:11) identici nei tre commit.
2. Contratto semantico dichiarato: docstring di contrastive_run.py:3 ("A recipe check is explicitly distinct from a complete
   release dry run"), help di --recipe-only (r. 241), protocollo docs/contrastive_comparison_protocol_20261116.md:188.
   Lo stesso confronto semantico e' in calibration_report (contrastive_prepare.py:273).
3. Requisito delle regole: CLAUDE.md e l'emendamento (r. 20) chiedono esplicite le DUE finestre: imposto (casi C, C1 rifiutati).
   target_version assente puo' dare solo v3, l'unica versione coerente con 125 ms (target_version_for_fast_window) e con la
   guardia del trainer (run.py:1246-1249); 25/50 senza versione e' rifiutato (caso E). Config storico della coorte vecchia
   (contrastive_comparison_20261116/configs/calibration.config.json, 25/50 assenti, v1): rifiutato (probe3.py).
4. Produzione: i quattro config pubblicati contengono target_version v3 esplicito e sono pinnati in
   review_configs/preparation.json (sha256 riletti: C0 ef0b3c38..., CN 5c88b285..., CR d3df3f6d..., calibration b50da400...).
   Il dispatcher legge il config solo tramite pin sha256 del job spec (contrastive_job.py:89 checked_file) e, per le fasi con
   release, config_sha256 della release (contrastive_job.py:120, contrastive_run.py:168). La verifica pre-lancio storica
   (calibration_submission_20261117/prelaunch_verification.json) comprendeva "unchanged ... configs".
5. Effetti: nessun effetto scientifico (cfg effettivo == atteso; il trainer scrive config_to_dict(cfg) completo, run.py:1496);
   tutto fallisce chiuso: audit config_sha256 dal file grezzo (contrastive_audit.py:182) contro canonico atteso
   (contrastive_run.py:99-101 alla release di calibrazione; readout 327-334); CUDA config_canonical_sha256 (cuda_check:265)
   controllato SOLO nel readout (readout:274); coorte per uguaglianza esatta (contrastive_run.py:76, 83-85).
   In fase di calibrazione un file privo del campo non sprecherebbe nulla (calibration_report semantico, config.json completo).

## Esito
Non confutato: il fatto e' riprodotto e la descrizione e' corretta. Gravita' NOTA confermata (nessun effetto scientifico,
precondizione = job spec/release rivisti che pinnano un file diverso da preparation.json).
Precisazioni: (a) non e' specifico di target_version: passa l'assenza di qualunque campo con default uguale all'atteso
(readout_version, amp_bf16, stream_segment, accum_steps, keep_calibration, loader.pose_unprompted); (b) il fix proposto
`_jsonable(raw) == expected` chiude i campi mancanti ma non le differenze di rappresentazione: rms_window_ms=100 (intero)
passa sia validate_recipe sia il fix (100 == 100.0) e produce comunque un canonico diverso, quindi ricevute audit/CUDA
rifiutate (probe2.py). Per allineare cancello d'ingresso e ricevute serve lo stesso hash canonico di contrastive_audit.py:182,
oppure un controllo meccanico che il pin del config nel job spec coincida con preparation.json; (c) per il CUDA lo spreco
emergerebbe solo al readout (dopo i benchmark, che restano validi perche' usano i file esatti della coorte).
Parte invariata dalla consegna: va fra le note non bloccanti, senza un nuovo giro di review.
