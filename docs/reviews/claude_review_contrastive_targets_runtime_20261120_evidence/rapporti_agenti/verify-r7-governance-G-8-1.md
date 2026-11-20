# verify-r7-governance-G-8-1 (scettico n. 1 su G-8, NOTA)

Avvio 20/11. Copia a 0efa4d45, codice letto a fbfde797.

## Progressi 1 (letture)
- Prontuario fbfde797 r.159-161: «L'operatore usa il binding ... costruire il job audit ... Poi committa/pusha metadati, verifica SHA, pubblica il deploy»; r.7-8: 3-launcher «unico operatore per deployment, invii, stato e raccolta»; r.178-179: «I comandi definitivi ... vengono emessi nella consegna post-review».
- Prompt Claude r.58: «Dopo la convergenza GPT prepara per 3-launcher la prima release eseguibile, limitata all'audit». decisioni.md fbfde797: «Dopo review si prepara la release audit» (impersonale).
- CLAUDE.md fbfde797 r.43-50 (testo d9020c20): GPT scientifico «prepara codice, protocolli, criteri e pacchetti di esecuzione»; 3-launcher «gestisce i lanci e i controlli operativi»; «La regola vale anche quando le due chat leggono lo stesso worktree».
- Prassi storica (prompt 3-launcher 18/11, r.8-13, 33-43): GPT pubblica job JSON/piano/comandi; 3-launcher esegue deploy_command, invia comandi esatti, committa le evidenze «sul tuo ramo», «Non modificare il worktree dello scientifico; usa il tuo checkout».
- Autori: 438c8d65, fbfde797, 0efa4d45, 94f50942, faa4a8d7, d9020c20 tutti «Simone Benatti» (git log).

## Progressi 2 (riproduzione mia, clone temporaneo g8s1clone a 0efa4d45)
- Script /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/g8s1_scripts/e2e_binding.py, output in e2e_binding.out (stessa cartella). Review di prova committate solo nel clone, con autore «GPT».
  - source_digest, plan e preflight attuali = delivery_pin: True/True/True.
  - A blocco reale unico (reviewed_commit fbfde797b486..., tre hash del delivery_pin) -> bind_review ACCETTATA (anche con autore «GPT»: provenienza non vincolata, limite dichiarato prontuario r.130-131 e release.py:129-131).
  - B blocco reale + modello del prompt in ````text -> RIFIUTATA «exactly one» (release.py:91-93): fallimento sicuro.
  - C solo modello del prompt -> RIFIUTATA «complete SHA» (release.py:98-99).
  - D blocco con «scope» -> RIFIUTATA (release.py:95-97).
  - make_job con il binding A: audit -> APPROVED_FOR_PREFLIGHT; TUTTE le 10 celle non audit RIFIUTATE (evidenza vuota: KeyError 'audit'; audit senza accounting: «audit accounting required before release», release.py:240-242).
- Rieseguiti test r7 + tests/cpu/test_contrastive_release.py sulla copia: 26 passed in 5,67 s; copia pulita.

## Valutazione
- Confermato (NOTA): prontuario r.159-161 assegna all'operatore (=3-launcher, r.7-8 e r.175) costruzione del job audit e commit/push dei metadati; prompt r.58 assegna a GPT «la prima release eseguibile»; r.178-179 «comandi definitivi ... nella consegna post-review» (termine delle consegne GPT; gli SHA del commit di release si conoscono solo dopo averlo creato). Prassi 18/11 (contrastive_readout_release_20261118.md: «GPT scientifico pubblica piani e riferimenti, 3-launcher sottomette») coerente col prompt, non con r.159-161. Il release APPROVED_FOR_PREPARATION delle fasi successive non ha builder (solo consumatori job.py:134, run.py:168): anche li' autore non nominato.
- Confutato in parte: «Prima release solo audit resta solo procedurale» e' falso nel meccanismo: ogni cella non audit richiede accounting di un audit nuovo COMPLETED 0:0 <=7200 s in make_job (release.py:240-242), dispatcher (job.py:80-81) e runtime (run.py:188-195); training fuori dai modi di make_job (r.227) e con APPROVED_FOR_TRAINING + pin di decisione/budget (run.py:160-161,168). L'assenza di ambito nel blocco e' voluta: il prompt chiede un'unica review del pacchetto e «Calibrazione e seguito richiedono le rispettive ricevute» (r.58-60).
- «Due blocchi» e «autore/ramo» sono veri ma gia' dichiarati dal rilievo come fallimento sicuro e limite dichiarato: non difetti.
- Rischio «due writer» attenuato: d9020c20 dice «La regola vale anche quando le due chat leggono lo stesso worktree» (condizionale, non constatazione); ID logici fissi nel preflight e submitter che non risottomette un ID (submit_job.py:12), write_new in modo «x» (release.py:246-251), checkout che verifica HEAD == CAMPAIGN_SHA: nessuna doppia esecuzione possibile, al piu' confusione di tracciabilita'.
- Esito: NON confutato nel nucleo documentale; gravita' NOTA corretta (non MINORE/BLOCCANTE). Fix sensato: nominare un solo autore dei commit di release (e dei release APPROVED_FOR_PREPARATION); review di Claude con un solo blocco reale (variante A verificata accettata).
- Pulizia: g8s1clone e g8s1_pt cancellati; g8s1_scripts conservato come prova.
