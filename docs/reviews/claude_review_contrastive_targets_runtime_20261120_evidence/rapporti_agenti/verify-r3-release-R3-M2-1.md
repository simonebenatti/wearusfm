# verify-r3-release-R3-M2-1 — scettico n. 1 sul rilievo R3-M2

Avvio: 2026-11-20. Compito: confutare R3-M2 (provenienza della review verificata solo in bind-review; check-binding, make_job, dispatcher e runtime accettano review_commit e original_review_path falsi).

## Letture iniziali
- contrastive_release.py a fbfde797 letto per intero. validate_review_binding (106-134): review_commit usato solo alla riga 109 (_full_sha, formato); original_review_path mai letto; riga 132 confronta original_review_sha256 con file_sha(copia) (autoreferenziale). verify_git=True verifica sorgenti e piani contro declared["reviewed_commit"] (dal blocco della copia), non contro review_commit.
- Commento 129-131: "The original review commit can live on another branch. Its original bytes are checked by bind-review; this copy remains pinned on the deploy branch. This is a provenance declaration, not a cryptographic author signature."
- Prontuario 123-131: bind-review controlla i byte originali nel commit della review; "L'autore/verdetto rimane responsabilita' del revisore e dell'operatore che ne legge la provenienza: non si pretende una firma crittografica d'autore." check-binding non compare nel prontuario.
- Dispatcher contrastive_job.py:69-73 e runtime contrastive_run.py:181-184 chiamano validate_review_binding (verify_git default True); il runtime confronta review_pin con package_review.review_commit (entrambi dichiarati).
- Risposta R-2 (response:11): "lega la review originale Git al commit completo rivisto ... Dispatcher e runtime rifiutano review storiche o di un altro pacchetto" (non dichiara riverifica del commit originale a runtime).
- Richiesta R-2 di 94f5094 (righe 88-106, 288): legare il pin review al pacchetto nuovo (sorgenti/piani); non chiedeva l'autenticazione del commit originale.

## Riproduzione indipendente (clone tmp/vm2c a 0efa4d45; script tmp/vm2_scripts/probe.py; uscita tmp/vm2_work/probe.out; 31,6 s)
- Hash correnti = delivery_pin (40d39f56...cde0, b03c0bc6...021a, e4b39591...6dd92).
- E1 review NON di Claude in commit ORFANO 538d6dbf (for-each-ref --contains vuoto): bind-review CLI rc 0. Release sintetica cf491d24 figlia di 0efa4d45, status pulito.
- E2 check-binding sul binding reale rc 0. E3 binding con review_commit "2"*40 e original_review_path "docs/reviews/non_esiste.md": check-binding CLI rc 0; `git cat-file -e 2...2^{commit}` rc 128. E4 binding scritto a mano senza bind-review: ACCEPTED.
- E5 cosa resta verificato: original_review_sha256 != copia -> respinto; review_commit non pieno -> respinto; blocco con reviewed_commit inesistente -> "reviewed Git artifact unavailable"; source_sha256 non corrente -> respinto.
- E6 versione superata (Ra APPROVED, figlio Rb ritirata): bind-review su Ra ACCEPTED.
- E7 correzione proposta (git show review_commit:original_review_path == copia): E7a binding falsificato E3 -> respinto; E7b review falsa in commit orfano (E1) -> ACCEPTED; E7c versione superata -> ACCEPTED.
- E8 make_job audit con binding falsificato: ACCEPTED. E9 dispatcher build_command(audit,cpu,7200): supera le righe 69-86 e si ferma su FileNotFoundError dell'input Leonardo (Mac). E10 runtime validate_release calibrazione: supera 179-184 con review_pin = review_commit = "2"*40, si ferma (voluto) sul limite di tempo; E10b review_pin diverso -> "release review commit differs" (confronto fra due valori dichiarati).
- Esito: TUTTI i fatti del rilievo riprodotti.

## Contesto che pesa sulla gravita'
- Commento contrastive_release.py:129-131 dichiara proprio cio' che il rilievo segnala: i byte originali si controllano in bind-review, dopo resta la copia pinnata; dichiarazione di provenienza, non firma. Prontuario 126-131: autore/verdetto responsabilita' di revisore e operatore che legge la provenienza.
- Il commit della review di norma NON sta nell'ascendenza della release: 94f50942 (review Claude precedente) non e' antenato di 0efa4d45 e nessun ref del clone lo contiene; GPT ne ha copiato i byte in fbfde797 (sha256 identici 01efe29f...). Il checkout di campagna e' un worktree di $WORK/wearusfm (campaign_checkout.sbatch:45), che ha solo main/deploy pubblicati: una riverifica a runtime rifiuterebbe release legittime salvo cambiare procedura.
- Contro l'unico attore capace di scrivere o alterare il binding (l'operatore che esegue bind-review e crea il commit di release) la correzione proposta non protegge (E7b, E7c). Cattura solo metadati incoerenti (E7a) che non cambiano cosa viene eseguito ne' i cancelli (blocco APPROVED unico con hash correnti, git dei sorgenti/piani rivisti, pin della copia: verificati a dispatcher e runtime).
- check-binding non compare nel prontuario (solo --check, --help, bind-review --help, make-job --help): non e' un passo dichiarato di verifica della provenienza.

## Altre verifiche
- review_commit e original_review_path non entrano in alcuna uscita: in scripts/contrastive_*.py (escluso release) compaiono solo in job.py:70-73,141 e run.py:179-184. Nessun esito attribuito a quel campo.
- Il dispatcher esige il job JSON dentro il checkout congelato (contrastive_job.py:179-180): il binding usato e' quello committato nel commit di release, leggibile prima del deploy.
- pytest tests/cpu/test_contrastive_release.py sulla copia di riferimento: 22 passed in 5,73 s. Nessun test afferma che check-binding/dispatcher/runtime riverifichino il commit originale.
- Nessun oggetto di prova finito nella copia di riferimento o nel repo canonico (cat-file -e fallisce). Copia di riferimento: HEAD 0efa4d45, status pulito. Clone vm2c cancellato; conservati tmp/vm2_scripts/probe.py, tmp/vm2_work/probe.out, tmp/vm2_work/binding_forged.json.

## Esito
- Fatti del rilievo: tutti riprodotti.
- Come difetto MINORE: CONFUTATO. E' il disegno dichiarato (codice 129-131, prontuario 126-131), necessario perche' il commit della review sta su un altro ramo e di norma manca su Leonardo. La correzione proposta non respinge review false o superate (E7b, E7c). Cattura solo metadati incoerenti, che non cambiano esecuzione, cancelli o attribuzione.
- Gravita' residua: NOTA. I messaggi "authenticated" (righe 108 e 133) promettono piu' di quanto si verifica. Come comodita' per chi rilegge la release sul Mac, check-binding potrebbe rieseguire git_bytes(review_commit, original_review_path) quando il commit c'e'. Il prontuario potrebbe dire che chi legge la provenienza controlla che review_commit sia la punta del ramo di Claude.
