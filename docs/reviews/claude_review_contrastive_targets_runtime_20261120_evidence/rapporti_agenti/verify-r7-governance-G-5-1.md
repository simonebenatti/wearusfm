# verify-r7-governance G-5 (scettico n. 1)

Avvio: 20/11. Clone HEAD 0efa4d45. Rilievo G-5 (NOTA): ricevuta locale solo pytest; --check e test del piano tautologici.

## Progressi
- (1) CONFERMATO: validation_final/local_validation.json ha `runs` di lunghezza 1 (solo pytest, 37 argomenti, 29 file di test); contrastive_local_validation.py:44-56 esegue solo pytest.
- (2) CONFERMATO: verification.json `bash_n: {passed: true, files: [3 sbatch]}` senza comando/log/hash. Nessun test contrastivo esegue bash -n (git grep su test_contrastive_*.py + 3 test finestre: nessuna occorrenza).
- (3) CONFERMATO: nessuna ricevuta di `contrastive_targets_plan.py --check` (git grep a 0efa4d45: solo prontuario r.75 e r.153, quest'ultimo come comando per 3-launcher).
- Prontuario r.204-205 e decisioni.md:8307-8309 a fbfde797: testo identico «465 test ..., launcher validati e piani rigenerati identici. Ricevuta con comandi/hash in validation_final/».
- Attenuante: i tre sbatch (campaign_checkout, contrastive_preflight_cpu/gpu) sono blob identici a 438c8d65 (dbcad2a0, 5f89cd15, 4dc96b4a): bash -n su file invariati.
- Risposta N-5 (response r.19) promette solo la ricevuta pytest (comandi, ambiente, hash, log storici): non promette ricevute di --check o bash -n. La review 94f5094 N-5 chiedeva solo produttore, pin dei log e comando dei 121 test.
- Prontuario r.18-19: «`evaluation` resta byte-equivalente come oggetto al piano scientifico di partenza» (affermazione di fatto, non attribuita a test).
- Clone temporaneo g5v1clone @0efa4d45: `contrastive_targets_plan.py --check` rc=0 (source 40d39f56..., 17 allocazioni); `bash -n` e `/bin/bash -n` rc=0 sui tre sbatch. I fatti dichiarati sono veri.
- Manomissione riprodotta da me: `t['evaluation']['__tamper__']=1` e margine p1 0,1 -> `science_plan(t)` li conserva (True, 0.1), idempotente su t, e l'asserzione tipo r.38 resta vera.
- Nessuno script contrastivo legge `evaluation` dal piano (git grep): la sezione e' dichiarativa; contrastive_compare.py invariato.
- Legame di release (contrastive_release.py:106-134): la review lega sha256 di plan.json e preflight_plan.json, preflight lega source_sha256 e science_plan.sha256; con verify_git i byte dei piani devono essere quelli del commit rivisto. Protezione reale contro la manomissione post-review = review + hash, non --check/test.
- Mutazioni (clone g5v1clone, ripristinato dopo ognuna):
  - M1: solo plan.json con margine p1 0,1 (codifica canonica) -> `--check` rc=1 «stale generated plan: .../preflight_plan.json» (il preflight pinna sha256 del piano); test plan+guards 8 passed.
  - M2: stessa modifica + generatore senza --check (riscrive i due piani) -> `--check` rc=0, test plan+guards 8 passed; sha256 del piano b03c0bc6... -> 26295963... (diverso da delivery_pin.json e da qualunque blocco di review sui byte consegnati: validate_review_binding r.117-128 rifiuterebbe).
  - M3: generatore che modifica evaluation -> `--check` rc=1 (plan.json stale) e test r.38 FALLISCE (1 failed, 2 passed).
- Rigenerazione dalla base: plan.json di 438c8d65 + generatore -> git diff vuoto; sha256 b03c0bc6.../e4b39591... = delivery_pin. Hash canonico di evaluation identico a 20261116, 438c8d65, fbfde797 (ecdc4c52...).
- `contrastive_local_validation.py --verify` sulla ricevuta finale: {"verified": true, "all_passed": true}; sha256 ricevuta 36d1d4d4... = verification.json e delivery_pin. Nessun conftest/configurazione pytest che esegua altro.
- Registro (claude_GPT_shared, aggiunte fbfde797): «465 test CPU passati ..., ricevuta riproducibile in validation_final/»: corretto (non cita launcher/piani).

## Esito
- Parte 1 (ricevuta solo pytest mentre prontuario r.204-205 e decisioni.md:8307-8308 accostano «launcher validati e piani rigenerati identici» a «Ricevuta con comandi/hash in validation_final/»): RIPRODOTTA. Effetto nullo sulla catena (fatti veri, deterministici, sbatch invariati dalla base, --check e' il primo comando di 3-launcher, prontuario r.153). NOTA confermata.
- Parte 2 («controlli tautologici»): SOVRASTIMATA. --check non e' auto-confronto per preflight_plan.json (lega piano, sorgenti e risorse storiche) e rileva una modifica isolata del piano (M1) e una deriva del generatore (M3); il test r.38 rileva un generatore che altera evaluation (M3). Vero solo che nessuno dei due ancora evaluation alla base: una modifica coordinata piano+rigenerazione passa (M2), ma cambia l'hash vincolato da review/delivery pin. La conservazione di un campo aggiunto e' il comportamento voluto di un trasformatore, non un difetto.
- Verdetto: non confutato nel nucleo; gravita' NOTA. Riformulare la seconda meta': «--check e test r.38 verificano coerenza del generatore, non l'invarianza di evaluation rispetto alla base; questa resta affidata al diff della review e al legame degli hash».
- Pulizia: g5v1clone e file temporanei g5v1_* cancellati; copia di riferimento pulita a 0efa4d45.
