# Prompt per Claude — confronto InfoNCE / Rank-N-Contrast

Simone ha autorizzato GPT a correggere la proposta RnC, implementare entrambe le loss e preparare un confronto controllato. **Questa consegna è per la tua review prima dei lanci. Nessun job, training o cambiamento della ricetta da parte tua.** Pubblica la review sul tuo ramo e aggiorna entrambe le sezioni del registro.

**Ramo:** `origin/codex/tokenizer-logcontrast-20261104`.
**Codice da rivedere:** `db5ac41`, base `924e19d`.
Il commit successivo contiene questo prompt e il registro, senza modifiche scientifiche.

```bash
git diff 924e19d db5ac41 -- src/wearusfm scripts tests/cpu docs/contrastive_comparison_protocol_20261116.md results/infra/contrastive_comparison_20261116
```

Leggi prima:

1. `docs/contrastive_comparison_protocol_20261116.md`;
2. `results/infra/contrastive_comparison_20261116/plan.json`, `configs/`, `verification.json`;
3. `docs/reviews/contrastive_comparison_critic_20261116.md`;
4. la review precedente `docs/reviews/rnc_claude_review_20261116.md` per i difetti del prototipo.

**Disegno proposto.** Tre training nuovi T100-G 30M, stesso stato iniziale, seme 0, dati/maschere, batch globale 32, quattro rank, sei worker per rank e WSD storico. C0 = ancore 0,2 + posa 1; CN aggiunge InfoNCE; CR aggiunge RnC. JEPA rimane a zero. Tutti e tre eseguono entrambi i termini e la memoria; solo il peso attivo cambia. Posa a 5.000 e 19.618, classificazione a 19.618. Nessun nuovo log-quadratico.

**Punti da verificare, con verdetti separati:**

1. **Loss e correzioni.** RnC `all`, nessuna coppia mantenuta solo come negativo; mascheratura senza sentinel finito; supporto/pareggi e distanza RMS; esclusione simmetrica dello stesso istante fisico. I duplicati restano nella popolazione contro gli altri punti: verificare che la descrizione non prometta deduplicazione completa. InfoNCE student mascherato / teacher EMA completo, proiezione separata, coda per dataset, esclusioni temporali, minimo 32 negativi e denominatore su tutte le finestre.
2. **Trainer e distribuzione.** Raccolta RnC con gradiente anche sui candidati, riduzione globale e fattori world-size; rank senza posa e lunghezze differenti; diagnostica `autograd.grad` seguita dal backward DDP e activation checkpointing. Controllare EMA, coda pre-update, checkpoint/ripresa, RNG e inattività del percorso storico senza config contrastivo. La parità CPU C0/storico è entro arrotondamento, non bit per bit; la ripresa CN è bit-identica in CPU.
3. **Peso e copertura.** Calibrazione con pesi fermi: 256 batch di riempimento, poi i primi 32 congiuntamente informativi entro altri 256; λ = 0,1 × mediana delle norme globali base/termine, range [1e−4; 1], niente clamp o selezione sulle sonde. Stessi λ nei config, cambia `arm`. Discutere il limite della testa posa iniziale a zero. Audit del loader e gate di copertura/rampa/blocchi; nessuna soglia aggiornata dopo i risultati.
4. **Confronto scientifico.** Tre coppie × E_dyn/P1/P2 = nove contrasti, bootstrap appaiato per soggetto/utente, 100.000 repliche, IC95 descrittivi e simultanei 99,4444%. Metriche aggregate corrette, pairing OOF/posa, 420 fit e convergenza, guardie di non inferiorità e nessun vincitore obbligatorio. RnC è supervisionata dalla posa già presente nella base; InfoNCE risponde a una domanda più generale. Un seme e sviluppo già osservato restano limiti espliciti.
5. **Preparazione e rilascio.** Verifica audit, lettore calibrazione, validazione config/pin e comparatore. Il riferimento di circa 50,83 GPU-ore per i tre training è storico, prima dell'overhead, non un preventivo esecutivo. `contrastive_run.py` rifiuta ancora la fase training: dopo la review vanno misurati i tre benchmark a 576 passi, collegati ricevute/risorse al validatore e preparati launcher e valutatori pinnati. Non trattare l'approvazione del codice come approvazione di quella release futura.

**Test già eseguiti:** 153 passati, 6 distribuiti deselezionati; ultimi 9 test di ingresso/audit rieseguiti e passati. Due test distribuiti storici passati separatamente non verificano il nuovo percorso. Nessuna CUDA, NCCL, calibrazione reale, estrazione o sonda.

Riesegui i test locali:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider tests/cpu/test_rank_contrast.py tests/cpu/test_contrastive_core.py tests/cpu/test_contrastive_memory.py tests/cpu/test_contrastive_runtime.py tests/cpu/test_contrastive_prepare.py tests/cpu/test_contrastive_audit.py tests/cpu/test_contrastive_run.py tests/cpu/test_contrastive_compare.py tests/cpu/test_pose_objective.py tests/cpu/test_training_run.py tests/cpu/test_training_wsd.py -k 'not ddp and not distributed'
```

**Priorità tecnica:** i sei nuovi test multiprocesso non sono stati completati perché la sandbox impedisce i socket locali; le richieste esterne sono state annullate. Se il tuo ambiente consente l'esecuzione locale, eseguili; altrimenti dichiarali ancora da verificare, senza aggirare restrizioni e senza sottomettere job:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider tests/cpu/test_contrastive_memory.py tests/cpu/test_contrastive_runtime.py -k 'ddp or distributed'
```

Per ogni rilievo riporta file/riga, caso che lo attiva, impatto e correzione minima; separa difetti dimostrati, dubbi scientifici e preferenze. Puoi aggiungere test indipendenti sul tuo ramo, ma non cambiare la ricetta o lanciare esperimenti. Concludi se approvi **il disegno**, **l'implementazione**, e quali verifiche mancano per preparare il collaudo. Il training richiederà comunque i risultati e la release successiva.
