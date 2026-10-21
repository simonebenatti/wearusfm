# Punto 5, bracci dell'obiettivo: critica di pianificatore e valutazione P1/P2 (16/10/2026)

Richiesta di Simone (16/10): «fai una critica a quello fatto finora su punto 5, in particolare pianificatore e valutazione P1/P2. cosa c'e
e come lo giudichi ?». Revisione di Claude (sessione separata da quella che ha scritto il codice), sul ramo `worktree-claude-training-engine`
a `5c3ec80`. Riferimenti: `docs/bracci_obiettivo_finestra1.md` (firmato il 16/10), `docs/decisioni.md` (firma dei bracci, esito 2C, encoder
casuale), accordo del 15/10 e CLAUDE.md. Nessun file di codice modificato da questa revisione.

**Verdetto: DA RIVEDERE.** L'implementazione dei bracci regge; pianificatore dei bracci e piano di valutazione P1/P2 non esistono ancora
(punto 4 di «Implementazione necessaria», autorizzato il 16/10; anche la review GPT `8508b7d` li elenca fra i residui). Tre problemi da
chiudere prima del lancio dei bracci, tre minori.

## Cosa c'e'

| Pezzo | Stato | Giudizio |
|---|---|---|
| Bracci JEPA-only e ancore-only (`JEPAConfig.jepa_weight`, `--jepa-weight`, `--anchor-weight`, b1238a1 + fe67339) | fatto, default identico bit per bit | buono: `critico` APPROVATO, GPT nessun difetto (45 test) |
| Sonde P1 e P2 (`scripts/probe_p1.py`, `scripts/probe_p2.py`, `harness/`) | esistono, usate nel 2C | buone; launcher non adatti alla campagna (problema 3) |
| Orchestratore `scripts/evaluation_chain.py` (eval-plan-v0) | collaudato (`w1_valutazione_v1`, `w1_selezione_v1`) | buono, riusabile |
| Confronto accoppiato `harness.downstream_ab.compare_reports`, `harness.probe.combine_seeds` | esistono | riusabili |
| Pianificatore dei bracci, piano P1/P2, confronto con la regola firmata | **mancano** | lavoro da fare |

Test eseguiti (CPU, `src/` del worktree in testa al path): `test_objective_arms`, `test_evaluation_chain`, `test_p1_cross_subject`,
`test_probe_p2`, `test_harness_probe`, `test_wsd_chain`, `test_step2_closed_reports`: **118 PASS**.

## Problemi

1. **[BLOCCANTE] Regola di lettura piu' debole di quella del 2C, mentre il protocollo dice «come la soglia dei confronti 2A/2C».**
   Dove: `docs/bracci_obiettivo_finestra1.md`, «Disegno», e firma punto 3; `docs/decisioni.md`, firma dei bracci del 16/10.
   Prova: il 2C (decisioni.md, «Criteri congelati», primario) chiedeva «utile se delta >= +0,01, limite inferiore > 0, positivo con tutti e
   due i teacher»; la regola firmata chiede solo +1 punto sulla media dei due semi, senza dire se il confronto e' accoppiato ne' come entra
   l'errore standard. Rumore misurato nel 2C: B P2 +0,75 con IC -0,76..+2,26; allenato contro casuale vale solo +1,92 / +2,30 punti su P1
   (punto 2 chiuso, 15/10): differenze fra bracci di circa un punto con due semi possono essere solo rumore.
   Come risolverlo: decisione nuova, prima di qualunque risultato, che renda la regola uguale a quella del 2C: delta accoppiato per seme
   (`compare_reports`), limite inferiore > 0 e segno concorde nei due semi, oltre a +1 punto; registrata in `decisioni.md`.

2. **[BLOCCANTE, decide Simone] Il test ufficiale sceglie il default.**
   Dove: stesso documento: «altrimenti A resta il default» (quindi un braccio che «batte» A diventa il default).
   Prova: la scelta avviene sui soggetti di test di P1 e P2, con 3 bracci (piu' la keep) e 2 metriche; la P1/P2 riportata poi per il modello
   finale sugli stessi soggetti diventa ottimista. Per il LR l'accordo del 15/10 lo esclude («nessun test ufficiale P1/P2 e nessun benchmark P2
   nella scelta del LR») e la calibrazione usa la validation interna S.
   Come risolverlo: scegliere il default con S (`scripts/calibration_probe.py`, gia' collaudata) e usare P1/P2 come conferma; in
   alternativa dichiarare per scritto, prima dei risultati, che la P1/P2 finale ha questo limite.

3. **[BLOCCANTE] I launcher di P1 e P2 non rispettano la regola dei checkout di campagna.**
   Dove: `scripts/slurm/probe_p1.sbatch`, `probe_p1_probe.sbatch`, `probe_p2.sbatch`, `probe_p2_probe.sbatch`.
   Prova: fanno `cd "$WORK/wearusfm"` (il codice mobile) e non usano `lib_code_dir.sh`: niente `CODE_DIR`, niente verifica dello SHA;
   CLAUDE.md: «Ogni launcher riceve `CODE_DIR`, verifica lo SHA atteso e che il checkout sia pulito». La catena non li puo' usare cosi'.
   Come risolverlo: nel lavoro del piano P1/P2, adattarli come `scripts/slurm/calibration_extract.sbatch` (o launcher nuovi), con test.

4. **[MINORE] Costo della valutazione sottostimato.** Il protocollo stima «~1,5 GPU-ora per checkpoint, come 2C». Nel 2C: 26,41 GPU-ora
   per 8 checkpoint (estrazioni 8,92, sonde 17,49 sui nodi GPU), cioe' ~3,3 per checkpoint. La stima regge solo con le sonde sulla CPU
   seriale (come in C1): scriverlo nel piano e nel preventivo.

5. **[MINORE] Test dei bracci promessi col pianificatore.** Il punto 4 di «Implementazione necessaria» prevede test di pesi effettivi,
   gradienti nulli e **stesso stato iniziale fra bracci a pari seme**: quest'ultimo non c'e' in `tests/cpu/test_objective_arms.py`.
   Da fare insieme al pianificatore.

6. **[MINORE] Dipendenze di calendario.** La calibrazione della keep sul corpus intero non e' fatta (ne dipendono 4 run, ~309 GPU-ora
   attese); i bracci aspettano il LR di C1-C2 e C2 non e' partita.

I problemi 1 e 2 toccano una regola firmata: servono una decisione di Simone e la registrazione in `decisioni.md` prima dei risultati. C'e'
tempo: i bracci non partono prima della fine di C2.
