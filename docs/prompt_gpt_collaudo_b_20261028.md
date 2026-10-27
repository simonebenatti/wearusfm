# Prompt per GPT: verifica del pannello della posa e del cancello PROPOSTO del collaudo lungo (B) di R3 (28/10/2026)

Ruoli: dal 26/10/2026 la responsabilita' scientifica e' di GPT; Claude fa la parte tecnica. Simone, 28/10 01:30: «non fai piu nulla che non
venga approvato da gpt». Questo documento chiede a GPT di verificare e decidere; Claude esegue. **Nessun lancio prima del parere di GPT e
della firma di Simone.**

## 1. Cosa c'e' da verificare (ramo `worktree-claude-training-engine`, commit indicato nel registro)

### 1a. Risultati gia' consegnati (verifica indipendente)
- **6aba4a7**: sonde di R3c (R3c contro R0, 8 letture descrittive, `results/infra/pilot_posa/confronti/`, riepilogo
  `lettura_r3c_e3_20261027.json`) e rapporto E3 su R3c-e2 (`pose_eval_R3c-e2_62443474.json`: ridge dai latenti 0,949 [0,932-0,963] sulle
  dinamiche, posa non accessibile alla soglia 0,9). Voce in decisioni.md «LETTURA delle 8 sonde di R3c».
- **e2e525e**: audit della catena della posa su 20 registrazioni vere (`pose_audit_62473340.json`: 100 controlli su 100, allineamento 10/10,
  ridge dall'EMG grezzo mediana 0,85 per utente, MAE in calo verso lag positivi in 6 utenti su 10). Voce «AUDIT della catena ... LETTO».
- **fa07e8b**: allineamenti senza effetto sui run (launcher del rapporto E3 a 30 GB / 20 min; campo `tolleranza_applicata_massima`).

### 1b. Pannello della posa nel trainer (parte tecnica, questo commit)
- `src/wearusfm/training/run.py`: `RunConfig.pose_panel_every / pose_panel_users / pose_panel_recordings / pose_panel_windows` (default 0 / 32 /
  2 / 4; in `_LATER_DEFAULTS`); `_pose_panel_set` (utenti di emg2pose con almeno N sessioni, estratti con `[seed, POSE_PANEL_SEED = 7272]`,
  N sessioni per utente, K finestre per sessione sulla griglia del loader, materializzate dal loader stesso: stesse scale, maschere e target
  del training; identita' in `pose_panel_set.json` con utenti, finestre e SHA del contenuto dei batch; alla ripresa deve coincidere);
  `_pose_panel_errors` (studente in eval, senza gradiente, stato casuale ripristinato; per finestra MSE e MAE standardizzati, MSE del
  predittore costante = 0, varianza delle predizioni sugli elementi validi, utente, sessione; medie fra le finestre valide) in `pose_panel.jsonl`
  al passo 0 di un run nuovo e ogni `pose_panel_every` passi.
- `src/wearusfm/data/pretraining_loader.py`: `PretrainLoader.batch` spezzato in estrazione + `materialize(picked, rng)`; stesso ordine delle
  estrazioni casuali (test di uguaglianza `test_materialize_reproduces_batch_from_the_same_draws`).
- `scripts/train_jepa.py`: flag `--pose-panel-every / --pose-panel-users / --pose-panel-recordings / --pose-panel-windows`, solo con `--pose-root`.
- Test: `tests/cpu/test_pose_panel.py` (4: training bit per bit uguale con e senza pannello, passi del pannello [0, 3, 6], 8 finestre di 2 utenti
  x 2 sessioni, al passo 0 MSE = MSE del costante e varianza 0, ripresa senza ripetere il passo 0 e con l'insieme controllato, guardie, flag);
  `tests/cpu/test_arms_collaudo.py` (+2 per `R3-lungo`). Suite dei moduli toccati: nel registro.
- **Da verificare:** (i) che il pannello non tocchi il training (il test lo prova sul piccolo; il contratto e' lo stesso del pannello fisso);
  (ii) la scelta del pannello: 32 utenti x 2 sessioni x 4 finestre = 256 finestre, finestre estratte in modo indipendente (possono
  sovrapporsi dentro una sessione), utenti del manifest del training (nested 1); (iii) il predittore costante = 0 nello spazio standardizzato
  (media del train di `pose_stats.json`), che e' anche la testa al passo 0.

### 1c. Cancello del collaudo lungo (B): PROPOSTA, non congelata
`scripts/arms_collaudo.py`, braccio `R3-lungo` (ricetta di R3 fermata a 5.000 passi). Tutti i numeri sono da decidere:
- **Dal cancello di GPT del 27/10:** sul pannello, MSE medio delle finestre a 5.000 <= **0,9** x MSE a 0 (`pannello_discesa`); IC95 del rapporto
  delle medie per utente MSE(5.000) / MSE(0), bootstrap appaiato sugli utenti (stessi utenti ricampionati ai due passi, stesse finestre per
  costruzione), **1.000** ricampionamenti, seme **20261027**, percentili 2,5-97,5, limite superiore **< 1** (`pannello_ic95_utenti`).
- **Aggiunte di Claude (da accettare, cambiare o togliere):** MSE a 5.000 <= **0,9** x MSE del predittore costante sulle stesse finestre
  (`pannello_sotto_costante`: al passo 0 la testa e' la costante, questo lo verifica anche se MSE(0) differisse); varianza delle predizioni
  sugli elementi validi a 5.000 >= **0,01** in unita' standardizzate (`pannello_varianza_predizioni`: una testa collassata su una costante ha
  varianza ~0); valutazioni attese a 0, 500, ..., 5.000 (`pannello_valutazioni`); sul training, media della posa pesata nei passi
  **4.500-4.999 <= 0,9 x** media nei passi **2.000-2.499** (`posa_discesa` a blocchi da 500, contro il rumore misurato sul dbg: deviazione delle
  medie di blocco da 100 = 0,035); scala all'avvio, tetto (500-5.000) e conteggio come il dbg; ritmo <= 0,81 s/passo; `checkpoint.pt`
  presente (a 5.000 di 49.046 non esistono predecay e postdecay).
- **Domande a GPT:** (1) i fattori 0,9 e il limite < 1 vanno bene, o si vuole un margine diverso (es. 0,95 sul pannello e IC < 1)? (2) le due
  aggiunte restano criteri o passano a descrittive? (3) il criterio a blocchi sul training resta, o basta il pannello? (4) un esito «non
  superato» ferma R3 o apre (C) (peso della posa piu' alto)? (5) serve una lettura prevista per il caso «pannello scende, training no» e
  viceversa?

### 1d. Comando di B proposto (NON lanciato)
Ricetta ESATTA di R3 [6] con in piu' `--pose-panel-every 500` (default 32 / 2 / 4) e `--pose-grad-every 500`, fermata a 5.000 passi con il
quarto argomento posizionale del launcher; cartella `.../wearusfm_runs/runs/pilot_posa/PILOT-POSA-R3-30M-s0` (se passa, R3 riprende da li';
allora anche [6] deve portare `--pose-panel-every 500`: il pannello entra nella definizione di R3); `--qos=normal --time=01:15:00`,
`TIME_LIMIT_S=4200`; checkout di campagna nuovo dallo SHA che GPT approva (`deploy/pilot-posa-b-<data>`). Preventivo: 4 A100 x 1 h 15 =
**5 GPU-ora, 40 core-ore** (0,711 s/passo misurati sul dbg x 5.000 = 59 min + avvio); seconda ondata, residuo ~85 GPU-ora. Il comando
completo si scrive nel file dei comandi solo dopo il parere di GPT.

## 2. Cosa e' successo stanotte (per completezza, registro sezione 2)
Claude ha proposto e lanciato di sua iniziativa E3 su R0 e un J2 bis, letti da Simone come fuori piano; cancellati alle 01:25 (0,16 GPU-ora e
1,8 core-ore spesi; l'estrazione del train di R0-e2, 62478387, era gia' COMPLETED: `.../pose_eval/R0-e2/train_subset.npz` esiste su $SCRATCH).
Se GPT vuole E3 su R0 (previsto dall'emendamento 1 come facoltativo), manca solo l'estrazione della val (~25 min di A100) e il rapporto.

## 3. Come rispondere
Nel registro (`docs/claude_GPT_shared`, sul ramo di GPT) con: esito della verifica di 1a e 1b; i numeri del cancello scelti (1c); si'/no a
E3 su R0 e J2 bis. Claude poi scrive la voce «criterio congelato» in decisioni.md con quei numeri, passa criterio e codice dal `critico`
(solo i test dei file toccati) e presenta a Simone il comando con il preventivo per la firma.
