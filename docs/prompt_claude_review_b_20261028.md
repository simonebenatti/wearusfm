# Prompt per Claude — review della preparazione di B, senza lancio

GPT ha completato la preparazione di B su richiesta di Simone. Il 28/10 Simone ha poi affidato a GPT il lancio di B e a Claude il controllo.
Consulta piano e registro per gli ID effettivi: non duplicare le sottomissioni. Leggi il ramo pubblicato
`origin/codex/r3-long-collaudo-20261028`, nel tuo worktree senza modificare quello di GPT.
Codice da rivedere: `24139ab901c052357a46a65bc20466f2eb1e8358`, dalla tua bozza `afd2e4d6aca7a417fef59efcd3df2c048fa23653`.
Il tip del ramo aggiunge protocollo, piano, comandi e registro; il codice eseguibile e' congelato allo SHA indicato.

Fai una review tecnica e scientifica della consegna, come previsto dalla revisione reciproca. **Non fare deploy e non lanciare job.**

1. Leggi `docs/collaudo_r3_b_20261028.md`, `docs/reviews/collaudo_b_preparation_20261028.md` e
   `docs/reviews/pose_scientific_assessment_20261028.md`; poi le nuove voci GPT nelle due sezioni del registro.
2. Rivedi `git diff afd2e4d 24139ab -- src/wearusfm/training/run.py scripts/arms_collaudo.py scripts/slurm/train_ddp.sbatch tests/cpu`.
   Controlla soprattutto selezione e supporto del pannello, indipendenza dal training, errori DDP, identita' alla ripresa,
   aggregazione/bootstrap e guardia di partenza da zero. Non modificare i checkout delle campagne precedenti.
3. Verifica che `results/infra/pilot_posa/piano_b_20261028.json`, `config_b_atteso_20261028.json` e `comandi_b_20261028.txt`
   concordino: nuovo run, 5.000 passi, schedule 49.046/warmup 1.962, ancore 0,2/posa 1/JEPA 0, seme 0, quattro A100.
   Il tetto PROPOSTO e' 6 GPU-ore e 48 core-ore, piu' 1/3 core-ora per checkout. L'aumento da 75 a 90 minuti e' esplicito e
   ancora da approvare; non applicare cambi di risorse o soglie senza una nuova proposta.
4. I test mirati sono gia' passati: 123 distinti, compreso DDP2. Riesegui quelli pertinenti a dubbi concreti, senza lanciare
   training sul cluster. Segnala difetti con una riproduzione e indica se bloccano B. Se non ne trovi, scrivi un verdetto chiaro.
5. Registra e pubblica review ed eventuali correzioni sul tuo ramo, senza riscrivere i commit GPT. Per correzioni che cambiano il
   codice congelato, serve un nuovo SHA verificato e l'aggiornamento coerente del piano prima della firma.

Il gate primario resta: MSE media per utente al passo 5.000 <=0,9 dell'inizio e IC95 appaiato con limite superiore <1.
Le predizioni al passo zero e l'integrita' del pannello sono controlli preliminari; trend del training e varianza sono descrittivi.
Un esito positivo non autorizza R3 completo. Nessun E3 R0, J2 bis, seed aggiuntivo, retry o job di scorta e' incluso.

Restituisci a Simone: verdetto, eventuali problemi verificati e conferma di corrispondenza fra codice/protocollo/comandi.
**Il lancio di B e' affidato a GPT dal via di Simone del 28/10; tu esegui il controllo e segnala subito eventuali difetti.**
