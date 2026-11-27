Consegna del lancio su Leonardo della prova mirata InfoNCE (linea 3): un checkout CPU e un job da 4 GPU per 15 minuti, già approvati e preparati; la sequenza con i comandi esatti è in un file del repository.

Ti scrive la chat Claude «3- log quadro e conrtrastive», revisore della linea 3. Questo messaggio viene da un'altra chat, non da Simone: se vuoi una conferma diretta, chiedila a Simone nella tua chat prima di inviare qualunque cosa.

CHI LO HA DECISO
Simone mi ha scritto il 27/11: «in realta farai tu il lancio. Anxi quando è tutto pronto, passa alla chat 0-leggi le tabelline e lei si occupera di fare il lancio su leonardo». La prova l'ha scelta Simone il 21/11 («Prova mirata»); il tetto della coorte (18 GPU-ore e 169,6 ore locali) l'ha firmato il 20/11.

CHE COSA È GIÀ PRONTO
- Review del revisore: diagnostico approvato, commit 76f5d0e7cface6f02bb24bf572ec57b74171a162 sul ramo origin/worktree-claude-training-engine.
- Commit di release, già su origin: b58a66f695c8744dfa5fe633829cafe13a5a1fc4, ramo claude/cn-restart545-release-20261127. È il codice di GPT 8c72655c più il solo file di binding.
- Worktree locale pulito su quel commit: /Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release. Usalo come cartella da cui lanciare i due invii. Non modificarlo.
- Sequenza operativa con tutti i comandi: /Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine/docs/contrastive_cn_restart_probe_launch_20261127.md (ramo worktree-claude-training-engine, commit @SEQ_SHA@). Leggila per intero prima di cominciare: è la fonte dei comandi.

CHE COSA DEVI FARE, IN ORDINE
1. Controlli sul Mac: worktree di release su b58a66f6… e pulito; nessuna riga con CN-RESTART545 nel ledger comune.
2. Push del riferimento di deploy su Leonardo: git push leonardo b58a66f695c8744dfa5fe633829cafe13a5a1fc4:refs/heads/deploy/cn-restart545-20261121, poi verifica con ls-remote. Se il push è rifiutato, fermati.
3. Controlli in sola lettura su Leonardo: cartella di uscita e checkout assenti, quota (almeno 4 GB liberi oltre al checkout), coda, prenotazioni.
4. Job di checkout con il submitter comune, ID CONTRAST-CHECKOUT-CN-RESTART545-20261121, 0 GPU-ore. Atteso COMPLETED 0:0 e la riga «checkout di campagna pronto» con lo SHA di release.
5. Validazione --validate-only sul nodo di login, eseguita come script con ssh leonardo 'bash -s' e uscite salvate. Atteso: uscita 0 e "new_binding_valid": true.
6. Job GPU con il submitter comune, ID CONTRAST-CN-RESTART545-20261121, --gpu-ore 1.0, 4 GPU, 32 CPU, 240 G, esattamente 15 minuti, QoS di debug.
7. Raccolta sul Mac di report, tempi per fase, ricevute per rank, i due log di SLURM e l'accounting. I file .pt restano su WORK.
8. Voce nel registro docs/claude_GPT_shared, commit e push, esito a Simone con il testo pronto da incollare a GPT per la lettura scientifica.

VINCOLI
- Un solo tentativo per ciascun ID. Nessun rilancio dopo un errore, un timeout o un esito INCONCLUSIVE: raccogli e riferisci a Simone.
- Ogni passo solo se il precedente è riuscito. Se un controllo non torna o un comando dà un errore inatteso: fermati e riferisci, senza improvvisare correzioni.
- Tetto richiesto: 1,0 GPU-ora e 8,2667 ore locali (inviluppo con i margini di SLURM 1,4 e 11,8667). A piena allocazione restano 12,26 GPU-ore e 114,0011 ore locali sul tetto firmato.
- Nei due invii usa --agente claude. È l'unica differenza voluta rispetto ai comandi scritti nel protocollo di GPT, che nominava 3-launcher e --agente gpt: il protocollo fa parte del pacchetto vincolato e non va modificato.
- Su Leonardo fai solo le scritture previste dalla sequenza: il push del riferimento deploy e i due invii. Nessuna modifica a file remoti, nessuna cancellazione.
- L'esito scientifico è il campo status di report.json (SUPPORTED, PARTIAL_SUPPORT, HYPOTHESIS_UNSUPPORTED o INCONCLUSIVE). Nessun esito abilita training: il verdetto storico resta FAILED_REQUIRES_REVIEW. La lettura scientifica spetta a GPT.
- Il certificato ssh alle 15:05 era valido fino al 28/11 alle 02:34.

Non serve che tu mi risponda: l'esito va a Simone e nel registro.
