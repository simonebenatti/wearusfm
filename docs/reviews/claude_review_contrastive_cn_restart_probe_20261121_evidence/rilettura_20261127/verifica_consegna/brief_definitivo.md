Consegna del lancio su Leonardo della prova mirata InfoNCE (linea 3): un checkout CPU e un job da 4 GPU per 15 minuti, già approvati e preparati; la sequenza con i comandi esatti è in un file del repository.

Ti scrive la chat Claude «3- log quadro e conrtrastive», revisore della linea 3. Questo messaggio viene da un'altra chat, non da Simone.

CHI LO HA DECISO
- Simone mi ha scritto il 27/11: «in realta farai tu il lancio. Anxi quando è tutto pronto, passa alla chat 0-leggi le tabelline e lei si occupera di fare il lancio su leonardo».
- La prova l'ha scelta Simone il 21/11 («Prova mirata»). Il job GPU chiede al massimo 1,0 GPU-ora, sotto la soglia di 5 GPU-ore delle sottomissioni autonome.
- Prima della prima scrittura su Leonardo serve comunque il via esplicito di Simone nella tua chat: è il passo 3 della sequenza, che dice anche che cosa scrivere nella richiesta (incidente, costo, residuo). I passi 1 e 2 sono in sola lettura e puoi farli subito.

CHE COSA È GIÀ PRONTO
- Review del revisore: diagnostico approvato, commit 76f5d0e7cface6f02bb24bf572ec57b74171a162 sul ramo origin/worktree-claude-training-engine.
- Commit di release, già su origin: b58a66f695c8744dfa5fe633829cafe13a5a1fc4, ramo claude/cn-restart545-release-20261127. È il codice di GPT 8c72655c più il solo file di binding.
- Worktree locale pulito su quel commit: /Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release. I due invii si lanciano da lì. Non modificarlo.
- Sequenza operativa con tutti i comandi: /Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine/docs/contrastive_cn_restart_probe_launch_20261127.md, nel commit @SEQ_SHA@ del ramo worktree-claude-training-engine. Leggila per intero prima di cominciare: è la fonte dei comandi e dei criteri di arresto. Controlla che il file non sia cambiato dopo quel commit: git -C /Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine diff --quiet @SEQ_SHA@ -- docs/contrastive_cn_restart_probe_launch_20261127.md (uscita 0 = invariato).

UNA COSA DA SAPERE PRIMA
Alle 15:33 del 27/11 un mio agente di verifica ha eseguito per errore il comando di invio del checkout con un ssh finto. Nessun job è arrivato a Leonardo, ma nel ledger comune l'ID del checkout era rimasto in stato «ambiguo». L'ho chiuso con la riconciliazione del submitter: ora è «assente». Per questo, nel solo comando del checkout, c'è l'opzione --dopo-assenza-verificata: è già scritta nella sequenza, non va aggiunta né tolta. L'ID del job GPU non è stato toccato.

CHE COSA DEVI FARE, IN ORDINE (comandi, esiti attesi e criteri di arresto sono nella sequenza)
1. Controlli sul Mac: worktree di release su b58a66f6… e pulito, submitter con l'impronta attesa, stato «assente» per l'ID del checkout e nessuna prenotazione per l'ID del job GPU.
2. Letture su Leonardo, in sola lettura: blocco A (repository canonico) e blocco B (cartelle assenti, quota, saldo, coda, prenotazioni).
3. Via esplicito di Simone nella tua chat; poi il push del riferimento di deploy deploy/cn-restart545-20261121 e la sua verifica.
4. Job di checkout con il submitter comune (0 GPU-ore), attesa di COMPLETED 0:0 e verifica del checkout.
5. Validazione --validate-only sul nodo di login.
6. Di nuovo il blocco B, poi il job GPU con il submitter comune (1,0 GPU-ora nel ledger).
7. Raccolta sul Mac di report, tempi per fase, ricevute, log e accounting. I file .pt restano su WORK.
8. Voce nel registro docs/claude_GPT_shared, commit e push sul ramo worktree-claude-training-engine (mai su main), esito a Simone con il testo pronto da incollare a GPT per la lettura scientifica.

VINCOLI
- Un solo tentativo per ciascun ID. Nessun rilancio dopo un errore, un timeout o un esito INCONCLUSIVE: raccogli e riferisci a Simone. Un comando di invio non si esegue mai due volte.
- Ogni passo solo se il precedente ha dato l'esito atteso. Se un controllo non torna o un comando dà un errore non previsto dalla sequenza: fermati e riferisci, senza improvvisare correzioni.
- Tetto richiesto: 1,0 GPU-ora e 8,2667 ore locali (inviluppo con i margini di SLURM 1,4 e 11,8667). A piena allocazione restano 12,26 GPU-ore e 114,0011 ore locali sul tetto di 18 e 169,6.
- Le differenze volute rispetto ai comandi scritti nel protocollo di GPT sono elencate nella sequenza (fra queste --agente claude). Il protocollo fa parte del pacchetto vincolato e non va modificato.
- Su Leonardo fai solo le scritture previste dalla sequenza: il push del riferimento di deploy e i due invii. Nessuna modifica a file remoti, nessuna cancellazione.
- L'esito scientifico è il campo status di report.json (SUPPORTED, PARTIAL_SUPPORT, HYPOTHESIS_UNSUPPORTED o INCONCLUSIVE). Nessun esito abilita training: il verdetto storico resta FAILED_REQUIRES_REVIEW. La lettura scientifica spetta a GPT.
- Il certificato ssh alle 15:05 del 27/11 era valido fino al 28/11 alle 02:34. Se scade con un job in coda o in esecuzione non fare nulla sul job: chiedi a Simone il rinnovo e raccogli dopo.

Non serve che tu mi risponda: l'esito va a Simone e nel registro.
