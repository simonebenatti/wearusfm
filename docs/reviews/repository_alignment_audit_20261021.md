# Audit locale, GitHub e registri — 21/10/2026

Verifica dei riferimenti GitHub effettuata tramite il connettore GitHub, confrontando
gli SHA locali con i rami remoti; non basata soltanto sui riferimenti origin locali.

## Stato trovato prima della pubblicazione

| Copia / ramo | Locale | GitHub | Riscontro |
|---|---|---|---|
| Canonico `/Users/simonebenatti/dev/wearusfm`, main | 79db894 | 79db894 | Pulito, allineato |
| GPT, codex/evaluation-w1-20261012 | 3272398 | 4203d59 | Ultima review non pubblicata; quattro file non tracciati |
| Claude, worktree-claude-training-engine | 134b77c | 134b77c | Pulito e allineato al secondo controllo |
| Vecchio clone `FM per EMG/wearusfm`, main | a3eda50 | main remoto 79db894 | Pulito, fermo al 05/10, 201 commit indietro |

Durante il controllo Claude ha committato e pubblicato il test inizialmente
non tracciato `tests/cpu/test_branch_pilot_trainer.py` in `134b77c`: risolto, non
un file perso. Il vecchio clone e' separato dal canonico e va riconosciuto come
copia storica. Cinque worktree temporanei risultano gia' rimossi ma ancora nel
registro Git come prunable; nessuna rimozione eseguita in questo audit.

## Passaggi rimasti incompleti

1. **Pubblicazione GPT:** `3272398` era solo locale. Il ramo GPT non aveva neppure
   un upstream configurato: il semplice `git status` non mostrava l'ahead remoto.
2. **Quattro file GPT mai committati:** `finestra1_job_plan.md`,
   `tokenizer_wavelet_proposal.md`, `step1_downstream_results_20261012.md` e
   `reviews/test_ultrareview_residuals_20261015.py`. Sono ora recuperati con una
   nota di archivio; non diventano nuovi piani approvati o verifiche correnti.
   La nota downstream cita `results/passo6/step1_downstream_20261012.json`, assente
   dalla storia Git controllata: riferimento esplicitamente lasciato da recuperare.
3. **Sintesi del registro arretrata:** Claude aveva gia' registrato C2 confermata,
   104,6/220 GPUh e prima tranche della diagnostica sottomessa; la tabella e il
   riepilogo GPT riportavano ancora stati precedenti. Corretti dai log e risultati
   versionati, senza reinterpretare le decisioni scientifiche.
4. **Pilot registrato solo nei commit:** e19f314/fadb881/134b77c descrivono campi
   di configurazione, cablaggio e test, ma non aggiornano il relativo riepilogo.
   Riportato nella tabella come codice consegnato, test dichiarati da Claude,
   verifica GPT del pilot ancora da fare. Nessuna evidenza di test saltati.
5. **Istruzioni SCRATCH superate:** CLAUDE.md citava ancora il 25/10, mentre il
   refresh 15/10 e il prossimo 12/11 erano gia' in decisioni.md. Istruzione aggiornata.

## Integrazione e allineamento documentale

La prima consegna risulta realmente integrata: 72cf24a, a7edd55 e 41ce569 sono
antenati di main. `c21ca40` non lo e': la seconda tranche resta aperta su R1/R2/R3.
Nessuna falsa chiusura del merge rilevata. Il ramo Claude ha 45 commit aggiuntivi
rispetto a main iniziale, incluso un merge; non sono 45 commit persi su GitHub.

Il ramo Claude corrente contiene anche il nuovo pilot, con modifiche in `src/`:
il vecchio «nessun file in src» valeva per la tranche precedente, non per HEAD.
Il prossimo merge del codice va delimitato e revisionato sullo SHA consegnato.

In questa consegna si consolidano nel registro le voci di Claude da 134b77c,
testualmente e in entrambe le sezioni, insieme alle voci GPT. decisioni.md riprende
le aggiunte gia' presenti sul ramo Claude allo stesso SHA. Per gli script,
protocolli e risultati non ancora integrati, la fonte rimane quel ramo:
[consegna Claude 134b77c](https://github.com/simonebenatti/wearusfm/tree/134b77c4ac57e0621896d187db0d6a86c554c1b7).
Riportare le registrazioni non equivale a integrare o approvare il codice.

Pubblicazione richiesta da Simone: commit di archivio/audit, aggiornamento del
ramo GPT su origin con upstream, avanzamento documentale di main conservando gli
SHA e controllo finale delle teste remote. I checkout di Leonardo rimangono quelli
delle rispettive campagne; nessuna sottomissione o deploy in questa attivita'.
Esito del push e SHA finale sono verificati dopo il commit e riportati nella consegna.
