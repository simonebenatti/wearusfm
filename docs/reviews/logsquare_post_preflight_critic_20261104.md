# Review interna del collaudo LQ e del controller — 04/11/2026

Tre verifiche indipendenti: lettura R1, preventivo R2 e controller del training.
Questa review interna non sostituisce quella concordata con Claude.

**Verdetto: lettura coerente e controller pronto per la review di Claude;
nessuna release del training.** Nessun job aggiuntivo è stato sottomesso.

La lettura R1 ricostruisce dai quattro rank 27.648 statistiche scalari dei log
per entrambi i bracci: nessuna discrepanza rispetto al report. Sono confermati
27 gruppi, 1.728 righe kernel e 28 confronti fra frequenze. Zero trigger del
pavimento o del mascheramento; 276 trigger della derivata non mascherata e
277 mascherata. Nessuna esenzione dai criteri. L'analisi offline si rigenera
identica; gli 11 test coprono soglie inclusive, attraversamenti in entrambe le
direzioni, std nulla, supporti diversi, tutte le coppie fra frequenze e rifiuto
di un report modificato.

La proposta epsilon=0,001 è ragionevole ma da rivedere da Claude: riconosce
il trigger e non lo trasforma in una certificazione di stabilità. I 64 passi
sono all'inizio del warmup; il gradiente di W_log non misura tutta la
sensibilità propagata al banco. La ricostruzione indipendente del costo
conferma rho=0,9855573781, stima 16,7052 GPUh e tetto invariato 22 GPUh.

Il controller conserva gli argv e la ricetta del controllo, verifica la
configurazione completa e l'inizializzazione condivisa con nuovi pesi a zero.
La ripresa richiede il proprio endpoint completo, stesso codice/piano,
optimizer, EMA, quattro stati RNG e firma dei dati. Il piano della consegna
è REVIEW_REQUIRED, con release nulla: non può avviare il training.

Il primo passaggio del critico ha rilevato un limite ereditato: la verifica
dei checkpoint controllava i valori presenti senza l'inventario completo.
La correzione è confinata al controller nuovo. Ora controlla chiavi, forme e
dtype di student/teacher, inventario ordinato dei parametri AdamW, forme/dtype
dei momenti presenti e presenza dei momenti W_log. Parametri mai attivati
possono legittimamente non avere momenti. Il critico ha riletto il delta e
rieseguito 31 test, tutti verdi, comprese dieci corruzioni del checkpoint.
I quattro pin aggiornati e i 102 hash dei sorgenti corrispondono.

Restano dichiarati: memoria/partizione nel piano e nelle direttive SBATCH sono
affidate al submitter, mentre nodo, CPU, GPU e walltime sono ricontrollati dal
launcher. Immutabilità dei sorgenti sul cluster e finitezza completa durante
il run sono attestate dal driver congelato; il collector non rilegge i raw.

Verifica integrata del responsabile: **171 test passati in 28,08 s**, sintassi
Bash valida e `git diff --check` pulito. La suite comprende preflight, LQ,
attivazioni, parità storica, trainer, nuovo controller e analisi offline.

Prima del rilascio restano la review di Claude su R1/controller, adattatori
di valutazione implementati e revisionati, pin/release e checkout pubblicati.
rho_infer va misurato prima delle estrazioni. L'autorizzazione operativa di
Simone è già presente e non va richiesta di nuovo.
