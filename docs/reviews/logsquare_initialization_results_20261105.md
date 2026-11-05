# T100-LQ: diagnostica 63591184 conclusa e decisione sul seguito

05/11/2026. Codice congelato `1dc6ded8eee8a788277994687ff3b4b162453616`.
Job **63591184 COMPLETED 0:0**, 14:56:56–14:57:52 CET, **56 secondi**.
Raccolti 25 file e verificati tutti gli SHA contro il cluster; checkout ancora pulito.
Rapporto originale SHA256 `270ef5135d2e0d9baeddcba03ef2e60f6345c3e4fdbe11e103c9c6df6355eddf`.
Archivio senza perdita: `results/infra/logsquare_initialization_20261105/completed/`.

## Risultati e verifica indipendente

| Controllo | Esito |
| --- | --- |
| Sei contesti, ritorno del processo | Tutti 0, nessun errore |
| Tre criteri prefissati | Tutti veri; lettura computabile |
| Dieci costruzioni, cinque contesti per due bracci | 269 tensori condivisi: 268 identici; differisce solo `tokenizer.fe_gain` |
| Gain del preflight | 548.6939187567023 |
| Gain di controller e trainer | 548.6939187567021, −1 ULP (−1,1368683772161603e−13) |
| Sostituzione del solo gain su copia | Recupera il riferimento `38cb4cce…` in 10/10 |
| Parità T100-L / T100-LQ | Stato e RNG CPU/CUDA identici in 5/5 contesti |
| Trasferimento CUDA | Stato preservato esattamente in 10/10 |
| Quattro rank reali | Tutti attestano lo stesso stato; 24.576 pesi del ramo log-quadratico nulli |
| Aggiornamenti | Nessuno: verify-only, modello non restituito al trainer, run directory vuota |
| Supervisione, interna ed esterna | 7/7 senza timeout, segnali o superstiti; pulizia 0,013–0,014 s |

Il critico ha ricalcolato gli esiti dai campi e dagli hash per tensore senza richiamare
`reading()` della produzione; controllate anche le quattro ricevute dei rank e il
pin del piano contro il commit congelato. Verdetto indipendente favorevole.

La discrepanza del vecchio controllo è riprodotta nei contesti odierni: il preflight
ricostruisce `38cb4cce…`, mentre controller e trainer ricostruiscono lo stato con il
gain storico, osservato oggi come `0b6abc835db60b8974219b4e0b160309ba3661896cdd1a6c667bf616382e8654`.
Quest'ultimo hash è un'osservazione del rapporto, **non un nuovo riferimento adottato**:
il guard rimane quello già revisionato, basato sul vecchio hash e sul gain storico.
Non ricostruiamo a posteriori un log completo del processo fallito, che non esiste.
Non è una misura di apprendimento né un risultato a favore del ramo log-quadratico.

## Consuntivo

- Diagnostico: **0,062222 GPU-ore**, **0,497778 core-ore**.
- Checkout 63590929: 46 s con 2 CPU, **0,025556 core-ore**.
- Totale tranche: **0,062222 GPU-ore e 0,523333 core-ore**, entro 1 e 8,333333.
- Il primo sbatch del diagnostico era stato rifiutato senza creare un job: nessun costo
  di calcolo aggiuntivo e nessun duplicato. Evidenza operativa già archiviata.

## Decisione di GPT

**La correzione ha superato la verifica richiesta: approvo il rilascio della stessa
ricetta di training T100-LQ, sotto l'autorizzazione operativa già data da Simone.**
Questa è la decisione esplicita del responsabile scientifico dopo lettura e review;
il rapporto conserva correttamente `training_authorized_by_reading: false`.

Prima di sottomettere: nuovi piano/release del 05/11 con quattro hash controller,
guard e rapporto/lettura pinnati; controllo finale dei pin; commit dedicato di rilascio,
aggiornamento A4 dei piani valutativi e checkout congelato nuovo. Piano/release del
04/11, guard, diagnostica e artefatti del fallimento restano immutati.

Stessi dati, epsilon 0,001, inizializzazione registrata, loss, seme, schedule e dose:
5.000 passi più continuazione fino a 19.618, con stato optimizer/teacher/RNG conservato.
Nuovi run, ricevute e ID del 05/11; override 01:30:00 e 04:00:00. R2 invariato:
stima **16,7052 GPU-ore**, tetto **22 GPU-ore / 176 core-ore**, più checkout 0,333333 core-ore.
L'inferenza richiede ancora checkpoint completi e benchmark rho_infer; nessuna nuova
loss, contrastiva o monitor di completamento in background.
