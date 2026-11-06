# Lettore temporale della posa: risultati — 02/11/2026

**Il contesto ±100 ms produce un piccolo beneficio sulla dinamica di B, ma non recupera l'ampiezza del movimento.**
R3c beneficia maggiormente dello stesso lettore, restando peggiore di B sulla posa.
Non emerge una ragione per modificare subito il FM o lanciare un altro pretraining.

Run63228748 COMPLETED0:0, codice `440af702ac3ec00feb52549acb1b4935251dce38`,
[protocollo congelato](../pose_temporal_pca_protocol_20261102.md).
Encoder student B/R3c a19.618, congelati; ridge istantanea384 contro concatenazione1152
di t−4/t/t+4, patch25ms, senza PCA. Media dei64 slot già presente nelle cache.
Due fold sui40 utenti train per selezionare alpha;15 utenti val disgiunti, già osservati,
quindi lettura esplorativa. Non è un test del criterio firmato all'endpoint completo.

## Risultati sulle finestre dinamiche

L'errore è SSE delle variazioni centrate divisa per varianza del bersaglio sullo stesso supporto:
più basso è meglio;1 corrisponde a nessuna variazione predetta. Non è un rapporto di MAE.

| Encoder/lettore | Errore dinamico | Correlazione centrata | Varianza predetta/target | MAE assoluta, rad |
|---|---:|---:|---:|---:|
| B istantaneo | 0,859512 | 0,374819 | 0,139686 | 0,307254 |
| B temporale | **0,855321** | **0,384317** | **0,108476** | 0,308248 |
| R3c istantaneo | 0,986567 | 0,131333 | 0,037288 | 0,346560 |
| R3c temporale | **0,961495** | **0,197461** | **0,030772** | 0,346304 |

| Contrasto fissato dell'errore | Delta | IC95 appaiato utenti | Utenti con delta negativo |
|---|---:|---:|---:|
| B temporale−istantaneo | **−0,004191** | **[−0,006329; −0,001939]** | 12/15 |
| R3c temporale−istantaneo | **−0,025072** | **[−0,029226; −0,021678]** | 15/15 |
| Differenza dei due delta: B−R3c | **+0,020881** | **[+0,016502; +0,026248]** | 0/15 |

Per B, riduzione relativa dell'errore dinamico circa0,49%; mediana del delta perutente−0,004598.
Per R3c, circa2,54%; mediana−0,026015. La differenza positiva significa beneficio maggiore per R3c,
non inversione dell'ordine: B temporale0,8553 resta sotto R3c temporale0,9615.
La correlazione migliora in entrambi; la varianza delle predizioni diminuisce.
La MAE B aumenta di0,000994rad: nessuna conclusione di miglioramento uniforme.

Sul gruppo statico, l'errore centrato B passa1,02979→0,99053 e R3c1,09943→1,04245;
su tutte le finestre B0,88780→0,87778 e R3c1,00532→0,97494. Sono letture secondarie:
il primario fissato resta il gruppo dinamico. Le riduzioni sono compatibili anche con
soppressione di variazioni spurie; non identificano da sole un meccanismo di filtraggio.

## Validità del confronto

- Quattro cache con hash originali; codice, sorgenti, piano e ambiente invariati prima/dopo.
- Supporti/bersagli identici nelle quattro celle:8.327 finestre dinamiche,15 utenti,
  20.663.620 elementi bersaglio validi. Tutti:16.630 finestre/42.306.560 elementi dopo le regole
  di supporto; non confondere questi conteggi con le18.548 finestre della cache iniziale.
- Fold train identici a quelli congelati; alpha1e5 per i due istantanei,1e6 per i due temporali.
  Minimi interni alla griglia estesa, rispettivamente2 e3 estensioni verso l'alto, tutte già previste.
  Nessuna scelta sulla val. L'alpha maggiore non è da interpretare da solo come prova di un meccanismo:
  cambiano dimensionalità e ridondanza delle feature.
- Bootstrap2000, seed20261102, pesi utenti appaiati fra tutte le celle. IC descrittivi,
  senza molteplicità corretta né variabilità fra pretraining/refit indipendenti.
- Critico: replica indipendente di punti e IC dalle statistiche perutente, senza richiamare il
  comparatore originale, discrepanza massima0,0. Provenienze, fold, minimi e riepilogo coerenti.

## Lettura scientifica e seguito

1. **B contiene più informazione utile alla posa di R3c in questa vista.** Il nuovo lettore conserva
   il quadro già visto con E3; il contesto aggiunge poco a B e non risolve le predizioni attenuate.
2. **Non è una diagnosi della media sui canali.** Entrambi i lettori partono da latenti già mediati
   sui64 slot; confrontiamo contesto/capacità/selezione della ridge, non media contro slot interi.
   Il backbone è già bidirezionale: nessuna evidenza di funzionamento online o inseguimento rapido.
3. **Questo primo controllo temporale è concluso.** Non trasformare il piccolo beneficio in una
   ricerca automatica di molti lag o teste sugli stessi15 utenti. Nessun R3 completo o nuovo FM avviato.
4. Per la prossima decisione distinguere l'obiettivo: posa dinamica oppure trasferimento generale.
   Se vogliamo decidere sul pooling, serve un confronto dedicato con gli slot conservati, da
   checkpoint congelati, con dimensionalità/costo controllati. Se vogliamo cambiare la loss, serve
   una sola ipotesi e una verifica di trasferimento anche in regressione distinta dalla posa supervisionata.
   Questi sono candidati da discutere, non job inclusi nella catena appena conclusa.

La PCA è separata: [confronto appaiato delle pipeline](pca_existing_results_20261102.md) già concluso,
rettifiche Claude `d93e22e` accolte. Un rifit con PCA come unica differenza non è stato eseguito.

## Costi e materiali

Run completo209s×8CPU =0,46444core-ore, MaxRSS4,194GiB. Preparazione0,08944core-ore;
**catena totale0,55389core-ore,0GPU**, tutti quattro i job COMPLETED0:0, coda della catena vuota.
Accounting dei job parent, senza sommare batch/extern. Tetto approvato20,5core-ore;
richiesta affinata del run30min/8CPU/16GiB prima dei risultati, descritta nel
[registro di esecuzione](pose_temporal_execution_20261102.md).

Rapporto originale [run.json](../../results/infra/pose_temporal_20261102/execution/run.json),
[riepilogo](../../results/infra/pose_temporal_20261102/summary.json),
[ricevuta review](../../results/infra/pose_temporal_20261102/review_receipt.json).
SHA256 rapporto `f247d0b4d8724fa02b2c2ad92b27e179a6ba9430a343012b179363a34e919690`.
Coefficienti conservati fuori da Git su $WORK e recuperati sul Mac con hash verificato;
percorso e ricevuta in `execution/coefficients_receipt.json`. Nessun array wearable nel commit.
