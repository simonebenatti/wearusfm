# Scaling 30M/300M a 5.000: lettura scientifica parziale di GPT

Richiesta di Simone del 06/11: analisi indipendente della linea 1, da confrontare
con Claude per scegliere il seguito. **La posa e' leggibile; il confronto completo
non e' ancora disponibile.** Alla verifica di origin, il ramo di esecuzione Claude
e' `37632d18e0078069c54d131c9ba235891712c1b5`: 14 fit conclusi, sei comandi degli
aggregati preparati, nessun aggregato o confronto pubblicato. La Review 10
`d3976c3` approva codice e integrita', non la qualita' dei modelli; dichiara score
non aperti. Non attribuiamo a Claude una lettura scientifica finale inesistente.

## Evidenza verificata

Report posa: `results/infra/scaling_300m_20261104/cpu_fit_release_20261105/submissions/collected_jobs/SCALING-pose-ridge-20261105/report.json`,
SHA256 `8f70ff3b58c8f54413f6c8a89a861a4fdb7e6c9d934a65e73a11d4172a47e4ff`.
Teacher a 5.000, ridge scelta su 40 utenti train, 15 utenti val, stessi bersagli
e supporti; medie dei 64 slot, dimensioni 384/896, controllo proiettato a 384.
Tutti gli alpha sono interni alla griglia estesa: nessun bordo irrisolto.

Ricalcolati indipendentemente i nove contrasti e gli IC dai riepiloghi per utente,
con 2.000 repliche appaiate/seme 20261102: concordanza entro 1e-12. Script
`scaling_pose_partial_20261106_check.py` in questa cartella; risultati compatti in
`results/infra/scaling_300m_20261104/scientific_partial_20261106/pose_independent_summary.json`.
Una seconda lettura indipendente concorda. Questa verifica non ricarica le grandi
predizioni/cache: la loro rivalidazione integrale resta nel confronto congelato.

## Risultato sulla dinamica

| Lettura su finestre dinamiche | 30M | 300M | 300M proiettato a 384 |
|---|---:|---:|---:|
| Errore dinamico normalizzato E_dyn, minore meglio | 0,899816 | 0,899931 | 0,899312 |
| Correlazione centrata | 0,321300 | 0,319380 | 0,317605 |
| Varianza predetta / varianza target | 7,080% | 7,585% | 9,242% |
| MSE totale standardizzato sul supporto centrato | 1,082232 | 1,091326 | 1,092475 |

Primario naturale 300M meno 30M: **+0,000114908**, IC95
**[-0,003386818; +0,003777880]**. Riduzione relativa puntuale -0,01277%:
praticamente nulla. Il controllo proiettato ha delta -0,000503872,
IC95 [-0,003473304; +0,002424663], riduzione +0,0560%: nessun vantaggio evidente.

E_dyn e' il rapporto fra errore quadratico delle traiettorie centrate per
finestra/giunto e variabilita' del bersaglio sullo stesso supporto. Non e'
un'accuratezza: circa 0,90 significa circa 10% di riduzione dell'errore centrato
rispetto a una previsione senza variazioni temporali. Il rapporto usa somme
aggregate, non la media dei rapporti degli utenti.

Il 300M produce oscillazioni leggermente maggiori senza aumentarne la correlazione
con il movimento vero. La deviazione standard predetta vale 26,6%/27,5% di quella
del target: il 7% riguarda la varianza, non l'ampiezza. Nove utenti su 15 migliorano,
ma questo conteggio descrittivo non sostituisce l'endpoint aggregato prespecificato.

## Interpretazione e limiti

1. **A questa dose non vediamo un guadagno della taglia sulla posa dinamica
   accessibile a questo lettore.** Il delta e il suo IC sono piccoli anche rispetto
   al riferimento pratico del 2% (circa 0,018 unita' di E_dyn). Non e' soltanto
   assenza di significativita', ma non e' una prova generale di equivalenza: gli IC
   sono condizionati a un solo training e non includono variabilita' fra semi.
2. Il controllo a dimensione pareggiata non rivela un vantaggio nascosto nella
   lettura compressa. Una proiezione casuale resta un controllo secondario: non
   dimostra che larghezza e profondita' abbiano effetti separabili, o che ogni
   altra lettura delle rappresentazioni fallirebbe.
3. Il MSE totale sulle dinamiche peggiora descrittivamente dello 0,84%; su tutto
   il supporto centrato passa da 0,874130 a 0,890675 (+1,89%). Circa il 98,15%
   dell'extraerrore complessivo riguarda il livello medio. Non aggiungiamo questi
   secondari a posteriori ai criteri di arresto di S8.
4. Il test legge una ridge su medie degli slot. Non certifica tutta l'informazione
   nei latenti, la capacita' di un lettore non lineare, o il fine-tuning dell'intero
   modello. La rappresentazione per patch puo' gia' contenere contesto temporale.
5. Sono 5.000 passi, di cui 1.962 di warmup, alla stessa esposizione e nella stessa
   ricetta, non due modelli confrontati a convergenza o a costo uguale. Dose,
   obiettivo, ottimizzazione e pooling rimangono spiegazioni possibili; questa
   prova non le distingue e non dimostra che il dataset sia troppo piccolo.

## Come procedere

La sola posa non giustifica spostare adesso le risorse sul 300M. Completiamo prima
la classificazione gia' prevista. Con EPN612 migliore e nessun danno prespecificato,
il seguito ha una motivazione concreta; con entrambi i primari irrisolti resta
una domanda sulla dose, da pesare rispetto alle altre linee. Un peggioramento
prespecificato richiede pausa e revisione della ricetta, non bocciatura dello
scaling in generale.

S8 rimane invariata: EPN612 e posa sono primari; P1/P2 hanno margine -1pp; il
segnale pratico della posa richiede riduzione puntuale almeno 2% e IC sotto zero.
Qui la sola posa e' **irrisolta**. Non eseguiamo S8 con valori mancanti e non
selezioniamo un nuovo endpoint. **Nessun seguito a 19.618 autorizzato:** decide Simone
dopo il confronto completo. Nessun job, deploy, raccolta remota o modifica alla
numerica effettuato da GPT per questa analisi.
