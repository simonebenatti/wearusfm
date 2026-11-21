# Diagnosi di architettura del controllo temporale — 21/11/2026

Lettura indipendente del runtime congelato nel worktree `wearusfm-tcn-control-analysis`
(HEAD iniziale `d3e41d06`). Nessun job, addestramento o modifica ai sorgenti del runtime.

## Riscontri

1. **Non emerge un'incompatibilità rappresentativa fra generatore e candidata.**
   `controls.generate_components` usa un generatore temporale largo 8, con le stesse
   convoluzioni dilatate 1/2/3 e lo stesso campo ricettivo di 13 patch della TCN
   candidata larga 64 e poi 32. La LayerNorm iniziale ha lo stesso epsilon; la
   candidata include la LayerNorm senza trasformazione affine fra i propri stati.
   Si possono collocare i pesi del generatore nei primi 8 canali della candidata,
   azzerare gli altri e assorbire la normalizzazione di T nell'ultimo strato.
   Il ramo affine rappresenta contemporaneamente la componente L normalizzata.
   Il dropout è disattivato in valutazione.

2. **Verifica costruttiva, non di apprendibilità.** `oracle_embedding.py` importa il
   generatore e la testa veri, senza modificarli; genera 8 finestre sintetiche,
   trasferisce i pesi noti e fa solo forward CPU. Per i due seed di generazione,
   con a=0,4, RMSE centrale circa 5,0e-7 e massimo errore assoluto 6,4e-6.
   I piccoli scarti sono compatibili con aritmetica FP32 e normalizzazione in
   float64; non è una pretesa di parità bit per bit. Risultati in
   `oracle_embedding.json`. Questo non dimostra che un ottimizzatore possa
   recuperare la funzione da questi dati rumorosi, né che generalizzi ai soggetti
   esclusi; il teacher-init non può diventare un criterio di ammissione.

3. **Non c'è un blocco permanente dovuto all'uscita zero.** Con gli ultimi pesi
   del ramo profondo a zero, i pesi nascosti non ricevono gradiente al primo
   backward; l'ultimo strato e l'affine lo ricevono. Dopo l'aggiornamento del
   primo, il cammino verso i nascosti si apre. La concorrenza tra un cammino
   affine più facile da ottimizzare e uno profondo resta una possibile dinamica
   dell'ottimizzazione, non un difetto dimostrato dal codice.

4. **Il controllo è deliberatamente rumoroso e il margine è limitato.** Le due
   calibrazioni reali hanno selezionato a=0,4 e sigma=1,3; il rapporto nominale
   fra deviazione del rumore AR(1) e componente T normalizzata è 3,25. Il margine
   relativo clean-LS→oracolo sugli inner held è rispettivamente 7,860% e 7,824%.
   T non è scomparso: la WLS istantanea lascia sul fitting il 71,17% e 87,88%
   della sua variabilità dinamica; la WLS L13 su y* recupera soltanto il 27,09%
   e 19,93% del margine nel diagnostico di fitting valutato su y. Questi ultimi
   sono descrittivi di fitting, non punteggi held e non una soglia per la TCN.

5. **LayerNorm non equivale a decorrelazione.** Il codice sottrae media e divide
   per deviazione entro ciascuna patch, ma non rende isotropa la covarianza fra
   feature o fra tempi. Non ho misurato tale covarianza nei latenti veri: un
   problema di condizionamento rimane un'ipotesi, non una diagnosi accertata.

## Ordine raccomandato per la proposta da revisionare

- Non cambiare ora soglia, calibrazione o selettore per convertire il fallimento
  in un passaggio. Il risultato presente resta `recipe_not_admitted`.
- Prima di scegliere un rimedio, un piccolo protocollo diagnostico può usare
  gli stessi due mondi e split, la stessa TCN e il LR già fissato 0,003, ma
  imparare il bersaglio noto y* senza il rumore sintetico. Due traiettorie con
  curve train/held fino all'orizzonte dichiarato separerebbero difficoltà di
  apprendimento della funzione, gap fra soggetti e impatto del rumore. Non è
  la ripetizione del controllo di ammissione e non autorizza i fit reali.
- Solo dopo la diagnosi scegliere un unico emendamento: se il pulito generalizza
  e il rumoroso no, capacità/regolarizzazione sono priorità più motivate di
  più epoche o di un semplice warm start. Un controllo con meno canali deve
  essere parsimonioso e prespecificato, non scelto per copiare la larghezza 8
  del generatore.
- Un warm start dell'affine dalla ridge fittata sul solo fitting è coerente
  matematicamente quando la LayerNorm parte dall'identità. Non è però un test
  di residuo fisso se affine e LayerNorm condivisa restano allenabili: entrambi
  possono far muovere la baseline. Congelarli isolerebbe il residuo, ma cambia
  la ricetta e richiede un confronto dichiarato. Il warm start non cura da solo
  una separazione fra buone prestazioni fitting e cattive prestazioni held.
- Whitening, nuova loss dinamica e nuova distribuzione sintetica cambiano
  ulteriori elementi: non li proporrei insieme al primo rimedio. Modificare
  il controllo dopo averlo osservato richiederebbe dichiarare il vecchio esito
  e una nuova specifica, non reinterpretarlo come passato.

## Sorgenti lette

- `scripts/pose_temponet/heads.py`: architettura, inizializzazione e forward.
- `scripts/pose_temponet/train.py`: loss per finestra, clipping, optimizer,
  passaggio train/eval e gestione RNG.
- `scripts/pose_temponet/controls.py`: generazione, normalizzazione, calibrazione
  e formule dei gate.
- `docs/pose_temponet_frozen_protocol_20261117.md`, §§4.2–4.4.
- `results/infra/pose_temponet_execution_20261121_A1/launcher/calibration_0.json`
  e `calibration_1.json` (solo scelte e diagnostici, non scansione riscritta).

Riproduzione della sola prova meccanica, dal worktree:

```
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python results/infra/pose_temponet_temporal_diagnosis_20261121/architecture/oracle_embedding.py
```
