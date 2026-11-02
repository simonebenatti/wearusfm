# Punto6: informazione nei contrasti fra slot — protocollo del 02/11/2026

Simone autorizza «vai con6? puoi parallelizzare?». Preparazione dell'estrazione e dei lettori
in parallelo; review indipendente prima dell'esecuzione. Encoder congelati B e R3c, student19618.
Nessun nuovo pretraining, modifica al pooling del modello o nuova loss inclusi.

## Domanda e disegno fissato prima dei risultati

La media dei64 slot può rendere inaccessibile informazione utile alla dinamica della posa.
Verifichiamo **sette contrasti fissi tra slot dello stesso backbone**, non tutti gli slot.
Un esito negativo vale per questi contrasti/proiezioni; un esito positivo non dimostra che
modificare il pooling durante il pretraining migliori il FM o il trasferimento generale.

`enc.z` finale del backbone, forma batch×tempo×64×384, prima di `latent_mean`.
Nessun decoder o cambiamento di strato. Proiezioni fisse senza ricerca:

- Matrice di Hadamard Sylvester64; sette righe non costanti estratte senza reinserimento
  con PCG64(20261102): **13,26,3,44,40,43,51**, coefficienti ±1/64.
- Proiezione condivisa delle384 coordinate in48: matrice gaussiana PCG64(20261103),
  QR in float64 con diagonale R positiva, memorizzata in float32.
- Le matrici sono salvate una volta in `results/infra/pose_slots_20261102/projections.npz`,
  con SHA nel piano. **Nessuna rigenerazione QR sul cluster**, nessun apprendimento o selezione
  di proiezioni sui dati. Gli slot hanno identità apprese ma nessuna geometria ordinata assunta.
- Estrarre i contrasti su `Z−mean_K(Z)` in float32, proiettarli a48 e salvare soltanto
  **336 valori float32 per patch** in `.npy` incrementale. Non salvare la cache completa degli slot.

La media storicaFP16 è riutilizzata dopo verifica della sua riproduzione. M=LayerNormFP32(meanFP16),
eps1e−5, identica al controllo temporale. I contrasti grezzi si dividono per il medesimo
`sqrt(var_D(meanFP16)+eps)`: niente LayerNorm separata che amplifichi residui quasi nulli,
nessuna PCA o standardizzazione appresa. Padding definito dalle lunghezze reali, non dalla maschera posa.

| Lettore | Feature | Scopo |
|---|---|---|
| M384 | Media normalizzata384 | Riferimento storico, ricomputato sul supporto comune |
| M48 | M384 proiettata con Q | Controllo della sola compressione della media |
| S384 | M48 + sette contrasti48 | Confronto a pari larghezza con M384 |
| S720 | M384 completa + sette contrasti48 | Contrasto aggiuntivo conservando tutta la media |

M48 è necessario: anche un vantaggio S384−M384 potrebbe derivare dal comprimere la media come
regolarizzazione. **S384−M384 confronta due ricette**; S384−M48 e S720−M384 valutano il contributo
incrementale dei contrasti, con capacità aggiuntiva dichiarata. Nessuna singola cella identifica
il pooling come causa esclusiva. Quattro lettori×due encoder, non una ricerca di teste o semi.

## Dati, fit e metriche

Stesse finestre del controllo temporale: train40 utenti/200 registrazioni/1907 finestre,
val15 utenti/1950 registrazioni/18548 finestre. Riutilizzare CSV, manifest, pose_stats,
loader, scala di sessione e indice originale `ri` delle registrazioni. Anche nel preflight,
non rinumerare i record scelti: il seme della scala dipende da `ri`.

Hash checkpoint/config/ricevute, cache storiche e sidecar fissati nel piano. Confronto esatto
di metadati/ordine, target, maschere, std e pose_var; nessuna esclusione aggiuntiva dopo i risultati.
Supporto comune `4 <= t` e `t+4 < n_patches`, già usato nel temporale, anche se queste viste
non aggiungono lag. Dinamiche da pose_var originale≥**0,2574231028556824**, soglia invariata.

Ridge FP64 con intercetta non penalizzata; stesso split train a utenti alternati ordinati,
due fold20/20. Alpha mediante SSE standardizzato mascherato aggregato, griglia E3 da1e−2 a1e4
con fino a3 estensioni per lato, pareggio al più piccolo; minimo ancora al bordo ferma il report.
Nessuna scelta sulla val. Refit sui40 utenti, valutazione sui15 già osservati: esplorativa.

Metrica principale: SSE delle variazioni centrate/varianza del bersaglio sulle dinamiche.
Affiancare correlazione centrata, varianza predetta/target, MAE assoluta e conteggi perutente.
Contrasti fissati per B e R3c: S384−M384, S384−M48, S720−M384; riportare anche le tre
differenze dei delta B−R3c. Stessi2000 ricampionamenti utenti, seme20261102, per tutte le celle.
Il lettore M384 deve essere confrontato con il precedente istantaneo sullo stesso supporto: eventuali differenze numeriche vanno spiegate prima dell'interpretazione. Non richiede un fit aggiuntivo.
Tutte/statiche sono letture secondarie; IC descrittivi senza correzione di molteplicità,
senza variabilità fra pretraining indipendenti. Nessuna soglia di promozione automatica del FM.

## Controllo numerico prima di accedere alle nuove metriche

Preflight sui soli indici di registrazione train **0,1,2,3** per entrambi i modelli.
Rigenerare meanFP16 e confrontare con la cache storica, per **ogni patch reale**:

- RMS su384 coordinate delle feature LayerNorm nuove−storiche ≤**1e−3**;
- massimo assoluto delle differenze normalizzate ≤**1e−2**;
- differenza relativa del denominatore di normalizzazione ≤**1e−3**.

Bitwise, differenze grezze e ULP sono diagnostici; il solo allclose grezzo non basta.
Gli stessi limiti si applicano a ogni patch nelle estrazioni complete. Un fallimento ferma
la catena: correggere la riproduzione, non allargare le soglie dopo aver visto i risultati.
Gli output incompleti restano distinguibili: ricevuta `complete` scritta soltanto dopo tutte
le verifiche e gli hash finali. Input/source/stati file controllati prima/dopo.

## Risorse e parallelismo

Riferimento storico osservato: train+val R3c1805s, B1800s, una GPU ciascuno, circa1GPU-ora
totale; MaxRSS circa19GiB. Non è un timing della nuova proiezione/scrittura.
I nuovi residui occupano **8,80GB circa (8,19GiB)** per entrambi i modelli in float32;
si riusano le medie esistenti. Cache completa dei64 slot sarebbe circa300GiB, non prevista.

| Fase | Risorse massime | Tetto GPU-ore | Tetto core-ore |
|---|---|---:|---:|
| Checkout congelato | 2CPU/2GiB/10min | 0 | 0,333 |
| Preflight train B e R3c, sequenziali nello stesso job | 1GPU/8CPU/60GiB/15min | 0,25 | 2 |
| Estrazione B | 1GPU/8CPU/60GiB/fino a60min | 1 | 8 |
| Estrazione R3c | 1GPU/8CPU/60GiB/fino a60min | 1 | 8 |
| Benchmark lettori sul solo train B | 8CPU/30GiB/15min | 0 | 2 |
| Report completo dopo timing | 8CPU/30GiB/fino a60min | 0 | 8 |
| **Totale massimo** | | **2,25** | **28,333** |

I core-ore includono anche le CPU dei job GPU. Prima delle estrazioni complete verificare
tempo, memoria e spazio dal preflight; non dare per scontato che quattro registrazioni siano
rappresentative del costo di tutto il corpus. Usare anche il riferimento completo storico,
scegliere walltime entro60min con margine esplicito, fermarsi se la proiezione supera il tetto.
Dopo estrazione B, benchmark CPU dei quattro lettori; run finale solo dopo benchmark e
completamento verificato di entrambe le estrazioni. Affinare risorse entro i tetti senza
cambiare vista/ricetta secondo risultati val. Nessun retry cieco o accodamento anticipato del report.

B e R3c sono indipendenti e **possono essere accodati in parallelo dopo il preflight**;
l'avvio simultaneo dipende dalla coda. La selezione e la lettura finale dipendono dalle cache.
Preparazione codice estrattore/lettori e review procedono in parallelo, con file distinti.

## Esecuzione e autorizzazione

L'approvazione di Simone copre punto6a/6b e il parallelismo;6c resta condizionale ai risultati.
Submitter e ledger comuni, SHA completo e checkout nuovo `pose-slots-20261102`, matrici/piano
congelati prima del preflight. Output sotto `$SCRATCH/wearusfm_runs/results/pose_slots_20261102/`,
log esterni in `$WORK/wearusfm_runs/logs/`; ricevute/rapporti ricopiati nel repository, nessun
dato wearable nei commit.

La regola generale riserva i lanci a semplicione, la cui chat non è raggiungibile. Il02/11 Simone
estende l'eccezione: **«Autorizzo il lancio diretto di questa catena»**. GPT procede con submitter e
ledger comuni, senza ulteriori consensi scientifici o operativi per le fasi già approvate.
La regola generale resta per campagne diverse.
