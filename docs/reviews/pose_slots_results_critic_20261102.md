# Punto 6: verifica indipendente dei risultati sui contrasti tra slot — 02/11/2026

**Esito della review: nessuna incongruenza numerica o di provenienza rilevata.** I sette contrasti tra slot aggiungono un beneficio rispetto alla stessa media compressa a 48 coordinate, ma non recuperano la perdita introdotta da quella compressione. Conservando la media completa, il beneficio aggiuntivo su B è molto piccolo. Questo risultato non identifica la media degli slot come causa principale dell'attenuazione del movimento e non giustifica automaticamente una modifica del pooling o del FM.

La verifica è stata svolta dal critico in modo indipendente, sui soli artefatti locali. Nessun nuovo fit, estrazione, accesso remoto o job è stato eseguito durante la review. Il presente documento registra i controlli già effettuati e la loro interpretazione.

Il riferimento è il [rapporto originale del job 63252509](../../results/infra/pose_slots_20261102/execution/run.json), con SHA256 `9e775208fcd141780c2fac0f408d980150baa2b2bef4d173f7540ea22195137b`. Codice scientifico: `c9d9c02de2597e2ab5e3907d7ffe06ffb544bb62`. Piano: SHA256 `0dafa6d4a17cbe196b2508ec6cd4a31e32e6e117fd36d2c720ad96caf0145526`. Disegno e limiti erano fissati nel [protocollo](../pose_slots_protocol_20261102.md).

## Provenienza, selezione e supporti

Sono stati verificati gli hash dei file raccolti, la coerenza fra piano e sorgenti dichiarate, le otto celle B/R3c × M384/M48/S384/S720, i pin delle cache storiche e le ricevute delle estrazioni. Le ricevute incluse nel rapporto coincidono con quelle raccolte separatamente; gli stati dei file residui coincidono con quelli registrati nella raccolta. Gli hash delle grandi cache remote sono attestati dalla validazione eseguita dal reader: questa review locale non ha scaricato nuovamente tali cache.

Le verifiche delle estrazioni, già completate prima del run, attestano 1.907 finestre train e 18.548 finestre val per modello, con media FP16 riprodotta esattamente e differenze nulle nei controlli LayerNorm e denominatore. Metadati e stati degli ingressi coincidono tra B e R3c.

Nel rapporto finale tutte le celle condividono gli hash del supporto di fit, dei target e del supporto di valutazione. La soglia dinamica resta `0.2574231028556824`. I 40 utenti train e i 15 utenti val sono disgiunti; i due fold interni coincidono con quelli congelati nel piano. Ogni alpha selezionato minimizza l’errore CV fra i valori valutati ed è interno alla griglia, senza bordo irrisolto:

| Modello | M384 | M48 | S384 | S720 |
|---|---:|---:|---:|---:|
| B | 100.000 | 10.000 | 10.000 | 100.000 |
| R3c | 100.000 | 100.000 | 100.000 | 100.000 |

Il supporto delle metriche centrate è identico nelle otto celle:

| Gruppo | Utenti | Finestre con supporto | Elementi validi |
|---|---:|---:|---:|
| Dinamiche | 15 | 8.327 | 20.663.620 |
| Tutte | 15 | 16.630 | 42.306.560 |
| Statiche | 15 | 8.303 | 21.642.940 |

I conteggi delle metriche centrate richiedono almeno due istanti validi per finestra e articolazione; non vanno confusi con il numero totale di finestre estratte.

## Replica numerica indipendente

La replica è stata implementata separatamente in memoria, leggendo i conteggi e le somme per utente del JSON. **Non è stata richiamata la funzione `paired_comparison` né il codice di aggregazione del nuovo reader.**

Per ciascuno dei tre gruppi sono stati ordinati i 15 utenti e generati 2.000 ricampionamenti con `numpy.random.default_rng(20261102)`. Lo stesso insieme di utenti ricampionati è stato applicato a tutte le otto celle. Per ogni contrasto si è ricostruita la combinazione delle SSE centrate per utente; il denominatore è la somma della varianza temporale dei target sul medesimo supporto. Punto stimato e repliche sono rapporti di somme, non medie non pesate dei rapporti individuali. Gli IC sono i percentili 2,5 e 97,5 delle repliche.

Sono stati controllati:

- Tutti i **27 contrasti**, ossia nove per ciascuno dei gruppi dinamiche, tutte e statiche, con i rispettivi IC.
- I 24 IC dei rapporti di errore delle singole celle, otto per gruppo.
- Le somme aggregate ricostruite dai contributi per utente, la parità dei denominatori e dei conteggi, e i contributi individuali ai contrasti.

**Discrepanza assoluta massima osservata: 0,0**, sia nella replica dei punti e degli IC sia nella ricostruzione delle somme aggregate dai contributi per utente.

Inoltre, M384 è stato confrontato con il [precedente rapporto temporale](../../results/infra/pose_temporal_20261102/execution/run.json): per B e R3c coincidono **esattamente** l'intera analisi, la selezione CV e lo SHA dei coefficienti della precedente cella istantanea. Il riferimento storico è quindi riprodotto sullo stesso supporto, senza discrepanze attribuibili al nuovo percorso di lettura.

## Risultato principale sulle finestre dinamiche

La metrica è SSE delle variazioni centrate divisa per la somma della varianza dei target. Un delta negativo indica miglioramento; le cifre sotto sono variazioni del rapporto, non punti percentuali di accuratezza.

| Contrasto | Delta | IC95 appaiato |
|---|---:|---:|
| B: S384 − M384 | +0,019901 | [+0,017084; +0,022394] |
| R3c: S384 − M384 | +0,004254 | [+0,002277; +0,006295] |
| Differenza B − R3c dei precedenti delta | +0,015647 | [+0,012499; +0,018281] |
| B: S384 − M48 | −0,003216 | [−0,003807; −0,002523] |
| R3c: S384 − M48 | −0,002173 | [−0,002876; −0,001525] |
| Differenza B − R3c dei precedenti delta | −0,001043 | [−0,002139; +0,000215] |
| B: S720 − M384 | −0,000106 | [−0,000159; −0,000048] |
| R3c: S720 − M384 | −0,001495 | [−0,001924; −0,001161] |
| Differenza B − R3c dei precedenti delta | +0,001389 | [+0,001007; +0,001861] |

**Il controllo M48 è determinante per l'interpretazione.** S384 migliora rispetto a M48 in 14 utenti su 15 per entrambi i modelli. Poiché la componente media è la stessa, il confronto sostiene un contributo utile dei residuali in questa lettura; il risultato non è spiegato dalla sola compressione della media. Rimane dichiarato l'aumento di capacità del reader. La differenza fra il beneficio di B e quello di R3c in questo confronto ha un IC che comprende zero: non dimostra né una differenza certa né equivalenza fra i due benefici.

S384 resta però peggiore di M384: la perdita nel comprimere la media supera quanto recuperato dai sette contrasti. In B, l'errore passa da 0,859512 con M384 a 0,882629 con M48 e a 0,879413 con S384. Non sarebbe quindi corretto interpretare il peggioramento di S384 contro M384 come assenza di informazione utile nei residuali.

Con tutta la media conservata, S720 porta B da **0,859512 a 0,859406**, pari a una riduzione di appena **0,0123% relativo** dell'errore. L'IC del delta è interamente negativo, ma l'entità del beneficio rimane molto piccola; migliora in 11 utenti su 15. Anche la correlazione varia poco, da 0,374819 a 0,374961, e la varianza predetta resta circa il 14% di quella del target. Non emerge un recupero sostanziale dell'ampiezza del movimento.

R3c migliora con S720 da 0,986567 a 0,985072, in tutti i 15 utenti; il suo beneficio è maggiore di quello di B, ma B conserva un errore nettamente inferiore nelle stime osservate.

## Portata della conclusione

Il risultato riguarda **sette contrasti fissi di slot, una proiezione condivisa a 48 coordinate e questi reader ridge**, su encoder congelati. Non esaurisce tutti gli slot, le proiezioni, le interazioni non lineari o le scale temporali. I confronti incrementali aggiungono parametri al reader; il confronto a pari larghezza cambia anche quanta media viene conservata.

Gli IC sono descrittivi, senza correzione di molteplicità, condizionati a checkpoint, proiezioni e reader fissati. Non includono variabilità da nuovi pretraining o refit. I 15 utenti val erano già stati osservati: questa analisi non costituisce un nuovo test intatto.

La conclusione sostenuta è circoscritta: i residuali esaminati sono utili oltre la media compressa, mentre il loro contributo aggiuntivo oltre la media completa è molto piccolo per B. **Non è dimostrato che la media sia il collo di bottiglia principale, né che modificare il pooling durante il training migliori il FM o il trasferimento generale.** Il risultato non attiva automaticamente una nuova campagna.
