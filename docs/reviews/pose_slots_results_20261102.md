# Punto 6 — risultati media e contrasti fra slot, 02/11/2026

**La diagnosi 6a/6b è conclusa. I sette contrasti conservano informazione utile oltre la media compressa, ma aggiungerli alla media completa riduce l’errore dinamico di B di appena 0,0123% relativo, senza recuperare l’ampiezza del movimento.** Questa prova non giustifica da sola un cambiamento del pooling o un nuovo training del FM.

Tutti i sei job sono `COMPLETED 0:0`; report finale **63252509**, 16:12:56–16:19:51 CET, 415 secondi. Nessun job residuo della catena. Codice scientifico congelato `c9d9c02de2597e2ab5e3907d7ffe06ffb544bb62`. [Protocollo](../pose_slots_protocol_20261102.md), [report completo](../../results/infra/pose_slots_20261102/execution/run.json), [sintesi macchina](../../results/infra/pose_slots_20261102/summary.json).

## Domanda e quattro viste

B e R3c student a 19.618 passi restano congelati. M384 usa la media completa dei 64 slot; M48 la proietta in 48 dimensioni. S384 concatena M48 e 336 componenti di sette contrasti fissi fra slot. S720 aggiunge le stesse componenti a M384. Non sono gli slot interi né un nuovo pooling addestrato nel FM.

Ridge FP64, intercetta non penalizzata, alpha scelto solo su 40 utenti train in due fold; tutti gli alpha sono interni alla griglia. Val: 15 utenti, 18.548 finestre; train: 1.907 finestre. Soglia dinamica congelata 0,2574231028556824; 8.329 finestre selezionate dinamiche, 8.327 con supporto sufficiente per le statistiche centrate. Supporto centrale comune al precedente temporale.

## Dinamica: risultati primari

L’errore è SSE delle variazioni centrate divisa per la varianza dei bersagli, più basso è meglio. La varianza predetta è espressa come percentuale di quella dei bersagli. MAE riguarda gli angoli assoluti sullo stesso gruppo.

| Modello / vista | Errore dinamico | Correlazione | Varianza predetta | MAE rad | Alpha |
|---|---:|---:|---:|---:|---:|
| B/M384 | 0.859512 | 0.374819 | 13.969% | 0.307254 | 100000 |
| B/M48 | 0.882629 | 0.345590 | 15.288% | 0.311902 | 10000 |
| B/S384 | 0.879413 | 0.349415 | 15.070% | 0.311157 | 10000 |
| B/S720 | 0.859406 | 0.374961 | 13.965% | 0.307232 | 100000 |
| R3c/M384 | 0.986567 | 0.131333 | 3.729% | 0.346560 | 100000 |
| R3c/M48 | 0.992995 | 0.099905 | 2.386% | 0.351394 | 100000 |
| R3c/S384 | 0.990821 | 0.107671 | 2.459% | 0.350294 | 100000 |
| R3c/S720 | 0.985072 | 0.134882 | 3.687% | 0.346034 | 100000 |

### Tutti i contrasti primari previsti

Negativo = meno errore nella prima vista. Le differenze delle differenze sono effetto in B meno effetto in R3c; in quelle righe il conteggio negativo non significa necessariamente miglioramento assoluto. IC95% esplorativi da 2.000 bootstrap appaiati per utente, seed 20261102, senza correzione per molteplicità.

| Contrasto | Delta errore | IC95% | Utenti con delta negativo /15 |
|---|---:|---|---:|
| B S384 minus M384 | +0.019901256 | [+0.017084152; +0.022394235] | 0 |
| R3c S384 minus M384 | +0.004254304 | [+0.002277499; +0.006295052] | 2 |
| S384 minus M384 difference of differences | +0.015646952 | [+0.012498708; +0.018280928] | 0 |
| B S384 minus M48 | -0.003216043 | [-0.003806791; -0.002522962] | 14 |
| R3c S384 minus M48 | -0.002173340 | [-0.002876135; -0.001524862] | 14 |
| S384 minus M48 difference of differences | -0.001042703 | [-0.002138585; +0.000214947] | 11 |
| B S720 minus M384 | -0.000105900 | [-0.000159386; -0.000047982] | 11 |
| R3c S720 minus M384 | -0.001495087 | [-0.001923689; -0.001160762] | 15 |
| S720 minus M384 difference of differences | +0.001389187 | [+0.001006947; +0.001860707] | 0 |

**Pari dimensione:** S384 è peggiore di M384 in entrambi i modelli. **Controllo sulla compressione:** S384 migliora invece M48 in 14/15 utenti per entrambi. I contrasti aggiungono quindi utilità alla vista compressa, ma non compensano la perdita dovuta alla compressione in questa ricetta. Non è corretto concludere che siano privi di informazione.

**Media completa conservata:** S720 migliora B da 0,859512 a 0,859406, delta −0,000105900 (11/15 utenti), pari a **0,012321% relativo**. L’IC esclude zero, ma l’effetto è minuscolo. La varianza predetta rimane 13,97% circa e cala leggermente; correlazione 0,374819→0,374961, MAE 0,307254→0,307232 rad. R3c guadagna di più (delta −0,001495087, tutti i 15 utenti) ma resta nettamente peggiore di B sulla posa. La differenza delle differenze non favorisce una perdita selettiva di dinamica di B recuperabile con questi contrasti.

## Tutte le finestre e statiche: analisi secondarie

| Modello / vista | Errore tutte | MAE tutte rad | Errore statiche | MAE statiche rad |
|---|---:|---:|---:|---:|
| B/M384 | 0.887799 | 0.266721 | 1.029793 | 0.228023 |
| B/M48 | 0.918979 | 0.271228 | 1.101449 | 0.232394 |
| B/S384 | 0.914482 | 0.270449 | 1.090520 | 0.231583 |
| B/S720 | 0.887668 | 0.266685 | 1.029539 | 0.227973 |
| R3c/M384 | 1.005316 | 0.302198 | 1.099431 | 0.259844 |
| R3c/M48 | 1.005991 | 0.310039 | 1.071232 | 0.270557 |
| R3c/S384 | 1.004162 | 0.308533 | 1.071130 | 0.268662 |
| R3c/S720 | 1.003576 | 0.301639 | 1.096462 | 0.259254 |

Tutti i 18 contrasti secondari e relativi IC sono nella sintesi JSON; insieme ai 9 primari sono stati replicati indipendentemente. L’effetto a pari dimensione non ha lo stesso verso in tutti i gruppi: su R3c le statiche favoriscono S384 rispetto a M384. Non generalizziamo quindi la conclusione primaria a ogni regime. S720 continua a dare un beneficio piccolo in B.

## Integrità e review

- M384 replica **esattamente** il precedente istantaneo, per entrambi: analisi, selezione CV, SHA del supporto fit e SHA dei coefficienti. Il confronto con la prova temporale è quindi ancorato allo stesso riferimento.
- Piano, 115 hash di sorgenti/evidenze, ricevute raccolte, checkpoint e supporti coerenti. Le medie FP16 riestratte sono identiche su 2.513.510.400 elementi totali; LayerNorm e denominatore con scarto zero. Il report ha verificato gli hash delle quattro cache residuali prima dei fit.
- [Critico indipendente](pose_slots_results_critic_20261102.md): 27 contrasti/IC e somme per utente replicati senza richiamare la funzione di confronto, discrepanza massima 0,0; otto celle, fold, alpha, supporti e provenienze coerenti. Nessuna incongruenza aperta.
- Nessun cambiamento al codice scientifico dopo il congelamento; 54 test già passati prima dei job. Questa consegna aggiunge analisi e documenti, senza nuovi fit o test ridondanti.

## Costi, limiti e seguito

Consuntivo completo: **1,155556 GPU-ore e 10,254444 core-ore**, inclusi checkout, preflight, estrazioni, benchmark e report; tetti 2,25 / 28,333. Report: 8 CPU per 415s = 0,922222 core-ore, MaxRSS 4.084.332 KiB (3,895 GiB), entro la richiesta pubblicata di 8 CPU / 30 GiB / 15 min. Cache e coefficienti restano fuori da Git; qui si archiviano report, ricevute e statistiche.

Conclusione limitata a sette contrasti, una proiezione e lettori lineari. I controlli incrementali aumentano anche la capacità; non isolano una proprietà causale dei soli slot. Val già osservata, nessuna variabilità fra pretraining indipendenti, nessun nuovo test intatto o prova online. Il risultato non dimostra che gli slot completi o un lettore non lineare siano inutili, né che il FM trasferisca a una regressione diversa dalla posa.

**Raccomandazione:** chiudere 6a/6b e non attivare automaticamente 6c. Fermare la ricerca di altre teste/proiezioni sulla stessa val; discutere prima la domanda del punto 7: migliorare la dinamica di posa (7a) oppure il trasferimento generale (7b, con una regressione diversa dalla posa supervisionata). Il termine dinamico resta una possibile ipotesi per il training, non una soluzione dimostrata; una contrastiva richiede motivazione e confronto separati. Nessuna nuova loss, architettura o run è avviata da questa conclusione.
