# T100-LQ: risultati della posa, verifica indipendente del 07/11/2026

**Il ramo log-quadratico non migliora la lettura del movimento, primario della
posa, rispetto a T100-L né a 5.000 né a 19.618 passi con questo lettore e ricetta.** Il movimento
diventa più leggibile aumentando la dose, ma il miglioramento è condiviso dai
controlli. Questa prova non giustifica una continuazione esclusiva di LQ a 49k.

La domanda registrata è l'aggiunta di `W_log log(0.001+x²)` alle risposte lineari
del tokenizer da 100 ms, inizializzata a zero, a dose uguale. T100-L resta il
controllo primario; T25-L e T100-G sono secondari. Non si cambia il controllo
dopo aver osservato quale delle quattro celle è migliore.

## Verifica effettuata

Originali: `results/infra/logsquare_evaluation_20261107/final/pose5000/report.json.gz`
e `pose19618/report.json.gz`. SHA-256 e byte decompressi coincidono con i pin
remoti archiviati nei rispettivi `verification.json`. I rapporti finali indicano
job 63786157 e 63786158, stage compare completo, codice
`9a53c7a65f59de683be093a4ad1ac450a473bdf2` e i piani B attesi.
I pin di sorgenti, ricetta, cache e riferimenti coincidono con i file pubblicati.
Le tre celle storiche sono identiche, come oggetti JSON completi, ai report
originali alla stessa dose; nessun refit o cambio di provenienza.

Ho ricalcolato dai sufficienti per utente, con codice separato e **senza importare
funzioni di produzione**: rapporto di errore centrato, decomposizione MSE,
correlazione, rapporto delle varianze, MAE, segni individuali e bootstrap.
Il bootstrap usa 2.000 ricampionamenti dei 15 utenti, seed 20261102 e gli stessi
draw fra celle; la ricostruzione usa le molteplicità dei soggetti estratti.
Tutti i 24 intervalli dei rapporti e i 18 contrasti fra bracci coincidono con
quelli salvati entro 2e-12 assoluto/relativo. Nessun draw scartato.

I risultati completi, inclusi tutti i gruppi dynamic/all/static, tutti i contrasti,
i valori per utente, alpha e diagnostica delle colonne logaritmiche, sono in
`results/infra/logsquare_evaluation_20261107/final/pose_independent_analysis.json`.
Le matrici di predizione remote non sono state rilette localmente: il confronto
di produzione ne ha già verificato hash e corrispondenza con i sufficienti;
questa analisi indipendente riparte dai sufficienti presenti negli originali.

## Confronto primario sulle finestre dinamiche

E_dyn è l'errore quadratico dopo aver centrato separatamente predizione e bersaglio
in ogni finestra/giunto, diviso per la varianza temporale del bersaglio.
Più basso è meglio; 1 equivale all'errore di una predizione senza variazione
temporale. La misura non valuta da sola il livello assoluto della posa.
Supporto comune: 8.329 finestre dinamiche, 15 utenti, 20.119.920 elementi validi.

| Dose | T25-L | T100-L, primario | T100-G | T100-LQ |
| --- | ---: | ---: | ---: | ---: |
| 5.000 | 0,921457 | 0,910612 | 0,899816 | 0,912850 |
| 19.618 | 0,851975 | 0,843363 | 0,838933 | 0,844088 |

| Dose | Contrasto LQ meno controllo | Delta E_dyn | IC95% appaiato | Riduzione relativa dell'errore | Utenti favoriti da LQ |
| --- | --- | ---: | --- | ---: | ---: |
| 5.000 | T100-L, primario | +0,002238 | [−0,000242; +0,004880] | −0,246% | 3/15 |
| 5.000 | T25-L, secondario | −0,008607 | [−0,011759; −0,005579] | +0,934% | 14/15 |
| 5.000 | T100-G, secondario | +0,013035 | [+0,007875; +0,018795] | −1,449% | 1/15 |
| 19.618 | T100-L, primario | +0,000726 | [−0,001190; +0,002724] | −0,086% | 6/15 |
| 19.618 | T25-L, secondario | −0,007887 | [−0,010246; −0,005484] | +0,926% | 15/15 |
| 19.618 | T100-G, secondario | +0,005156 | [+0,003040; +0,007297] | −0,615% | 3/15 |

Nessuno dei sei contrasti dinamici soddisfa insieme la riduzione pratica del 2%
e un IC del delta interamente sotto zero. LQ è sostanzialmente sovrapposto a
T100-L, un poco migliore di T25-L e un poco peggiore di GELU. Il vantaggio su T25-L
non dimostra il valore aggiunto del ramo: T100-L, alla medesima finestra lunga,
raggiunge già quel livello senza il ramo.

## Movimento e livello assoluto

| Dose/braccio | Correlazione centrata | Var(pred)/Var(target) | MAE dinamica, rad | MSE livello | MSE centrata |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5.000 T100-L | 0,29915 | 0,08359 | 0,331787 | 0,639700 | 0,478385 |
| 5.000 T100-LQ | 0,29997 | 0,06089 | 0,332258 | 0,627387 | 0,479561 |
| 19.618 T100-L | 0,39578 | 0,15610 | 0,300370 | 0,552159 | 0,443056 |
| 19.618 T100-LQ | 0,39493 | 0,16191 | 0,300370 | 0,554861 | 0,443437 |

Le due componenti MSE sono in angoli standardizzati al quadrato e sommano alla
MSE totale sul medesimo supporto. A 5.000 LQ guadagna un poco sul livello ma non
sul movimento; a 19.618 non mantiene neppure quel piccolo vantaggio sul livello.
La MAE dinamica finale è praticamente identica: 0,300369819 rad per LQ contro
0,300370153 per L.

LQ passa da E_dyn 0,91285 a 0,84409: −7,53% di errore centrato. T100-L passa da
0,91061 a 0,84336: −7,39%. La correlazione LQ sale da 0,300 a 0,395; la quota di
varianza temporale delle predizioni sale dal 6,1% al 16,2% di quella dei bersagli.
L'ampiezza RMS centrata è quindi circa il 40% di quella del movimento vero al
secondo endpoint, non il 16%: varianza e ampiezza non vanno confuse.

In LQ la MSE del livello scende dell'11,56% e quella centrata del 7,53%; entrambe
migliorano con la dose. A 19.618 tutti i 15 utenti hanno E_dyn sotto 1, ma circa
l'84,4% dell'errore normalizzato del riferimento senza movimento resta presente.
È informazione di movimento reale, ancora limitata; non una dimostrazione che
il ramo LQ abbia aggiunto informazione rispetto alla base.

La progressione tra i due checkpoint è descrittiva. Non prova che il guadagno
continui fino a 49k e non separa dose, dinamica di ottimizzazione e regolarizzazione
del lettore, che viene riselezionata sul train a ogni dose.

## Gruppi all/static e controllo delle eccezioni

Il JSON pubblica tutti i contrasti anche per all/static. Rispetto a T100-L:

| Dose/gruppo | E_LQ | E_L | Delta | IC95% delta | Riduzione relativa |
| --- | ---: | ---: | ---: | --- | ---: |
| 5.000 all | 0,934495 | 0,936962 | −0,002467 | [−0,004713; −0,000172] | +0,263% |
| 5.000 static | 1,043372 | 1,069504 | −0,026132 | [−0,032046; −0,020293] | +2,443% |
| 19.618 all | 0,874064 | 0,872725 | +0,001340 | [−0,000336; +0,002885] | −0,154% |
| 19.618 static | 1,024849 | 1,020421 | +0,004429 | [+0,000934; +0,008211] | −0,434% |

Esiste quindi un vantaggio centrato del 2,44% sulle statiche a 5.000, coerente
con predizioni meno variabili. Non è il primario dinamico, entrambi i rapporti
restano sopra 1 e il vantaggio scompare all'endpoint successivo. Non va omesso,
né usato per promuovere il candidato dopo l'esito nullo del primario.

## Alpha e ramo apprendibile

Otto selezioni alpha su otto sono risolte: minimo CV ricostruito con pareggio
all'alpha minore, scelta interna alla griglia effettivamente valutata e
`boundary_unresolved: false`. A 5.000: T100-L sceglie 1e5; T25-L, T100-G e LQ
scelgono 1e6. A 19.618 tutti scelgono 1e5. Sono state usate le estensioni registrate:
due o tre verso l'alto, nessuna nuova griglia. La ridge ha soluzione algebrica;
non c'è una mancata convergenza iterativa da invocare per spiegare l'esito.

Tutte le 64 colonne di W_log sono non nulle in student e teacher a entrambe
le dosi. Nel teacher, norma media per famiglia:

| Dose | Fourier | Spline | MLP |
| --- | ---: | ---: | ---: |
| 5.000 | 0,13833 | 0,06549 | 0,10694 |
| 19.618 | 0,26775 | 0,19490 | 0,32581 |

Il ramo è stato aggiornato, inclusa la famiglia MLP: l'esito non deriva da nuovi
pesi rimasti a zero. Le norme non misurano da sole il contributo funzionale delle
famiglie, perché le scale delle feature differiscono; non sono un criterio di
promozione e non giustificano modificare epsilon dopo i risultati.

## Implicazioni e limiti

Per la posa terrei T100-L come base del prossimo confronto controllato. Non
promuoverei LQ e non proporrei un 49k LQ isolato per inseguire questo risultato.
Se una continuazione più lunga viene valutata per ragioni indipendenti, occorre
un confronto alla medesima dose e una nuova proposta esplicita; qui non viene
autorizzata né avviata.

La misura riguarda teacher congelato, media K64 e ridge istantanea su utenti di
validazione di sviluppo. Non misura tutti i latenti, un lettore temporale diverso,
il fine-tuning end-to-end o l'utilità generale del FM. Un solo seme e dati di
sviluppo riutilizzati: gli IC sono sull'incertezza fra utenti, non fra pretraining
indipendenti. I confronti secondari e multipli restano esplorativi. L'esito nullo
non esclude in generale la potenza di banda, una vera estrazione di energia o LEAF;
questa trasformazione per kernel non implementa quelle alternative.
