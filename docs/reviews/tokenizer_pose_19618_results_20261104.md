# Tokenizer: ridge teacher a 19.618 passi — risultati del 04/11/2026

**La finestra da 100 ms e la GELU migliorano la lettura della dinamica, ma il
vantaggio resta piccolo all'endpoint.** T100-L riduce l'errore dinamico dell'1,01%
rispetto a T25-L; la GELU aggiunge lo 0,53%; il vantaggio combinato è 1,53%.
Tutti e tre gli IC appaiati sono sotto zero, ma nessuna riduzione puntuale
raggiunge il 2% fissato come utilità pratica. Non è un risultato nullo né una
soluzione al movimento troppo attenuato. Questa lettura riguarda soltanto la
posa emg2pose a 2 kHz; la conclusione sull'endpoint completo attende le sonde
P1/P2, in particolare EPN-612 a 200 Hz e la loro convergenza.

Ridge **63495299**, COMPLETED 0:0, 142 secondi, 8 CPU. Runtime congelato
`b315dadfedc09c522deeb56af829d34ebc04f4b4`; gate pubblicato `57a2ba9` prima del
lancio. Nessun nuovo training, tuning sulla val, cambiamento di loss o dose.

## Provenienza e verifica

Report originale: `results/infra/tokenizer_evaluation_20261103/cpu_19618/final/report.json`.
SHA256 `e1011f7a91645ce07399654950a7230084e8d22dbf248cca277e76e487f2e446`,
uguale al digest ricalcolato direttamente sul remoto. Verificati anche gli hash
remoti di tutti i coefficienti e delle predizioni: questi sei file restano fuori Git.
Piano SHA256 `7a469c694637b870280ce6fcc8f9dcac1181e6b16644bc0042d1f942dc206e49`.

Tre celle teacher complete a 19.618 passi, sei pin cache/provenienza/ricevuta,
sorgenti, runtime e ricetta coerenti. Media K64, D384, cache FP16, LayerNorm
stateless FP32 e ridge FP64 con intercetta. I 40 utenti train e i 15 val sono
disgiunti; stessi due fold alternati per la selezione. Tutti e tre gli alpha
selezionati sono 10^5, con due estensioni superiori e nessun minimo al bordo
irrisolto. Selezione interamente sul train.

Crop6 comune: `6 <= t` e `t+6 < n_patches`; target della patch centrale da 25 ms,
senza nuovi lag o pooling temporale. Soglia dinamica 0,2574231028556824 applicata
al `pose_var` originale della finestra. Firme di target, metadati e maschere
coincidono tra bracci e con lo screening5k. Dei 18.548 record val, 16.620 finestre
hanno supporto centrato; dinamiche 8.329 selezionate, 8.327 supportate, 20.119.920
elementi validi; statiche 8.329 selezionate, 8.293 supportate. Nessuna esclusione
specifica di un braccio.

`final/verify_report.py` ricalcola dai soli aggregati, senza importare il reader,
le somme per utente, tutti i rapporti/correlazioni, gli IC e i nove contrasti
(tre coppie per dinamiche/tutte/statiche). Le 2.000 repliche bootstrap per utente,
seed 20261102, riproducono IC e hash dei draw; tolleranza numerica 10^-12 relativa
e 10^-14 assoluta. `final/compare_doses.py` controlla identità di ricetta,
supporti, riferimenti e sorgenti prima del confronto descrittivo tra dosi.

## Dinamica sui 15 utenti esclusi dal train

`E_dyn` è la SSE dopo aver centrato separatamente predizione e target per
finestra/giunto, divisa per la varianza supportata dei target. Più basso è meglio;
1 corrisponde, su questa metrica centrata, a non riprodurre variazioni temporali.
Il MAE include invece anche il livello assoluto della posa.

| Braccio | E_dyn [IC 95%] | Mediana per utente | Correlazione centrata | Varianza predetta / target | MAE dinamico, rad |
|---|---:|---:|---:|---:|---:|
| T25-L | 0,851975 [0,830512; 0,872204] | 0,857101 | 0,384740 | 14,7801% | 0,304224 |
| T100-L | 0,843363 [0,822275; 0,863226] | 0,850422 | 0,395775 | 15,6104% | 0,300370 |
| T100-G | 0,838933 [0,818348; 0,858542] | 0,850317 | 0,401443 | 16,8815% | 0,300068 |

| Contrasto | Delta E_dyn [IC 95%] | Riduzione relativa | Utenti con errore inferiore |
|---|---:|---:|---:|
| T100-L − T25-L: finestra | −0,008613 [−0,010603; −0,006726] | 1,01093% | 15/15 |
| T100-G − T100-L: GELU | −0,004430 [−0,007286; −0,001475] | 0,52529% | 11/15 |
| T100-G − T25-L: combinato, secondario | −0,013043 [−0,016256; −0,009656] | 1,53090% | 15/15 |

La direzione favorevole è consistente, soprattutto per la finestra lunga.
L'entità resta sotto il criterio preregistrato (riduzione puntuale >=2% e limite
superiore dell'IC del delta <0). Il confronto combinato era e resta secondario;
non si sposta la soglia dopo averlo letto. Non soddisfare la soglia pratica non
significa dimostrare equivalenza o assenza di segnale.

Anche il braccio con E_dyn minore produce soltanto il 16,88% della varianza
dinamica dei target. La correlazione di circa 0,40 e l'aumento della varianza
mostrano una lettura del movimento migliore dello screening, ma l'attenuazione
rimane marcata. Il risultato non giustifica dire che il collo di bottiglia fosse
interamente la durata del token o l'assenza della GELU.

## Confronto descrittivo con 5.000 passi

Stessi utenti, teacher, supporto crop6 e lettore ridge rifittato con selezione
alpha sul train. Sono due snapshot della stessa traiettoria prima del decay,
non esperimenti indipendenti né una nuova prova causale sulla dose.

| Braccio | E_dyn a5k → a19618 | Riduzione rispetto al proprio5k | Varianza predetta / target a5k → a19618 |
|---|---:|---:|---:|
| T25-L | 0,921457 → 0,851975 | 7,54045% | 5,3393% → 14,7801% |
| T100-L | 0,910612 → 0,843363 | 7,38511% | 8,3589% → 15,6104% |
| T100-G | 0,899816 → 0,838933 | 6,76620% | 7,0797% → 16,8815% |

Tutti migliorano proseguendo il training nella ricetta fissata. I vantaggi
relativi fra architetture si riducono: finestra 1,18%→1,01%, GELU 1,19%→0,53%,
combinato 2,35%→1,53%. Il segnale intermedio non anticipava un guadagno crescente
con la dose. Questi numeri non selezionano un checkpoint diverso dall'endpoint
predefinito e non autorizzano a completare 49.046 passi.

## Secondari e limiti

| Braccio | E centrato, tutte | E centrato, statiche | MAE tutte, rad | MAE statiche, rad |
|---|---:|---:|---:|---:|
| T25-L | 0,880350 | 1,023079 | 0,263240 | 0,224113 |
| T100-L | 0,872725 | 1,020421 | 0,260592 | 0,222615 |
| T100-G | 0,869735 | 1,024675 | 0,260645 | 0,223007 |

Sulle statiche la GELU peggiora il rapporto centrato rispetto a T100-L di
0,004254 [IC95% 0,000281; 0,008296], con peggioramento in 11/15 utenti.
La varianza aggiunta non è quindi tutta utile, e il MAE complessivo di T100-G
non migliora quello di T100-L. Nessun braccio è migliore su ogni misura.

Sono encoder teacher congelati con un lettore rifittato: non confondere questo
report con la testa student o con i pannelli di train, né confrontarlo
numericamente alle vecchie analisi B/R3c con crop4. La posa era supervisionata
nel pretraining: qui si misura generalizzazione a utenti esclusi, non
trasferimento a una regressione nuova. Gli IC sono esplorativi, condizionati
all'unico seme e agli utenti di sviluppo già utilizzati, senza correzione
globale per confronti multipli o variabilità tra pretraining.

La lettura complessiva delle modifiche attende le sonde convergenti P1/P2/EPN.
La domanda centrale sui segnali a 200 Hz rimane aperta finché non leggiamo EPN.
Nessuna nuova modifica o campagna viene lanciata sulla base di questo solo report.

## Risorse

Ridge: 142 secondi su 8 CPU, **0,315556 core-ore**, **0 GPU-ore**;
picco RSS batch 4.665.244 KiB (circa 4,45 GiB), tetto richiesto 30 GiB.
Il gate prevedeva 420 secondi contro un limite di 1800.
Validate+benchmark+ridge totalizzano **0,377778 core-ore**.
Il checkout comune endpoint aggiunge 0,009444 core-ore una sola volta;
non va duplicato nella contabilità delle sonde.
Le tre estrazioni posa19618 hanno già un consuntivo separato di 1,451667 GPU-ore.
Monitor automatico rimasto spento; questa raccolta risponde alla richiesta attiva.
