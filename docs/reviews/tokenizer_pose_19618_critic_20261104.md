# Tokenizer: seconda review indipendente della posa a 19.618 passi — 04/11/2026

**Analisi verificata. La finestra lunga e la GELU danno piccoli miglioramenti
della dinamica letta dalla ridge, ma nessun contrasto raggiunge il 2% relativo
puntuale previsto come grandezza pratica.** Il risultato non è nullo e non è
una soluzione all'attenuazione del movimento. Non conclude ancora la matrice:
la domanda sulla classificazione EPN a 200 Hz richiede le sonde.

## Provenienza e replica numerica

Report originale `results/infra/tokenizer_evaluation_20261103/cpu_19618/final/report.json`,
SHA256 `e1011f7a91645ce07399654950a7230084e8d22dbf248cca277e76e487f2e446`.
Il digest locale coincide con quello calcolato sul file remoto. Job 63495299
COMPLETED 0:0, 142 secondi, 8 CPU, nessuna GPU; RSS batch 4.665.244 KiB.
Costo ridge 0,315556 core-ore, entro i 420 secondi previsti dal gate e il tetto
di 1.800 secondi. Runtime `b315dadfedc09c522deeb56af829d34ebc04f4b4` e piano
`7a469c694637b870280ce6fcc8f9dcac1181e6b16644bc0042d1f942dc206e49` corretti.

Controllati separatamente tutti i pin del report, le sorgenti locali, i sei
hash remoti di coefficienti/predizioni, l'identità teacher dei tre checkpoint,
le firme dei supporti e i fold. I 40 utenti train sono disgiunti dai 15 val;
alpha selezionato sul train pari a 10^5 per tutti, due estensioni superiori,
nessun minimo irrisolto al bordo. Crop6, target centrali originali e soglia
dinamica coincidono con il piano e con lo screening a 5.000 passi.

Una seconda implementazione, senza importare il reader né il primo helper
di verifica, ha ricalcolato dai soli aggregati:

- somme per utente, rapporto di errore, varianze, correlazioni e MAE;
- parità dei denominatori e dei conteggi, 8.327 finestre dinamiche supportate
  e 20.119.920 elementi validi in ciascun braccio;
- tutte le nove differenze appaiate nei gruppi dinamico/tutti/statico;
- i 2.000 bootstrap utenti, seed 20261102, mediante matrice dei conteggi di
  ricampionamento, inclusi hash dei draw, IC e flag della soglia pratica;
- confronto descrittivo con i report a 5.000, con ricetta, utenti, target e
  supporti invariati.

Tutti i controlli passano entro tolleranza relativa 10^-12 e assoluta 10^-14.
Non sono state rilette le grandi cache NPZ remote, rieseguiti fit o modificati
criteri. Questa è una replica dell'aritmetica del report, non una replica del
pretraining con un altro seme.

## Risultati verificati e lettura

| Braccio | E dinamico [IC 95%] | Correlazione centrata | Varianza predetta/target |
|---|---:|---:|---:|
| T25-L | 0,851975 [0,830512; 0,872204] | 0,384740 | 14,7801% |
| T100-L | 0,843363 [0,822275; 0,863226] | 0,395775 | 15,6104% |
| T100-G | 0,838933 [0,818348; 0,858542] | 0,401443 | 16,8815% |

| Confronto | Delta E dinamico [IC 95%] | Riduzione relativa | Utenti migliori |
|---|---:|---:|---:|
| T100-L − T25-L, finestra | −0,008613 [−0,010603; −0,006726] | 1,0109% | 15/15 |
| T100-G − T100-L, GELU | −0,004430 [−0,007286; −0,001475] | 0,5253% | 11/15 |
| T100-G − T25-L, combinato secondario | −0,013043 [−0,016256; −0,009656] | 1,5309% | 15/15 |

Gli IC appaiati dinamici sono tutti sotto zero. È corretto parlare di un
vantaggio piccolo e coerente sui checkpoint osservati. È scorretto chiamarlo
assenza di effetto perché inferiore al 2%, o promozione automatica perché gli
IC escludono zero. La soglia pratica puntuale non è raggiunta nemmeno dal
confronto combinato secondario.

La varianza predetta cresce rispetto a 5.000 passi, ma rimane il 16,88% di quella
dei target nel braccio migliore. È un rapporto di varianze: il corrispondente
rapporto RMS centrato è circa 41%, non 16,88%. Correlazione e errore, oltre alla
varianza, sostengono il piccolo miglioramento; non emerge un inseguimento forte.

Come secondario, sulle statiche la GELU aumenta E centrato rispetto a T100-L
di 0,004254, IC [0,000281; 0,008296]. L'effetto è piccolo, ma impedisce di
descrivere un miglioramento uniforme di ogni gruppo. Il confronto statico
T100-G−T25-L comprende zero. Non trasformare questo secondario in una nuova
regola di selezione.

Da 5.000 a 19.618 passi E dinamico scende, descrittivamente, del 7,54% in
T25-L, 7,39% in T100-L e 6,77% in T100-G. Il vantaggio combinato fra bracci
passa dal 2,35% intermedio all'1,53% finale. Non era quindi giustificato
estrapolare il vantaggio dello screening. Questi confronti fra dosi usano
checkpoint successivi della stessa traiettoria e ridge rifittate: non sono
una stima causale separata dell'effetto della dose né una conferma fra semi.

## Limiti e decisione consentita

La posa è emg2pose a 2 kHz. Nessuna conclusione su EPN a 200 Hz, trasferimento
generale del FM o altra regressione può essere dedotta da questi soli numeri.
Un solo seme e una validazione di sviluppo già utilizzata limitano la portata
degli IC: incertezza da utenti condizionata ai training osservati, senza
correzione globale per confronti multipli. Il risultato riguarda teacher più
ridge rifittata, distinto dalla testa student e dai vecchi B/R3c con crop4.

La lettura corretta è completare le sonde già approvate e discutere il quadro
insieme. Nessun nuovo training, loss, seme, dose o ramo log-quadratico è
autorizzato da questa review. Nessun difetto bloccante rilevato nell'analisi.

Letto anche `tokenizer_pose_19618_results_20261104.md`: tabelle, confronto fra
dosi e conclusioni coincidono con questa replica indipendente. Nessuna
correzione scientifica richiesta al rapporto principale.
