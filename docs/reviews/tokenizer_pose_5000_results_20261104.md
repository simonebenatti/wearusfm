# Tokenizer: ridge teacher a 5.000 passi — risultati del 04/11/2026

**La finestra da 100 ms e la GELU mostrano ciascuna un piccolo miglioramento della
lettura del movimento alla dose intermedia.** L'effetto cumulato di T100-G contro
T25-L è una riduzione del 2,35% dell'errore dinamico; l'inseguimento resta debole.
Questo è lo screening descrittivo a 5.000 passi, non il verdetto all'endpoint
19.618 né una selezione del braccio da continuare.

La ridge 63405791 è COMPLETED 0:0 dal 03/11, 13:50:56, dopo 134 secondi.
La raccolta è stata eseguita il 04/11 su richiesta di Simone; nessun nuovo job o
controllo periodico è stato attivato da questa lettura.

## Provenienza e controlli

- Report originale: `results/infra/tokenizer_evaluation_20261103/cpu_5000/final/report.json`,
  SHA256 `2a7d8ccfaade9d8a540ecfb4e95821f5d67a93df735afbafb4ea6e863df980c3`.
  Copia locale identica al digest calcolato direttamente sul file remoto.
- Runtime congelato `70fd080149c11d10917059875d31a5a2820f176a`; sorgenti scientifiche
  identiche al piano. Piano SHA256
  `9efbf8279b1a6fdbcd551bae2326e36b74f551c7469f5cbd46842c49906b39ba`.
- Tre celle complete, tutte teacher allo stesso passo 5.000. Sei pin cache,
  provenienze e ricevute coincidono con il piano. Il job ha verificato gli hash
  prima dell'analisi e l'invarianza dei file alla fine. Gli hash dei sei output
  coefficienti/predizioni sono stati ricalcolati in remoto e coincidono col report;
  questi file restano fuori Git.
- Media K64, D384, cache FP16, LayerNorm stateless FP32 e ridge FP64 con intercetta.
  Stessi 40 utenti train e due fold alternati; i 15 utenti val sono disgiunti.
  Alpha scelti sul train: T25-L 10^6, T100-L 10^5, T100-G 10^6. Estensioni superiori
  rispettivamente 3/2/3, tutte entro la regola; nessun minimo irrisolto al bordo.
- Supporto `6 <= t` e `t+6 < n_patches` comune a fit e scoring. Target della patch
  centrale da 25 ms, senza nuova media temporale o cambio di maschera. Soglia dinamica
  0,2574231028556824 applicata al `pose_var` originale della finestra intera.
- Verifica aritmetica separata in `final/verify_report.py`: somme per utente,
  identità di supporti e denominatori, rapporti, correlazioni, IC e tre contrasti
  riprodotti dai soli aggregati del report, senza importare il codice del reader.
  Le 2.000 repliche bootstrap per utente, seed 20261102, riproducono gli IC entro
  tolleranza numerica 10^-12 relativa e 10^-14 assoluta.

Dei 18.548 record val, 16.620 finestre contribuiscono alla metrica centrata su tutte
le finestre. Le dinamiche selezionate sono 8.329, di cui 8.327 con supporto centrato:
166.540 celle finestra/giunto e 20.119.920 elementi validi. Le statiche selezionate
sono 8.329, di cui 8.293 con supporto centrato. Tutti questi conteggi sono identici
nei tre bracci; gli esclusi non sono rimossi separatamente per migliorare un braccio.

## Metrica primaria sulle finestre dinamiche

`E_dyn` è l'errore quadratico dopo aver centrato separatamente predizione e target
per finestra/giunto, diviso per la varianza supportata dei target. Più basso è meglio;
1 è il riferimento di una predizione priva di movimento nella finestra. Il MAE
riporta invece anche il livello della posa.

| Braccio | E_dyn [IC 95%] | Mediana E_dyn per utente | Correlazione centrata | Varianza predetta / target | MAE dinamico, rad |
|---|---:|---:|---:|---:|---:|
| T25-L | 0,921457 [0,906890; 0,933432] | 0,924974 | 0,285489 | 5,3393% | 0,336274 |
| T100-L | 0,910612 [0,892517; 0,925879] | 0,914322 | 0,299146 | 8,3589% | 0,331787 |
| T100-G | 0,899816 [0,882383; 0,915685] | 0,905400 | 0,321300 | 7,0797% | 0,325310 |

| Confronto | Delta E_dyn [IC 95%] | Riduzione relativa | Utenti con errore inferiore |
|---|---:|---:|---:|
| T100-L − T25-L: finestra | −0,010845 [−0,015351; −0,006861] | 1,17695% | 14/15 |
| T100-G − T100-L: GELU | −0,010797 [−0,017374; −0,005466] | 1,18563% | 12/15 |
| T100-G − T25-L: combinato, secondario | −0,021642 [−0,030177; −0,013868] | 2,34863% | 14/15 |

I tre IC appaiati sono sotto zero. I due passaggi principali hanno una riduzione
puntuale inferiore al 2%; soltanto il confronto pratico combinato supera
numericamente tale soglia. Il campo automatico
`signal_2percent_and_ci_below_zero=true` per quest'ultimo **non promuove lo
screening a verdetto finale**: il confronto è secondario, siamo a 5.000 passi e
mancano le sonde P1/P2 della dose finale. Gli IC di ciascun braccio possono
sovrapporsi pur avendo un contrasto appaiato sotto zero.

## Secondari e interpretazione

| Braccio | E centrato, tutte le finestre | E centrato, statiche | MAE, tutte le finestre (rad) | MAE statico (rad) |
|---|---:|---:|---:|---:|
| T25-L | 0,942023 | 1,045470 | 0,294952 | 0,255503 |
| T100-L | 0,936962 | 1,069504 | 0,289242 | 0,248624 |
| T100-G | 0,923817 | 1,044548 | 0,283736 | 0,244045 |

La finestra lunga aumenta la varianza dinamica predetta e riduce l'errore dinamico,
ma sulle statiche aggiunge variazioni che peggiorano l'errore centrato. La GELU
abbassa l'errore e migliora la correlazione rispetto a T100-L anche con varianza
predetta inferiore: questo illustra perché una curva di varianza più alta, da sola,
non prova un miglior inseguimento.

Passando da T25-L a T100-G, entrambi a5.000 passi, l'errore dinamico scende da
0,9215 a0,8998. È un segnale favorevole, di entità ancora contenuta. La varianza
dinamica predetta rimane il 7,08% di quella dei target; non presentare il risultato
come una soluzione all'attenuazione del movimento. La regressione è svolta su posa
emg2pose a 2 kHz: questi numeri non rispondono ancora alla domanda sui segnali
EPN-612 a 200 Hz.

La rappresentazione è teacher e il lettore è rifittato: questa analisi va tenuta
distinta dal pannello student del training e dalle vecchie prove B/R3c con crop4.
Il confronto usa un solo seme e utenti di validazione già utilizzati nello sviluppo;
gli IC quantificano l'incertezza da utenti condizionata a quel training, non quella
fra addestramenti. Gli IC sono esplorativi, senza correzione globale per confronti
multipli. Un secondo revisore ha replicato indipendentemente i nove contrasti e IC,
confermando la lettura descrittiva. Non dimostra trasferimento a una regressione diversa dalla posa,
né superiorità generale del FM.

Il passo già approvato rimane completare e leggere la stessa matrice a 19.618
passi, con P1/P2/EPN e le loro verifiche di convergenza. Nessun cambiamento di loss,
dose, seme, lettore o candidato log-quadratico è introdotto da questa lettura.

## Risorse

Ridge: 8 CPU, 30 GiB richiesti, walltime 134 s su un tetto di 1.800 s;
**0,297778 core-ore e 0 GPU-ore**. Picco RSS batch 4.600.864 KiB,
circa 4,39 GiB. Il gate aveva previsto 420 s.
Checkout, validazione e benchmark CPU precedenti costavano 0,072778 core-ore:
la catena CPU5k completa totalizza **0,370556 core-ore**. Le estrazioni GPU5k,
già rendicontate separatamente, restano 1,522500 GPU-ore; non aggiungerle due volte.

Evidenze: `final/report.json`, `final/remote_sha256.txt`, `final/sacct.txt`,
`final/ridge_63405791.out`, `final/verification_summary.json`. Nessuna cache di
segnali, predizioni o coefficienti è stata copiata nel repository.
