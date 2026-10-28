# B a 5.000: separare postura media e dinamica nella cache E1/E2

28/10/2026, GPT. **Autorizzata da Simone** con «va bene, procedi» alla proposta di separare postura media e variazioni
entro finestra usando la cache gia' prodotta, senza altra inferenza GPU o addestramento. Una sola analisi CPU,
metodo definito qui prima della sua esecuzione. E1/E2 e il collaudo B conservano risultati e regole originali.

## Domanda e dati congelati

La testa migliora soltanto la posizione media nella finestra, oppure predice anche una parte delle variazioni intorno a
quella posizione? Il gap E2 confronta finestre diverse e non separa questi due contributi. Questa analisi li distingue
al checkpoint student B a **5.000 passi**, sui 15 utenti val gia' estratti.

- Cache `B5000_student_E1E2_20261028/val.npz`, 2.484.409.931 byte,
  SHA256 `70bb551309b502565d38072abf41ffee322a06de14589d36b720570f1cfbfa46`.
- Checkpoint di origine SHA256 `815956b98d27cd2ff71f1255d0e220e2b0e08e467e7bedf93a54d0984b10783e`,
  codice di estrazione `24139ab901c052357a46a65bc20466f2eb1e8358`, seed finestre 20261026.
- Si leggono soltanto predizioni, bersagli, maschere, scale e metadati: **non servono i latenti**, il checkpoint o il modello.
- Nessuna selezione di checkpoint, utenti, giunti, lag o iperparametri in funzione del risultato. Nessuna modifica delle maschere.
- Primario: stesse finestre **dinamiche di E1**, `pose_var` finita e maggiore o uguale alla mediana globale finita della cache.
  Secondari descrittivi: tutte le finestre con supporto e quelle poco dinamiche secondo la medesima mediana. Nomi e conteggi
  dei gruppi restano tracciati; le finestre con varianza non definita rimangono solo nel gruppo tutte.

## Definizione dei contributi

Unita' primaria: **MSE nello spazio standardizzato** di predizioni e bersagli della cache. I radianti servono soltanto a
ricostruire E1 per il controllo di coerenza; MAE in radianti e MSE standardizzata non sono confrontabili come una stessa misura.

Per ciascuna finestra e ciascun giunto, siano p e y predizione e bersaglio, e n il numero di istanti validi comuni.
Si calcolano su questo solo supporto le rispettive medie mp e my, le varianze di popolazione Vp e Vy (divisore n), e la
covarianza C. La scomposizione esatta e':

```
MSE totale = (mp - my)^2 + MSE centrata
MSE centrata = media[((p - mp) - (y - my))^2] = Vp + Vy - 2*C
```

Il primo termine misura l'errore sul livello medio, il secondo l'errore sulle variazioni intorno al livello medio.
Per n=0 non si calcola alcuna metrica. La scomposizione totale e' definita anche per n=1; le metriche di dinamica e il loro
confronto usano **solo finestre/giunti con n>=2**, riportando eventuali differenze di supporto. Non si impone una soglia sulla
varianza per escludere a posteriori traiettorie difficili o quasi statiche.

La baseline per la dinamica centrata e' una predizione centrata nulla: errore **Vy**. Si riportano, sul medesimo supporto n>=2:

- rapporto `MSE centrata / Vy` e guadagno assoluto `Vy - MSE centrata`;
- rapporto di varianze `Vp / Vy` e **correlazione pooled delle traiettorie centrate** `C / sqrt(Vp*Vy)`;
- errore delle medie, MSE totale e rispettive componenti della baseline costante del train (zero standardizzato).

La relazione con il guadagno rispetto alla costante del train si conserva senza usare quote di energia come quote di miglioramento:

```
MSE della costante del train = my^2 + Vy
guadagno sul livello medio = my^2 - (mp - my)^2
guadagno sulla dinamica = Vy - MSE centrata
guadagno totale = guadagno sul livello medio + guadagno sulla dinamica
```

I contributi sono additivi in MSE, possono avere segni opposti e non vengono forzati in percentuali fra 0 e 100.
La correlazione pooled non e' la media delle correlazioni per utente o per giunto; non e' una misura di causalita'.
Vp, Vy, C e tutti gli errori vengono aggregati con gli **stessi pesi n**. Il rapporto delle somme non viene sostituito dalla
media dei rapporti per finestra o per utente. Si salvano conteggi e somme sufficienti per utente, oltre agli aggregati.

## Incertezza, validazione e interpretazione

IC95 percentile con **1.000 ricampionamenti degli utenti con reinserimento, seed 20261009**, ordinamento degli utenti
deterministico. Ogni utente porta insieme tutte le somme e il conteggio degli elementi: il ricampionamento mantiene
l'appaiamento fra predizione, bersaglio e baseline. IC per rapporto centrato e guadagno assoluto; il gruppo primario e'
quello dinamico. Si riportano anche utenti con segno favorevole/sfavorevole, senza chiamarli test significativi individuali.

Denominatore nullo, varianza nulla per la correlazione, meno di due utenti o repliche non finite producono una dichiarazione
di metrica/IC non valutabile e la causa; **non si scartano repliche** per ottenere un intervallo. Le somme e i conteggi
rimangono disponibili quando definiti. Calcolo float64 per limitare la cancellazione numerica; identita' verificate
numericamente prima della scrittura del risultato.

Attese dichiarate prima del calcolo:

- Una testa che recupera la sola media di finestra ha Vp=0 e rapporto centrato pari a 1 (quando Vy>0).
- Un rapporto centrato inferiore a 1 indica miglioramento sulle variazioni rispetto a una traiettoria piatta; la sua
  dimensione e l'IC descrivono quanto, senza una nuova soglia di promozione. Il rapporto puo' superare 1 se le variazioni
  introdotte dalla testa peggiorano la ricostruzione.
- Varianza predittiva piccola con covarianza positiva puo' dare un miglioramento reale ma attenuato. Un'alta correlazione
  da sola non implica ampiezza corretta o piccolo errore, quindi si legge insieme a MSE e rapporto di varianze.

Centrare usando la media del bersaglio osservato e' un **oracolo diagnostico**, non una baseline disponibile durante l'uso
del modello. Centrare **entrambe** le traiettorie concede alla predizione una correzione perfetta del bias per finestra/giunto:
un rapporto centrato <1 descrive il vantaggio sulle fluttuazioni dopo questa correzione, non prova che la predizione operativa
batta la media vera della finestra senza correzione. Per questo MSE totale, errore delle medie e MSE centrata si leggono insieme.
L'analisi e' condizionata alle finestre val gia' campionate e alla loro varianza del bersaglio, a un seme e a un checkpoint; non dimostra trasferimento
ad altri dataset. Rimane descrittiva a 5.000 passi: nessun nuovo cancello, modifica del verdetto B o via automatico a R3.

Controlli tecnici: hash della cache e della provenienza, origine student/val/step5000/code/stats concordi, forme/maschere/valori
finiti, copertura e utenti; ricostruzione di E1 MAE dinamiche e relativo IC contro il rapporto congelato, entro tolleranza
numerica dichiarata di 1e-6 relativa e 1e-8 assoluta. Test sintetici includono predizione perfetta, media di finestra, fase
invertita, bias, supporti incompleti e n=0/1, oltre al peso degli elementi e al ricampionamento per utente.

## Esecuzione e preventivo

**Un job CPU**, partizione `lrd_all_serial`, **4 core, 16 GB, massimo 10min**, **0 GPU-ore / massimo 0,666667 core-ore**.
Attesi circa 1–3 minuti piu' coda: il rapporto precedente ha richiesto 51s inclusi lettura e controlli; questa e' una stima,
non una durata garantita. Nessun nuovo checkout di campagna: ambiente e verifica del checkout `24139ab` gia' presente;
lo script di analisi nuovo e' distribuito nel payload versionato su stdin, con hash distinto e verificato prima dell'esecuzione.

ID logico `PILOT-POSA-B5000-dynamics-20261028`, campagna `pilot-posa-b-dynamics-20261028`, submitter/ledger comuni.
Output nuovo `$WORK/wearusfm_runs/results/pilot_posa/B5000_student_dynamics_20261028.json`, fuori dal codice e senza sovrascrittura.
Cache e provenienze lette senza modifica; niente dati grezzi o cache nel repository. Nessun retry automatico.
Preflight, protocollo, script/test, review del critico, piano e comandi vengono pubblicati prima della sottomissione.
Il via di Simone copre questa analisi; nessuna nuova autorizzazione viene richiesta per i suoi normali passi di esecuzione.
