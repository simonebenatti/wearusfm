# Linea 1 — Lettura dell'endpoint a 19.618 passi, fissata prima dei lanci

Data: 06/11/2026. Scritta da Claude (operatore e autore del codice della linea 1) sulle definizioni gia' fissate;
la responsabilita' scientifica e la review sono di GPT.

Quando e' scritta: dopo aver visto gli esiti a 5.000 passi, prima di qualunque job della continuazione. Non esiste
alcun punteggio del 300M a 19.618. Le definizioni sono quelle di S8 (`docs/scaling_300m_protocol_20261104.md`,
fissate il 04/11/2026 prima delle estrazioni a 5.000) e della politica `scientific_reading_policy.json`: qui non
cambia nessuna soglia, nessun margine e nessuna unita' di ricampionamento.

## 1. Confronto

- Candidato: `T100-G-300M` a 19.618 passi, ripresa esatta del run accettato a 5.000 (stesso seme, stessi dati,
  stessa ricetta, schedule originale 49.046/1.962).
- Controllo: `T100-G` da 30M a 19.618 passi, gia' esistente: checkpoint
  `a31c895061c1f273eaac2d796fc00cf76cffa4318b521a20be4d6ba65b63a056`, due cache di posa e cinque di classificazione
  elencate negli inventari rivisti. Nessun nuovo training del 30M e nessuna riestrazione del controllo.
- Teacher contro teacher, stessa dose (19.618 contro 19.618), un solo seme.
- Viste naturali primarie. La proiezione a dimensione pareggiata e' secondaria: aiuta a interpretare, non
  sostituisce l'esito dei primari.
- Stessi fold, supporti, popolazioni, righe e selezione degli iperparametri sul solo train usati a 5.000.

## 2. Primari (invariati)

| Misura (300M meno 30M) | Migliore | Peggiore | Irrisolto |
|---|---|---|---|
| EPN-612, delta in punti percentuali | limite inferiore IC95 > 0 | limite superiore IC95 < 0 | altrimenti, anche con limite uguale a zero |
| `E_dyn`, delta del rapporto di errore | limite superiore IC95 < 0 | limite inferiore IC95 > 0 | altrimenti, anche con limite uguale a zero |

- `E_dyn` e' l'errore della ridge istantanea sulle finestre dinamiche, con il supporto gia' fissato.
- Posa «migliorata in misura pratica»: riduzione relativa puntuale `(E30 - E300) / E30 >= 0,02` e IC del delta
  interamente sotto zero. Un miglioramento sotto il 2% resta misurato ma non prende questa etichetta.
- Nessun margine di 1 pp si applica a EPN-612.

## 3. Controlli P1 e P2 (invariati)

Per ciascun IC95 `[L, U]` del delta in punti percentuali, con margine di non inferiorita' di 1 pp:

- `L > -1`: conservazione sostenuta;
- `U < -1`: perdita oltre il margine;
- altrimenti irrisolta; le uguaglianze sono irrisolte.

## 4. Intervalli

Bootstrap appaiato per soggetto, 2.000 repliche, seme 20261102. Gli intervalli sono marginali e condizionati a un
solo seme di pretraining e ai dati di sviluppo: nessuna garanzia congiunta del 95% su piu' endpoint.

## 5. Che cosa dice la lettura e che cosa non dice

- Dice, per il 19.618: l'etichetta di ciascun primario (migliore, peggiore, irrisolto), l'etichetta pratica della
  posa, lo stato di P1 e di P2, con stime puntuali e intervalli. E' una descrizione dell'esito.
- La tabella delle conseguenze operative di S8 (proporre il 19.618, oppure pausa) serviva a decidere il seguito a
  partire dal 5.000: e' stata applicata allora e resta immutata. Non si riapplica al 19.618: la lettura nuova non
  contiene la voce `propose_19618_confirmation` ne' alcuna raccomandazione di arrivare a una dose.
- Non autorizza altro training (49.046 o altre dosi): ogni seguito e' una decisione nuova di Simone.
- Non sceglie il checkpoint migliore fra 5.000 e 19.618 dopo i risultati. Le due letture restano affiancate,
  ciascuna alla propria dose; il confronto fra dosi e' descrittivo e non ha una regola di selezione.
- Un solo seme: la differenza fra 5.000 e 19.618 non misura la variabilita' fra training indipendenti.
- Un esito irrisolto non e' equivalenza. Un esito peggiore non boccia la taglia 300M ne' lo scaling in generale.
- Artefatti, appaiamento o convergenza dei lettori non validi: si corregge la valutazione; nessuna lettura
  scientifica su punteggi incompleti.

## 6. Descrittive

Conteggi meglio, uguale, peggio per soggetto con aritmetica esatta (interi o `Fraction`), mai con confronti a zero
su differenze in virgola mobile (rettifica `10fc9cc` sulla terza review di GPT).

## 7. Stato dell'implementazione

Il codice che applica questa lettura al 19.618 (politica versionata con protocollo distinto da
`scaling-reading-5000-20261104-v1`, lettori e confronto) appartiene allo stadio di valutazione: non e' in questa
consegna. Si scrive mentre gira il training, con i suoi test, e si pubblica con i pin reali del checkpoint a
19.618. Se quel codice si discostasse da questa pagina, vale questa pagina e la differenza si segnala prima di
aprire i punteggi.
