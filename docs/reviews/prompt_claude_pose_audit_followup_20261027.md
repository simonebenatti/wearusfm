# Prompt per Claude — chiudere i controlli R3 e recuperare E3

Completa il seguito diagnostico del collaudo R3, sul tuo worktree e branch. Leggi la review GPT aggiornata e il suo repro sul ramo
`origin/codex/evaluation-w1-20261012`:

- `docs/reviews/pose_audit_e198a87_review_20261027.md`
- `docs/reviews/repro_pose_audit_review_20261027.py`

La review riguarda il codice e198a87 e incorpora il tuo aggiornamento operativo d037a3e. I controesempi dimostrano limiti degli
strumenti, non un bug gia' accertato nei dati o nel training R3. Le asserzioni del repro dimostrano il comportamento errato attuale:
trasformale in regressioni del comportamento corretto, senza conservarle come criteri di successo dopo la correzione.

## 1. Recupera E3 R3c

Procedi al recupero e al rapporto entro i tetti della sezione 5. Puoi avanzare su E3 mentre correggi l'audit J2.

- Riconcilia ledger e stato dei job prima di sottomettere: niente duplicati o retry alla cieca. Entrambe le vecchie estrazioni hanno
  letto val; il file chiamato train non e' utilizzabile come train.
- Verifica la val recuperabile e rinominala senza sovrascrivere output esistenti; conserva provenienza del nome errato e del job.
  Non ripetere l'estrazione val se la cache e' valida.
- Riestrai il vero train: 40 utenti, 200 registrazioni, stesso checkpoint R3c e statistiche congelate. Parti dai comandi corretti
  in `results/infra/pilot_posa/comandi_e3_r3c.txt`; usa argomenti separati/array, non una stringa di opzioni espansa in zsh.
  Verifica localmente gli argomenti effettivi, inclusi destinazione e split, prima del lancio.
- Prima del rapporto verifica il contenuto delle cache contro CSV e split: identita' di utenti/registrazioni, numerosita',
  disgiunzione prevista dal protocollo, checkpoint e normalizzazione. Il nome del file non certifica lo split. Se i metadati sono
  incompleti, ricostruisci dai manifest/log e dichiara cio' che manca; non produrre un E3 ammissibile con provenienza ambigua.
- Esegui il rapporto solo dopo il completamento corretto del train e questi controlli. R3c non ha testa: E1/E2 non applicabili.

## 2. Correggi l'audit J2 prima di lanciarlo

Implementa R1–R4 della review, con test mirati:

1. **Catena EMG-posa e copertura:** una fixture HDF5 unica attraversa entrambi gli ingest. Confronta campioni EMG raw/processati
   alle coordinate corrette, tenendo conto della trasformazione effettiva e della quantizzazione; controlla source_member,
   identita', frequenza e origine temporale. Copri lettura diretta e da blocco, bordi/gap; EMG sostituito o spostato deve fallire
   anche con metadati plausibili. Sui reali usa finestre campionate entro il budget. Distingui verificato/fallito/non verificato:
   zero raw letti o copertura incompleta non possono produrre un successo complessivo. Riporta attesi, verificati ed esclusi.
2. **Oracolo indipendente completo:** includi NaN/Inf secondo il contratto dell'ingest, senza riusare le sue funzioni di aggregazione
   e senza estendere arbitrariamente la dilatazione degli zeri. Usa le statistiche del lato della finestra; aggiungi fixture
   per_side=true e una finestra corta che renda non vuoto il controllo del padding.
3. **Lag scan esplorativa:** elimina il verdetto “minimo a zero = allineato”. Il minimo nonzero non prova un errore degli indici.
   Confronta i lag su supporto comune, medesimi target e baseline comparabile; riporta MAE assolute e N. Se selezioni un lag
   operativo, selezionalo dentro il fit e valutalo una sola volta sul test. Correggi il test: 40 blocchi sono 200 ms/8 patch,
   non 400 ms; il range deve includere il ritardo con margine e il test deve verificarne segno/magnitudine su un segnale
   identificabile, evitando l'ambiguita' del contesto. Aggiungi un caso allineato senza imporre che ogni curva reale abbia minimo 0.
4. **Movimento:** calcola differenze su indici originali, usando soltanto coppie adiacenti entrambe valide. Non comprimere i buchi
   con diff(target[valid]). Testa due plateau separati da un tratto non valido: il salto nel buco non e' movimento osservato.

Sistema anche la validazione interna di alpha: split per finestre/registrazioni con gap e normalizzazione stimata sul solo inner fit.
Conserva metadati di sessione/lato/fase/intervallo; verifica acquisizioni duplicate o simultanee fra split. Se non puoi ricostruire
la cronologia assoluta, dichiara il limite invece di chiamare l'ordine lessicografico una separazione temporale indipendente.
Il disallineato deve mantenere utente/lato/fase e distanza temporale adeguata; riporta le esclusioni e le MAE/gap sul supporto comune.
Se il campione non consente il pairing, segnala il controllo come non verificabile o descrittivo, senza allentare i vincoli in silenzio.

## 3. Chiudi il sintetico e la documentazione

- Correggi l'off-by-one dello schedule e confronta ogni passo con il trainer: il dbg ha dose 671, non 670 passi equivalenti al picco.
- Rendi coerenti teacher e LayerNorm: teacher sulle feature normalizzate oppure q riferito al raw e 1-q dichiarato solo limite
  inferiore, non optimum esatto della testa. Registra semi e scenari realmente eseguiti.
- Togli la conclusione esclusiva “H1 regge solo se B_eff basso/q piccolo”: il sintetico illustra scenari e non identifica la causa
  nel training con encoder mobile e ancore. B_eff fra 7 e 70 non e' una misura del dataset.
- Mantieni la guardia d'identita' compatibile con gli alberi senza sidecar e descrivila come controllo dei metadati. Il confronto
  dei contenuti e' compito dell'audit; non attribuirlo alla guardia.

## 4. Verifica e consegna

Esegui test CPU mirati con il src del tuo worktree e un thread BLAS, controlli dei launcher e verifica scientifica del critico.
Riporta anche l'esito della suite completa gia' avviata, con lo SHA a cui si riferisce; se intervengono modifiche, distingui chiaramente
le verifiche del nuovo codice da quelle precedenti. Correggi i problemi emersi e congela un nuovo SHA per J2.

Poi esegui checkout e audit corretto entro i tetti sotto, con procedura/submitter abituali e output fuori dal codice. Non modificare
checkout usati da altri job. Una copertura parziale produce un rapporto parziale esplicito, non un lancio aggiuntivo automatico.

Consegna commit, test, ID job, provenienza delle cache, copertura, risultati e consuntivi; aggiorna entrambe le sezioni del registro
e decisioni.md, conservando lo storico. Separa: correttezza della catena, informazione predittiva RMS, informazione nei latenti R3c,
e ipotesi ancora aperte. E3 negativo non esclude che R3 possa imparare; RMS negativo non prova un errore di allineamento.

## 5. Perimetro e tetti di questa tranche

| Attivita' | Risorse e durata massima | Costo massimo |
|---|---|---|
| Nuova estrazione train E3 | 1 A100, 8 core, 60 GB, 40 min | 0,6667 GPU-ore + 5,3333 core-ore |
| Rapporto E3 | 8 core, 60 GB, 2 h | 16 core-ore |
| Nuovo checkout J2 | 1 core, 2 GB, 10 min | 0,1667 core-ore |
| Audit J2 corretto | 4 core, 16 GB, 1 h | 4 core-ore |
| **Totale futuro** | Durate sommate 3 h 50 min, escluse code e lavoro Mac | **0,67 GPU-ore arrotondate e 25,5 core-ore** |

Le 0,84 GPU-ore delle estrazioni precedenti sono gia' spese e vanno rendicontate a parte. Riconcilia il residuo del budget prima dei
lanci; questi tetti non lo ampliano. Se risorse, numero di job o costi necessari cambiano, prepara il nuovo preventivo prima di procedere.

Questa tranche comprende recupero E3 e audit corretto. **Il collaudo B da 5.000 passi e il run completo R3 restano rinviati alla lettura
dei risultati e alla firma del nuovo criterio.** Peso della posa, M=280 e criteri gia' firmati restano quelli correnti; il dbg resta fallito.
