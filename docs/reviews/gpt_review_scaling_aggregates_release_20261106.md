# Review GPT del rilascio dei sei aggregati scaling

Commit Claude esaminato: `37632d18e0078069c54d131c9ba235891712c1b5`, ramo
`claude/scaling-300m-exec-20261106`; ancore Review 10 `d3976c3`, sezioni 3 e 5.
Review eseguita sul materiale pubblicato, senza SSH, invii, runner o modifiche
del worktree Claude. Ruoli: Claude esegue, GPT rivede.

**Verdetto: i sei comandi, i pin e il budget sono conformi. Prima dell'invio e'
obbligatorio controllare esplicitamente il successo della lettura della coda:
il PASS dell'helper attuale non basta in caso di errore di squeue.** E' ammessa
una verifica esplicita e documentata dell'operatore su exit=0 prima del conteggio;
preferibile la correzione minima dell'helper con regressione, a cura di Claude.
Non occorre un nuovo consenso di Simone. Restano tutti i controlli freschi del
piano; il confronto successivo non e' coperto da questa approvazione.

## Cosa e' verificato

- Piano SHA256 `abc489613dfbfddc5e33e2150854245bcd3b69292806953bafa622b91a22941c`.
  Dispatcher `c2bd7979699b2d70150cdfad9d2a6eca657615551ba5073630417c8c46cf14d5`,
  submitter `814da4bc66dbf18d69ee2ba332675fa0788ab25247d5f5460224cfd24d7e0d82`;
  codice numerico `aef665ba7ea66fc80291846487907bb3ef9107df` invariato.
- Snapshot Claude delle 10:28:22: 14 shard COMPLETED 0:0, 60 combinazioni
  cella/dataset/fold, 420 fit convergenti, firma `1ac8481e...`, nessun failed.json.
  Fit 14 `63635524` concluso alle 09:56:12, 4.759 s, 21 fit, shard hash
  `9887484ff2b1ac051cca6721bc280975284d60f8b4752af70b5f515f6bcdf5b7`.
- Mappatura: 30M P1=02+12, P2=04; 300M naturale P1=01+10+11+14,
  P2=06+07+08+09; proiettato P1=05+13, P2=03. Tutti i 14 shard usati una volta.
  Pin interni e impronte OOF coerenti con le ancore di Review 10 e la nuova
  lettura di fit 14 pubblicata da Claude.
- Sei argv e quoting coerenti: `/dev/stdin`, 2 CPU, 30G, 600 s, output nuovi
  distinti, CODE_DIR frozen e piano sonde `d2cc7da1...`, nessuna GPU.
- La v2 elimina solo la dipendenza tra aggregati: leggono shard distinti e non
  dipendono scientificamente l'uno dall'altro. Un fallimento non annulla gli altri
  job gia' inviati; non autorizza retry. Il manifest del confronto verra' pubblicato
  e rivisto solo dopo raccolta e validazione dei sei aggregati.
- Audit indipendente aggiuntivo: 58/58 controlli sul materiale pubblicato.
  Le prove del dispatcher pubblicate da Claude coprono sei casi validi e sette
  rifiuti; non sono una nuova esecuzione SLURM dell'allocazione.

## Rilievo operativo riprodotto

`final_stages.py` SHA256
`9533c82fa4e4cde93860fc62218bd1fd586f6919021ac2870b1b27330d60b9b8`,
funzioni `checks` e `cmd_check`: si contano le righe di `squeue_serial.stdout`
senza prima richiedere `squeue_serial.exit == 0`.

Riproduzione in `scaling_aggregate_queue_guard_20261106_check.py`, accanto a questa
review: estrazione AST delle sole due funzioni, fixture temporanee, nessun import
del CLI operativo o chiamata remota.

| Evidenza | Risposta dell'helper |
|---|---|
| Snapshot reale, nove job in coda | Rifiuto: meno di sei posti |
| Identico snapshot, solo squeue exit=1 e stdout vuoto | CANCELLO SUPERATO |

Una coda non leggibile non e' una coda vuota. Correzione minima consigliata:
rifiutare exit diverso da zero prima di qualsiasi parsing/conteggio; verificare
analogamente sacct per robustezza. La modifica riguarda solo il nuovo helper di
Claude, non i valutatori congelati, il dispatcher, il submitter o S8. Fino alla
correzione, l'operatore deve attestare esplicitamente exit=0 della lettura fresca
prima degli invii; un errore interrompe la fase, senza retry cieco.

## Conto cumulativo pubblicato da questa review

Il piano Claude riporta il costo dei fit e degli aggregati, ma non il cumulativo.
Lo completiamo senza cambiare l'inviluppo: parent sacct del suo snapshot per ridge
e fit; base accettata 28,373888889 prima della ridge, incluso checkout metadata.
Non sommiamo di nuovo il validate, gia' compreso nella base.

| Voce | Core-ore |
|---|---:|
| Base accettata prima della ridge | 28,373888889 |
| Ridge conclusa | 0,642222222 |
| 14 fit conclusi | 95,977777778 |
| Consumato cumulativo | **124,993888889** |
| Sei aggregati, incluso margine scheduler | 2,600000000 |
| Confronto, incluso margine scheduler | 4,400000000 |
| Checkout metadata residuo, incluso margine | 0,266666667 |
| Massimo previsto | **132,260555556 / 310** |

Il residuo e' sufficiente. Non e' un'autorizzazione a nuovi fit, dose o esperimenti.
Confermare prima dell'invio che nessun nuovo costo della linea sia intervenuto.

## Condizioni operative residue

Claude rilegge stato, pin, ledger e uscite; verifica successo dei comandi e sei
posti liberi effettivi prima della tranche, con stop al primo invio non accettato.
La decisione seriale riportata in `bf7f7a4` (due fit della linea 2 per volta dopo
i benchmark finche' il confronto non e' sottomesso) non modifica questo controllo
ne' l'ordine dei job gia' in coda. Nessuna risottomissione o cancellazione implicita.

Restano review separate per gli aggregati raccolti, il manifest/comando del
confronto e l'esito finale. S-1...S-4 di Claude non si applicano alla numerica
congelata in questo giro; le verifiche manuali richieste da Review 10 restano.
La lettura scientifica della posa pubblicata separatamente e' parziale: nessuna
raccomandazione S8 complessiva e nessun via a 19.618.
