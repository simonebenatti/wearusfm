# Review Claude: rilettura mirata della revisione v3 della proposta TCN (19/11/2026)

Oggetto: la revisione di GPT della proposta «TCN ispirata a TEMPONet con ramo lineare, encoder congelato», ID
`pose-temponet-frozen-encoder-readout-20261117-v3`, sul ramo `codex/pose-temponet-proposal-20261117`. Contenuti in
`455b3220256990ef20f97756aeda845334e163c5`, consegna con gli hash in `0e56c8b1c67f8257e800f6cd51bafacac28971b4`, base del diff
`3e007c841c97da5793ed4aac6cfceb41c99a00ee`. Risponde alla mia review `ee118a25`
(`docs/reviews/claude_review_pose_temponet_revision_20261118.md`).

Richiesta: due messaggi di GPT incollati da Simone il 19/11, coerenti fra loro. Chiedono la rilettura delle parti cambiate, la
verifica di tre punti (calibrazione con `y*` e `y`, lettura della MLP, tolleranza del bordo inferiore del LR con S-2), la chiusura
di OB-1...OB-5 e delle minori, senza job e senza un nuovo giro di agenti. È il passo fissato nella sezione 7 della `ee118a25`.

Metodo: letti la risposta `docs/reviews/pose_temponet_response_20261119.md`, il prompt, il protocollo (P) e `proposal.json` (J) a
`455b3220`, le voci nuove di `docs/decisioni.md` (DEC) e `CLAUDE.md`, `analytic_checks.json`, `document_checks_20261119.json` e
`delivery_pin_20261119.json`. Ho rifatto lo SHA-256 di protocollo, proposta e risposta: coincidono col pin e sono uguali a
`455b3220` e `0e56c8b1`. Ricalcolati 15 × 121 + 121 = 1.936 combinazioni, (1 − 1e-2 · 0,01)^9000 = 0,4066 e 89,8/90,8 = 98,9%.
Nessun agente, nessun job, nessun accesso al cluster. Punta del ramo di GPT ricontrollata prima del commit: `0e56c8b1`.

## Esito

**Convergenza documentale sul protocollo v3.**

Nessun blocco residuo. BL-1, BL-2 e BL-3 sono chiusi. GPT ha scelto le alternative ammesse dalla review, e ciascuna chiude il suo
controesempio. OB-1...OB-5 e le sei minori sono chiuse. Restano tre ritocchi di testo da fare senza un nuovo giro (sezione 3), più
tre facoltativi.

La convergenza riguarda il disegno. Non certifica codice, test, benchmark o gate. Prima degli invii restano la review del pacchetto
operativo e i gate meccanici, come dicono P §8 e P §10.

## 1. I tre punti richiesti

**1. Controllo positivo e calibrazione con `y*` e `y` (BL-1).** Chiuso.
- L'ancora è una ridge L0 pesata in forma chiusa (P:90; J `positive_controls.calibration.ridge`):
  - somma delle MSE per finestra più `alpha·‖beta‖²`, intercetta libera, alpha non riscalato da 10 a 20 utenti;
  - alpha scelto sulla MSE pooled dei due held inner, a pari merito il più piccolo;
  - griglia 10^-2...10^4, al più tre decadi per lato, bordo finale irrisolto con la conseguenza scritta.
- Le grandezze sono tre e distinte (P:92; J `calibration.quantities`):
  - E_cal: ridge fittata su `y`;
  - E_cleanLS: WLS fittata su `y*`;
  - E_oracle: `y*` stesso.

  Tutte e tre si valutano su `y` osservato. Solo `pose_var` e la soglia delle finestre dinamiche usano `y*` (P:59, P:92; J
  `calibration.threshold`, `fit.internal_dynamic_threshold.source`). Banda inner e outer, margine del 50% e distanza dall'oracolo
  usano tutti E_cleanLS (P:96, P:100; J `temporal_band_uses`). È la tabella della `ee118a25`, con lo stimatore dichiarato (WLS).
- Controllo lineare (P:99; J `gates.linear`):
  - la ridge deve recuperare almeno il 90% del margine costante→oracolo, altrimenti «non diagnostico» prima dei fit neurali;
  - ogni testa deve recuperare almeno il 90% del margine costante→ridge.

  È l'alternativa che proponevo, «cancello riferito alla forma chiusa». Nel controesempio una L0 che regolarizza come la WLS ha
  89,8/90,8 = 98,9% e passa. Prima avrebbe avuto «ricetta non ammessa».
- Temporale: a fino a 3 e sigma fino a 6 con passo 0,05, per 1.936 combinazioni per direzione outer. La quota di T non leggibile in
  modo istantaneo e la quota recuperata da L13 si misurano prima dei fit. Se nessuna coppia entra, «controllo non calibrato» e stop
  (P:96; J `calibration.no_pair`). È una regola fissata prima delle misure; l'etichetta è vera, non attribuisce colpe alla ricetta.
- Accolgo la precisazione di GPT: la WLS su `y*` toglie il rumore dal fit, ma non garantisce la stessa banda fra inner e outer.
  Nella `ee118a25` avevo scritto «con E_LS* il passaggio da inner a outer non sposta la banda»: era troppo forte. Il controllo
  outer conservato (P:100) è la risposta giusta.

**2. La MLP nel controllo temporale (BL-2).** Chiuso.
- La MLP è descrittiva, fuori dalle condizioni che devono passare. Una MLP vicina o migliore della TCN non ferma la catena; dice
  però che questo controllo non qualifica il vantaggio specifico del contesto temporale. Non c'è nessuna soglia di vicinanza.
- Lo stesso testo sta in tre punti: P:102, J `gates.temporal.MLP` (l'unica testa del cancello è `TCN_post`) e DEC:6778 («MLP
  temporale solo descrittiva»). Il prompt e la risposta dicono lo stesso.
- Nel controesempio della review (TCN 0,870, MLP 0,869) le due implementazioni fedeli ora danno lo stesso esito: la catena
  prosegue, con la lettura qualificata.

**3. Convergenza, bordo inferiore e S-2 (BL-3).** Chiuso.
- Sul bordo inferiore esteso l'allarme scatta solo se `(MSE_vicino − MSE_minLR)/MSE_vicino > 0,005`. Si usa il migliore checkpoint
  MSE ammesso del LR immediatamente più alto, sulla stessa popolazione pooled. Il caso zero e i valori non finiti sono definiti
  (P:69; J `fit.extensions.unresolved`, `lower_edge_relative_mse_gain_threshold`; DEC:6779). Vale per tutte le teste, nei
  controlli e nei fit reali. Restano allarmi il bordo superiore dopo l'estensione, il minimo nelle ultime 5 epoche dell'orizzonte
  massimo e la scelta al primo quarto di epoca.
- Controesempio della review: 3e-5 batte 1e-4 di 0,0001 su 0,7269, cioè lo 0,014%. Non scatta più l'allarme e L0 resta ammessa.
- S-2 resta leggibile come compromesso fra livello (MSE, errore assoluto) e dinamica (E_dyn, capacità di seguire il movimento):
  - è definito da distanza > 5 epoche e perdita ≥ 2% rispetto al minimo pooled di E_dyn fra i checkpoint ammessi del LR scelto,
    passo 0 escluso;
  - non è un allarme e non toglie etichette né ammissibilità; un allarme vero sulla stessa testa prevale;
  - su L0, MLP o L13 qualifica il confronto come prestazione sotto la selezione MSE comune; su L13 il contrasto 3 non diventa una
    prova d'impossibilità; sulla TCN un nullo resta relativo alla ricetta (P:72, P:134; J `fit.objective_tradeoff`,
    `evaluation.candidate.objective_tradeoff`).

## 2. Correzioni obbligatorie e minori

| Voce | Esito | Dove |
| --- | --- | --- |
| OB-1 | Chiusa: un allarme non ferma da solo una testa che passa. Per la testa che non passa: con allarme «ottimizzazione non risolta», senza allarme «ricetta non ammessa». Ordine degli stati fisso; l'allarme prevale su S-2 | P:104, P:72; J `positive_controls.state_order`, `failure`, `fit.optimization_alarm_consequences.controls` |
| OB-2 | Chiusa nel protocollo e nel JSON: «compromesso S-2», `is_optimization_alarm: false`. Elenco degli allarmi completo in P e J; in DEC manca ancora (sezione 3, punto 1) | P:72, P:104; J `fit.objective_tradeoff`; DEC:6779 |
| OB-3 | Chiusa con l'opzione (a). Il veto è solo `deterioramento_pratico` sui contrasti 2 e 3; il caso misto ha il nome neutro, mai «non peggiore»; S-2 sul comparatore è nominato. J:104 è diventato un rinvio a `candidate` | P:132, P:134; J `evaluation.candidate`, `practical_readout_recommendation`; DEC:6779; §10 |
| OB-4 | Chiusa con la conseguenza direzionale. La candidata richiede L0, MLP e L13 senza allarmi; un allarme della sola TCN-post non la esclude; la catena prosegue con i soli allarmi di ottimizzazione | P:114; J `fit.optimization_alarm_consequences.real`; §10 righe «Positivo» e «Negativo/irrisolto» |
| OB-5 | Chiusa per la riga «Positivo» (E_dyn ≤ 0,7630564046643697 necessario, MAE/livello descrittivi, casi misti e S-2) e per la riga «Portata»; spazi della tabella sistemati. Nella riga «Criteri» manca la conseguenza (sezione 3, punto 2) | §10 |
| `architettura-2` | Chiusa: denominatore `‖Xw‖₂·‖yw‖₂`, test CPU con RHS esatto | P:76 |
| `architettura-5` | Chiusa | P:134; J `descriptive_reading.width_caution` |
| `criteri-2`, `consegna-4` | Chiusa nei due versi, compreso «1 non passa, 4 e 5 passano» | P:137; J `descriptive_reading` |
| `criteri-3` | Chiusa | P:129; J `outcome_priority[2]` |
| `criteri-5`, `consegna-2` | Chiusa: `technical_failure` con cause e conseguenze; R-1 fallito ferma la fase | P:112, P §8; J `fit.technical_failure`, `historical_reference_gate.failure` |
| `criteri-6`, `S-E` | Chiusa: uguaglianza dopo il cast a float64, i cinque conteggi con i nomi dei campi della ricevuta | P §8; J `historical_reference_gate` |

Note della sezione 5 della `ee118a25`: c'è una riga di disposizione per ciascuna. Come previsto (`ee118a25`, sezione 7, punto 3),
le ricontrollo alla review del pacchetto operativo. Ne ho viste alcune già chiuse: `costo-2` (la chiave «superseded» di
`analytic_checks.json` è corretta), `consegna-9` (controllo rinominato), `ricetta-7` (0,4066), `ricetta-10`, `costo-4`
(`47ca96e`), `costo-6` e `controlli-7`.

## 3. Da correggere senza un nuovo giro

Non bloccano: P e J sono completi e coerenti, e nessuno dei tre punti cambia un'etichetta. Vanno corretti insieme al pacchetto
operativo, e li ricontrollo lì.
1. **DEC, elenco degli allarmi (OB-2).** DEC:6779 dice «altri allarmi invariati», ma la voce precedente (DEC:6747) chiamava ancora
   S-2 «Allarme» ed elencava solo il bordo finale. Va scritto una volta l'elenco completo, come in P:69:
   - minimo MSE o E_dyn nelle ultime 5 epoche dell'orizzonte massimo;
   - checkpoint scelto al primo quarto di epoca;
   - bordo superiore dopo l'estensione;
   - bordo inferiore esteso solo oltre lo 0,5%.
2. **§10, riga «Criteri» (OB-5).** Aggiungere la conseguenza delle due scelte, che oggi sta solo in P:124 e P:139: «con 15 utenti la
   copertura può essere più bassa; senza bootstrap a due livelli il "no" resta condizionato ai tre semi». La tabella è quella che
   firma Simone.
3. **P:135, riferimento del delta di MAE e livello (`criteri-7`).** «delta candidato meno controllo positivo» si può leggere come
   il controllo positivo sintetico. Scrivere «delta = candidato − comparatore; un delta positivo è un peggioramento stimato», come
   dice già J `evaluation.secondary_worsening`.

Facoltativi:
- P:86 e P:90, «non calibrabile»: usare il nome dello stato, «controllo_non_calibrato» (`consegna-3`, `controlli-8`).
- J `fit.objective_tradeoff.consequence`: nomina solo L0, MLP e L13; aggiungere la lettura della TCN di P:72 («un nullo resta
  relativo a questa ricetta»).
- Parole ancora attaccate fuori dal §10, per esempio P:53 «hanno160 patch» e P:65 «a150 epoche, semeCV0».

## 4. Che cosa non ho verificato

- Codice, test, benchmark, contesto numerico e gate: non esistono ancora in questa consegna. Sono la review del pacchetto operativo
  (ramo `codex/pose-temponet-implementation-20261119`), prima degli invii, e restano le verifiche della sezione 6 della `ee118a25`.
- Le parti che già convergevano nella `ee118a25`: non le ho rilette, come previsto.
- I numeri di risorse e saldo del §9: sono storici e li verifica l'operatore nel preflight.

## 5. Prompt per GPT

```text
Rilettura Claude della revisione v3 della proposta TCN (consegna 0e56c8b1, contenuti 455b3220, base 3e007c8):
**Convergenza documentale sul protocollo v3.** Nessun blocco residuo.
Review: docs/reviews/claude_review_pose_temponet_v3_20261119.md, ramo claude/temponet-db8-20261109 (commit nel messaggio di Simone).

Chiusi BL-1, BL-2, BL-3, OB-1...OB-5 e le sei minori. I tre punti richiesti sono verificati:
- calibrazione: E_cal dalla ridge su y, E_cleanLS dalla WLS su y*, oracolo y*, tutto valutato su y; solo pose_var e soglia
  dinamica usano y*;
- MLP del temporale descrittiva, con lo stesso testo in P, J e DEC;
- tolleranza dello 0,5% sul bordo inferiore, e S-2 come compromesso, mai allarme.
Accolgo la tua precisazione su E_cleanLS fra inner e outer: la mia frase della ee118a25 era troppo forte, il controllo outer
conservato è giusto.

Da correggere senza un nuovo giro, insieme al pacchetto operativo (li ricontrollo lì):
1. DEC: dopo «altri allarmi invariati» scrivi una volta l'elenco completo, come in P:69: minimo nelle ultime 5 epoche
   dell'orizzonte massimo, scelta al primo quarto di epoca, bordo superiore dopo l'estensione, bordo inferiore oltre lo 0,5%.
2. §10, riga «Criteri»: «con 15 utenti la copertura può essere più bassa; senza bootstrap a due livelli il "no" resta
   condizionato ai tre semi».
3. P:135: «delta = candidato − comparatore; un delta positivo è un peggioramento stimato», come in J.
Facoltativi: «controllo_non_calibrato» in P:86 e P:90; la lettura della TCN di P:72 anche in J objective_tradeoff.consequence;
spazi fra parole attaccate fuori dal §10.

La convergenza riguarda il disegno: non rilascia invii. Prima dei lanci restano la review del pacchetto operativo (codice, test,
piano eseguibile) e i gate. Nessun job né deployment da parte mia; l'operatore resta 2-TCN head launcher.
```
