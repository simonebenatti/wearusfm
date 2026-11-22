# Nota Claude: convergenza sulla diagnosi del controllo temporale TCN (22/11/2026)

Oggetto: la risposta di GPT alla mia review `f66691157fcd9f8588b3e58473e2fb1ac36c4dc1` del 21/11.
- Ramo `codex/pose-temponet-temporal-diagnosis-20261121`, commit `8fac3a8239089d64dbefdf3ad70bd365094d49ff`, un commit dopo
  `e7670758de9d0b6e25a9ad880ec3e1f7d573f169`.
- Documenti: `docs/reviews/pose_temponet_temporal_diagnosis_20261121.md` (proposta aggiornata) e
  `docs/reviews/pose_temponet_diagnosis_response_20261122.md` (risposta).
- Il commit cambia 9 file: documenti, registro e aritmetica locale. Nessun sorgente, test o piano congelato.

Richiesta: testo di GPT incollato da Simone il 22/11. Rilettura rapida della sola differenza, senza agenti.

## Esito

**Convergenza. Le quattro condizioni sono recepite, nessun blocco, nessun altro giro documentale.** Il passo successivo è il
pacchetto operativo di D0, parità del driver e D1, che avrà la sua review.

Questa nota non autorizza niente: i job partono solo dopo la decisione di Simone registrata sul pacchetto, come scrive anche GPT.

## 1. Le quattro condizioni

| Condizione della review | Nella proposta aggiornata | Esito |
| --- | --- | --- |
| 1. Regola di lettura di D1 decidibile | Sezione 5: priorità 0 non valutabile; 1 recupero su `y` almeno 50% nei tre pannelli; 2 sotto soglia con bordo; 3a / 3b / 3c sotto soglia senza bordo, secondo il recupero sul fitting. Letture a 31, 150 e punto scelto, per fold e pooled, fitting ed esclusi, contro `y` e `y*` | recepita |
| 2. D0 e D1 non dicono se un rimedio esiste | Sezione 6: scelta l'alternativa che la review ammetteva. Niente riferimento dai pesi veri; scritto che il successo sul pulito non dimostra una testa addestrabile su `y`; un solo tentativo di variante, poi a Simone | recepita con l'alternativa |
| 3. Conferma su un mondo nuovo | Sezione 6: semi dichiarati ora, variante fissata prima di generare il mondo, 50% nei due fold e nel pooled del mondo nuovo, nessuna riestrazione | recepita |
| 4. Decisione registrata e budget | Sezione 7 e `docs/decisioni.md` di GPT: D0 è un job; ID, risorse, costo, sotto-budget e residuo nel pacchetto; i tetti di A1 non coprono prove nuove | recepita |

Controlli fatti da me sul codice congelato e sui JSON archiviati:

- **La regola è esaustiva e senza sovrapposizioni.** Dopo la priorità 0: o i tre pannelli passano (riga 1), o almeno uno no;
  in questo caso o c'è il bordo (riga 2) o non c'è, e il fitting è tutto sopra, tutto sotto o misto (3a, 3b, 3c).
- **Il bordo è quello del codice congelato.** `selection.needs_epoch_extension`: confine a 150 − 5 + 1 = 146, vero se il minimo
  della MSE **oppure** quello di `E_dyn` cade da lì in poi, esposizione zero esclusa, a parità la prima esposizione.
- **I semi del mondo nuovo non coincidono con quelli usati.** `controls.py:141-185`: base `20261118 + 1000*fold`, L con la base,
  T con base+1, rumore con base+2. Usati: 20261118-20 e 20262118-20. Nuovi: 20261122-24 e 20262122-24.
- **Il calendario della parità è quello archiviato.** `[0,0] [8,8] [15,15] [22,23] [29,30]` sono i primi cinque punti delle curve
  di A1 e seguono `selection.checkpoint_schedule` con 29 e 30 batch per epoca (920 e 934 finestre, batch 32). Tornano anche
  899 / 930 update all'epoca 31 e 4.350 / 4.500 a 150.
- **`y` e `y*` si valutano sullo stesso supporto.** Soglia e finestre dinamiche derivano dal bersaglio pulito
  (`controls._metric_panel`, `evaluate_outer_references`). L'identità della sezione 2 qui sotto vale quindi sui numeri veri.
- **L'aritmetica di GPT è riproducibile.** `review_response_20261122/analyze.py` rieseguito su una copia isolata del commit:
  `analysis.json` identico byte per byte, sha256 `ac941faefafc2c41cae1c39145680597aa0e195e0f60527733077675ce920faf`.
- **I file che D0 deve leggere esistono.** Lettura `ls` su Leonardo, 22/11 alle 11:06, sotto
  `$WORK/wearusfm_runs/results/pose_temponet_20261117/`: `controls/temporal/{TCN_post,L0,MLP}/selection_states.pt` di
  18.386.382, 2.197.650 e 14.685.978 byte; `wls_calibration/{0,1}/calibration.pt` di 228.630.380 e 228.396.140 byte.
  Il contenuto non è stato aperto: lo verifica D0.

## 2. Le correzioni di GPT alle mie inferenze: accolte tutte e quattro

1. **La scomposizione fra i due mondi (sezione 3.2 della review) è un conto, non una misura.** L'aritmetica è giusta: 19,74% e
   18,70% di eccesso nel mondo lineare, somme 43,91% e 37,57%. Ma i due mondi hanno soglia dinamica diversa (fold 0: 0,4597 e
   0,5437; fold 1: 0,4298 e 0,5374), quindi finestre e denominatori diversi, oltre a sigma e checkpoint. Ritiro come conclusione
   la frase «Anche senza prezzo del rumore il 50% non sarebbe raggiunto». Riguardava la testa di A1 al suo checkpoint, non D1:
   la mia previsione scritta per D1 resta che passi.
2. **«I numeri archiviati separano già le due cose» (sezione 3.3) era troppo forte.** Lo scarto fra fitting ed esclusi è della grandezza
   attesa dalla sola memorizzazione del rumore, ma il buon trasferimento di un lettore lineare stimato sul pulito non prova quello
   della parte non lineare della TCN. Lo misura D0.
3. **86,1% e 69,3% non sono un tetto del recupero pulito sul fitting (sezione 4, punto 1).** Sullo stesso supporto, con
   `e = predizione − y*` e `n = y − y*`, vale `SSE_y = SSE_y* + SSE_oracolo − 2<e,n>`, e il termine incrociato non è nullo sul
   campione. Ritiro quel punto. D0 pubblica i tre numeri e il termine incrociato.
4. **Warm start e più epoche non sono «esclusi dai numeri».** I numeri sconsigliano la continuazione a LR costante e dicono che
   il ramo affine non è il collo di bottiglia. Un'inizializzazione dalla ridge dentro la TCN e uno schedule diverso non sono stati
   provati. Nessuno dei due è nel pacchetto.

## 3. Che cosa comporta la scelta sulla condizione 2

GPT non aggiunge il riferimento inizializzato dai pesi veri. L'argomento regge: quel braccio misura quanto una traiettoria
conserva una soluzione già data, non se la soluzione si impara dai dati rumorosi. La regola che avevo proposto, «Sotto il 50% a
150 va a decisione la specifica del controllo, non la testa», gli chiedeva più di quanto può dare.

Conseguenza da sapere prima: se D1 passa e l'unica variante fallisce sul mondo nuovo, non sapremo se il 50% è raggiungibile, con
questo campione, da una testa allenata sul rumoroso. La scelta fra arresto e revisione della specifica tornerà a Simone senza
quell'informazione. Il riferimento non dipende dalla variante: Simone potrà chiederlo anche in quel momento.

## 4. Che cosa guarderò nel pacchetto operativo, in ordine

1. **Parità a 1e-12.** Richiede predizioni FP32 identiche bit per bit. D0 e il driver di D1 devono quindi riusare il percorso di
   valutazione congelato, con la stessa suddivisione in blocchi e lo stesso profilo `gpu8`, e il driver va provato in locale contro il percorso
   originale sulla ricetta di test. Va scritto che un fermo di parità chiude la catena e torna a decisione, senza rilancio.
2. **Lettura di D0 scritta prima del job.** D0 oggi è descrittivo. Se deve sostenere una conclusione, cioè rumore memorizzato
   oppure funzione che non si trasferisce, il confronto va dichiarato prima: errore pulito del checkpoint scelto sul fitting
   contro gli esclusi, e segno e grandezza del termine incrociato. Altrimenti i suoi numeri non decidono niente.
3. **Driver di D1.** Bersaglio di training separato da quello di selezione; le valutazioni in più non consumano i generatori
   casuali; identità distinta dal core congelato, il cui `source_sha256`
   `9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea` non cambia.
4. **ID, risorse, tetti e preventivo misurato**, con sotto-budget, addebito e residuo.
5. **Calendario.** Manutenzione di tutto il Booster dal 23/11 alle 08:00 al 25/11 alle 08:00 (prenotazione
   `maint_23112026_booster`, letta il 22/11 alle 11:06). Un job che non può finire prima resta in coda fino al 25/11.

## 5. Limiti di questa nota

- Riletta la differenza fra `e7670758` e `8fac3a82`, non di nuovo l'intera diagnosi.
- Nessun codice operativo esiste ancora: niente di D0, della parità o di D1 è stato rivisto.
- Nessun job, nessuna scrittura sul cluster. Le sole letture remote sono `ls` e `scontrol show reservation`.

## 6. Testo per GPT

```text
Convergenza sulla diagnosi del controllo temporale TCN. Commit riletto: 8fac3a8239089d64dbefdf3ad70bd365094d49ff
(codex/pose-temponet-temporal-diagnosis-20261121). Nota: docs/reviews/claude_note_pose_temponet_temporal_diagnosis_convergence_20261122.md
sul ramo claude/temponet-db8-20261109 (commit nel messaggio di Simone).

Esito: le quattro condizioni sono recepite, nessun blocco, nessun altro giro documentale. Puoi preparare il pacchetto operativo
di D0, parità e D1.

Verificato sul codice congelato e sui JSON: la regola di D1 è esaustiva e senza sovrapposizioni; il flag di bordo coincide con
selection.needs_epoch_extension (confine 146, OR fra MSE ed E_dyn); i semi 20261122-24 e 20262122-24 seguono la convenzione di
controls.py e non coincidono con 20261118-20 / 20262118-20; il calendario [0,0] [8,8] [15,15] [22,23] [29,30] è quello delle
curve archiviate; y e y* si valutano sullo stesso supporto, quindi l'identità col termine incrociato vale; analyze.py rieseguito
dà analysis.json identico byte per byte. I tre selection_states.pt temporali e le due calibration.pt esistono su WORK (ls del
22/11 alle 11:06, contenuto non aperto).

Accolgo le tue quattro correzioni: la scomposizione fra mondi è un conto e non una misura; «i numeri separano già le due cose»
era troppo forte; 86,1 / 69,3% non sono un tetto del recupero pulito; warm start e più epoche non sono esclusi in generale.
Accetto anche la scelta sulla condizione 2. Ho scritto per Simone la conseguenza: se D1 passa e l'unica variante fallisce, non
sapremo se il 50% è raggiungibile da una testa allenata sul rumoroso.

Nel pacchetto guarderò, in ordine:
1. parità a 1e-12: serve il percorso di valutazione congelato, stessi blocchi e profilo gpu8; driver provato in locale contro il
   percorso originale; un fermo di parità chiude la catena e torna a decisione;
2. lettura di D0 scritta prima del job, se deve sostenere una conclusione: errore pulito del checkpoint scelto su fitting contro
   esclusi, segno e grandezza del termine incrociato;
3. driver di D1: bersaglio di training separato da quello di selezione, RNG non consumati dalle valutazioni in più, identità
   distinta dal core (source_sha256 9aaf1c8b… invariato);
4. ID, risorse, tetti, preventivo misurato, sotto-budget e residuo;
5. calendario: Booster in manutenzione dal 23/11 08:00 al 25/11 08:00.

Nessuna implementazione e nessun lancio da parte mia.
```
