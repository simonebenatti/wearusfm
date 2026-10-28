# Prompt per Claude — review della proposta E3 sul checkpoint B

Ti chiedo un secondo parere scientifico sulla proposta di GPT prima di procedere. **Questa richiesta e' solo di review:**
leggi codice, protocolli e risultati gia' disponibili; non implementare correzioni, non estrarre dati, non addestrare sonde
e non lanciare esperimenti o job. L'obiettivo e' capire se convergiamo sul prossimo passo, con eventuali modifiche motivate.

Il riferimento da esaminare e' **9142ca66e2f49c87bdba11d4627dcd79ed66bf90**, pubblicato sul ramo
`origin/codex/r3-long-collaudo-20261028`. Aggiorna i riferimenti di origin e leggi dal tuo worktree, anche con `git show`:
non occorre fare merge e non modificare il worktree GPT. La proposta e' una direzione scientifica, non ancora un protocollo
E3 B eseguibile con tutti i dettagli congelati.

Leggi come base:

- `docs/reviews/sintesi_scientifica_20261028.md` — diagnosi aggiornata e proposta successiva.
- `docs/reviews/dinamica_b_results_20261028.md` e `docs/dinamica_b_20261028.md` — ultima analisi e metodo fissato prima del calcolo.
- `docs/reviews/b_e1e2_results_20261028.md` e `docs/reviews/collaudo_b_results_20261028.md` — inferenza su val e collaudo B.
- Per i dettagli: `results/infra/pilot_posa/b_dynamics_20261028/final/report.json`, relativo README e analisi indipendente;
  `src/wearusfm/harness/pose_eval.py`, `src/wearusfm/model/pose_head.py`, `scripts/pose_eval.py`, `scripts/pose_dynamics.py`.
- Per confronti e regole precedenti: `docs/emendamento_pilot_posa_v1.md`, `docs/pilot_posa_v1.md`,
  `docs/reviews/pose_scientific_assessment_20261028.md` e ultime voci di `docs/claude_GPT_shared`.

Il punto di partenza e' questo: B a 5.000 passi ha superato il suo criterio sul train (MSE −11,713%). Sui 15 utenti esclusi
dal training E1 migliora la MAE dinamica del 6,245%; E2 ha gap positivo in 14/14 utenti valutabili. La nuova scomposizione
mostra sulle dinamiche MSE centrata −7,252%, IC95 [6,083%; 8,752%], con beneficio in 14/15 utenti. Tuttavia la varianza
predetta e' solo il 6,324% del target, la correlazione centrata e' 0,270 e sulle finestre poco dinamiche l'errore centrato
peggiora del 5,945%, in 15/15 utenti. La centratura concede una correzione ideale del bias per finestra: non e' prestazione
operativa. Non confrontare direttamente percentuali di metriche/supporti diversi.

**Proposta GPT:** prima di R3 completo, fare E3 sullo stesso **student B a 5.000**, congelato e senza maschera come la cache
val esistente. Estrarre soltanto il sottoinsieme train necessario, fittare una ridge e selezionare gli iperparametri
esclusivamente fra utenti train. Su val confrontare ridge e testa esistente, includendo la stessa separazione fra livello
medio e dinamica, non soltanto MAE totale. Nessun nuovo addestramento dell'encoder. La domanda e': c'e' informazione sul
movimento accessibile a questo lettore che la testa attuale non recupera?

Vorrei la tua review su sei punti:

1. **Utilita' del prossimo esperimento.** E3 B e' davvero il controllo piu' utile per decidere come procedere, oppure c'e'
   un controllo piu' semplice o una ragione per proseguire direttamente il training? Indica quale decisione concreta
   cambierebbe con un esito favorevole, simile o peggiore. Un guadagno della ridge non dimostra un unico collo di bottiglia;
   un risultato debole non dimostra un limite assoluto dell'encoder.
2. **Confronto con la testa.** La testa usa `LayerNorm -> Linear` sulla media dei K latenti; E3 esistente fitta sui latenti
   prima della LayerNorm. E' sufficiente per la domanda, o serve come confronto primario la stessa trasformazione della
   testa congelata? Valuta anche intercetta/baseline stimata sul medesimo subset train, pesi per finestra/istante e differenze
   di supporto fra ottimizzazione della testa e ridge. Proponi il disegno minimo interpretabile, senza moltiplicare varianti.
3. **Separazione train/val e riuso delle cache.** Verifica dal codice le condizioni necessarie: stesso checkpoint e vista,
   statistiche e maschere coerenti, selezione del subset fissata, split interno per utenti, preprocessing stimato solo sul
   fit interno. Segnala eventuali limiti della cache float16, della selezione degli istanti o del trattamento dei giunti
   invalidi che cambierebbero il confronto. Distingui difetti dimostrati da controlli da fare prima dell'implementazione.
4. **Metriche e selezione.** Quale obiettivo interno e quale confronto primario fisseresti, dato che E3 legacy sceglie alpha
   su MAE standardizzata e qui interessa anche la dinamica? Valuta differenza appaiata ridge-testa sullo stesso supporto,
   IC ricampionando utenti, termini medio/centrato/totale, varianza e correlazione, mantenendo il secondario poco dinamico.
   Evita che un guadagno sull'intercetta o sulla postura media venga letto come migliore inseguimento del movimento.
5. **Limiti della conclusione.** A 5.000 passi la lettura resta descrittiva: non applicare come verdetto finale le soglie
   firmate per 49.046, ne' inventare un nuovo cancello dopo i risultati. Val e' gia' stata esaminata per costruire questa
   diagnosi: chiarisci il limite esplorativo e come evitare tuning ripetuto su questi 15 utenti. R0 confronta ricette;
   l'attribuzione alla loss di posa richiede R3c abbinato, perche' R0 cambia anche JEPA.
6. **Perimetro minimo e fattibilita'.** Se concordi, elenca solo cio' che manca per un protocollo eseguibile: parti riusabili,
   modifiche indispensabili, verifiche necessarie e preventivo indicativo basato sui job precedenti. Non assumere che
   l'estrazione duri quattro minuti senza verificarne la base documentale. Valuta se R0 o J2 bis siano necessari adesso
   oppure possano restare successivi. In questa consegna limitati alla proposta: nessuna implementazione o esecuzione.

Restituisci un verdetto chiaro: **concordo**, **concordo con modifiche**, oppure **propongo un altro passo**, con motivazione.
Distingui le correzioni indispensabili dai miglioramenti facoltativi; per ogni rilievo indica il riferimento e l'effetto
sull'interpretazione. Non serve riaprire tutta la diagnosi se un punto e' gia' risolto dai dati, ne' convergere per cortesia.

Salva la review in `docs/reviews/claude_review_proposta_e3_b_20261028.md`, registra l'esito nelle due sezioni di
`docs/claude_GPT_shared` sul tuo ramo, committa e pusha. Riporta a Simone verdetto, modifiche proposte e commit.
GPT confrontera' la review con la proposta prima della fase esecutiva; nessun esperimento parte da questa richiesta.
