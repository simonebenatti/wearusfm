# Prompt per ChatGPT: revisione scientifica indipendente della diagnosi del pretraining e della seconda ondata (R3)

Preparato da Claude il 26/10/2026 su richiesta di Simone («la cosa migliore sarà usare entrambi e far rivedere il lavoro di uno agli altri,
preparami un prompt per chatgpt poi ti dico cosa risponderà»). Simone lo incolla a ChatGPT e riporta la risposta a Claude; Claude risponde nel
registro condiviso. Il testo da incollare comincia sotto la riga.

---

Sei il revisore scientifico indipendente del progetto WearUsFM (modello di fondazione per sEMG, pretraining su Leonardo). Dal 16/10/2026 i
ruoli sono: Claude fa la parte scientifica e tecnica, tu fai le verifiche e sei l'integratore su `main`. Simone ha deciso oggi (26/10) che i due
agenti si rivedono il lavoro a vicenda. Questa richiesta e' una **seconda opinione scientifica completa** sulla direzione presa da Claude dopo i
risultati nulli delle ultime due settimane, non solo una verifica di conformita'. Nessun job va lanciato; non modificare i file di Claude: la
tua risposta e' un documento di review piu' le voci del registro, e ogni correzione che chiedi va nel registro (sezione 2) con una prova o una
regressione, come da accordo.

## Dove sta il materiale

Ramo di Claude `worktree-claude-training-engine`, commit `820c91b`, pubblicato su GitHub (`git fetch origin worktree-claude-training-engine`).
Leggi in quest'ordine:

1. `docs/claude_GPT_shared`, voci di Claude del 25 e 26/10/2026 (sezioni 1 e 2): il racconto dei fatti.
2. `docs/diagnosi_pretraining_20261023.md` e `docs/sintesi_pretraining_20261024.md`: la diagnosi («il readout non e' il collo di bottiglia;
   la taglia non lo e' con questo obiettivo; il JEPA non aggiunge nulla di lineare alle ancore»), con `results/readout_canale_20261023/`.
3. `docs/pilot_obiettivi_v1.md` e `results/pilot_obiettivi_20261024/README.md`, `criterio.json`, `provenienza_criterio.json`: la prima ondata
   (R1 bersaglio a, R2 bersaglio per regione, R5 finestra 150 ms contro R0 = A): nessun braccio passa, letta con la regola firmata il 24/10.
4. `docs/pilot_posa_v1.md`: la seconda ondata, R3 = ancore + testa di posa su emg2pose senza JEPA, con controllo R3c (ancore-only a 2
   epoche), FIRMATA da Simone il 26/10 (regola §5, deroghe a v10 §6.1 e §6.5/§3.7, soglie). Con `docs/decisioni.md` (le tre voci del 26/10:
   proposta, firma, criterio del pilot), `docs/fatti_da_verificare.md` riga 28 (fatti su emg2pose raccolti dal fact-checker),
   `results/infra/pilot_posa/preventivo.json` e `piano.json`.
5. Il codice di R3 (commit d9e8fa6, 1bf33fd, b41b40c, 71c0a93, dfb6d3e, 820c91b): `src/wearusfm/ingest/emg2pose_pose.py`,
   `src/wearusfm/data/pose_targets.py`, `src/wearusfm/model/pose_head.py`, i diff di `training/jepa.py`, `training/run.py`,
   `data/pretraining_loader.py`, `scripts/train_jepa.py`, `scripts/arms_collaudo.py`, e i test `tests/cpu/test_pose_*.py`,
   `test_ingest_emg2pose_pose.py`, `test_readout_reference.py`, `test_arms_collaudo.py`. Il `critico` di Claude ha gia' fatto tre review del
   protocollo (8+9, 3+6, 1+5 rilievi) e sta rivedendo il codice: non ripetere quelle, ma leggile nel registro e di' se condividi le chiusure.

Riferimenti di scala: S della validazione interna, P1 = NinaPro DB2/DB3/DB6 cross-soggetto, P2 = EPN-612 + UCI-EMG; baseline classica
(Hudgins) 14,30 / 67,48; A storico 18,34 / 81,92; R0 del pilot 18,84 / 81,22; R1 18,08 / 79,95; R2 18,57 / 81,77; R5 18,31 / 81,58. La S su P1
e' ferma intorno a 18-19 dal C1 (settimana del 16/10).

## Le domande, in ordine di importanza

**A. La diagnosi regge?** Claude sostiene, dai dati: (i) il readout medio non nasconde informazione (readout per canale `local`/`dec` non
migliora con la regola congelata); (ii) il rango effettivo non cresce da 30M a 300M; (iii) ancore-only = A in quattro letture e JEPA-only
collassa; (iv) tre varianti del JEPA non spostano nulla. Conclusione: l'informazione lineare che la sonda legge viene dai bersagli esterni (le
ancore: statistiche locali del segnale) e il termine JEPA e' peso morto. Quali spiegazioni alternative sono compatibili con gli stessi dati e
non sono state escluse? Per esempio: un tetto della sonda lineare cross-soggetto su questo corpus; rumore del compito P1 (DB3 amputati); la
fuga attraverso il filtro del loader (token visibili vicino alle regioni nascoste, dichiarata); durata del training (2 epoche; A piatto da 1 a 4);
la miscela dei dataset (emg2pose + emg2qwerty = 73% delle ore, polso Meta) che domina la rappresentazione. Per ciascuna: quale **singola prova a
basso costo** la discriminerebbe, con un costo in GPU-ora o core-ore.

**B. R3 e' la mossa giusta?** Dato il quadro, un bersaglio esterno alla scala del movimento (angoli articolari da motion capture, solo su
emg2pose) e' la scommessa con piu' evidenza, oppure metteresti prima altro? Candidati gia' sul tavolo: sonda non lineare (MLP) o fine-tuning al
posto della sonda lineare (cambia la misura, non il modello); mascherare prima del filtro; token da 100-200 ms sopra le patch; piu' dati o piu'
epoche; un altro obiettivo auto-supervisionato (contrastivo cross-soggetto, ricostruzione dell'inviluppo). Ordina con le ragioni, un costo
indicativo e che cosa ciascuno escluderebbe. Di' esplicitamente se secondo te il problema e' nel modello, nei dati, nella misura, o nella
scala, e con quale confidenza.

**C. Critica del disegno di R3** (`docs/pilot_posa_v1.md` §2-§6). In particolare: testa lineare a zero sulla media dei K latenti del backbone
per istante (la parte di `pool_p1p2` che il criterio legge); target su tutte le patch, visibili e nascoste, con ritardo zero (deroga al lead time
di v10 §3.7, motivata dal backbone bidirezionale); peso 1,0 sul target standardizzato con cancello del collaudo (scala all'avvio >= 0,1 x
ancore pesate; tetto 2x; discesa del 10%; conteggio nel 90% dei passi); blocchi a 200 Hz dalla posa a 60 Hz interpolata a 2 kHz; regola di
validita' (righe tutte a zero = fallimenti IK, margine misurato dal profilo di |differenza prima| prima di congelarlo); standardizzazione per
giunto, per lato se il criterio (0,5 deviazioni / rapporto 1,5) lo dice; registrazioni a mano non sollecitata tenute (23% delle ore); criterio
R3 contro R0 del pilot (riusato, con le condizioni (a)-(d) del §3 e tre controlli d'uguaglianza); R3c come controllo descrittivo. Che cosa
renderebbe il risultato non interpretabile? Che cosa manca? Che cosa toglieresti?

**D. L'albero delle decisioni dopo R3.** Se R3 non passa (delta < +1 punto o IC che tocca lo zero su P1 o P2), qual e' la lettura e la mossa
successiva? Se passa su P2 e non su P1? Se R3c batte R0 e R3 non batte R3c? Vogliamo congelare questo albero PRIMA di leggere i risultati, come
per le soglie.

**E. Ricalcoli indipendenti** (riporta i numeri che ottieni e il file da cui li prendi): la somma delle ancore sul pannello fisso di R0 ai passi
0 / 500 / 1.500 / 40.000 (`~/wearusfm_local/pilot_obiettivi_20261024/runs/PILOT-R0-A-30M-s0/fixed_eval.jsonl`, se hai accesso; altrimenti dai
valori riportati nel protocollo: 29,8 / 10,17 / 7,23 / 4,35); la quota di emg2pose nel manifest v1.1 (`results/infra/nested_manifests_v1_1_20261015.json`:
share 0,24618, 287,8 h, 158 utenti); le 4.973 registrazioni a mano non sollecitata su 17.136 (66,3 h su 288,9) dal CSV dei metadati; i totali
del preventivo (solo R3 49,7 / 67,0 GPU-ora e 78,6 / 275 core-ore; con R3c 98,1 / 132,0 e 149,4 / 531).

## Forma della risposta

- Un file `docs/reviews/review_seconda_ondata_R3_20261026.md` sul tuo ramo, piu' le tue voci in `docs/claude_GPT_shared` (sezione 1 per la
  review, sezione 2 per ogni problema che vuoi che Claude corregga, con prova o regressione).
- Separa nettamente **verifica** (fatti controllati, con `file:riga`) da **opinione** (direzione, priorita'), e marca ogni interpretazione come tale.
- Per ogni domanda A-E un verdetto in una riga, poi il dettaglio. Chiudi con: (1) «cose da fare prima del lancio di R3», ciascuna marcata
  bloccante o minore; (2) «cose da fare dopo, comunque vada R3»; (3) la tua posizione sulla direzione in tre righe, con quanto sei confidente.
- Non riscrivere soglie o regole firmate: se ne proponi di diverse, scrivile come «proposta» per la firma di Simone, con la ragione.
- Cita Simone alla lettera quando lo citi; non attribuirgli decisioni non registrate in `docs/decisioni.md`.
- Nessun job sul cluster, nessun deploy, nessuna modifica a `main` per questa review.
