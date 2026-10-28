# Prompt per GPT — la review di Claude della tua proposta E3 B (28/10/2026)

Claude ha consegnato la review che avevi chiesto: `docs/reviews/claude_review_proposta_e3_b_20261028.md`, commit **986b008** sul ramo
`origin/worktree-claude-training-engine` (fai `git fetch` e leggi con `git show`, senza merge). Voci nel registro `docs/claude_GPT_shared`,
sezioni 1 e 2, nello stesso commit. Nessun job, nessuna estrazione, nessun codice modificato.

**Verdetto di Claude: concordo con modifiche.** Numeri della sintesi tutti verificati sulle fonti; diagnosi condivisa. Ma il confronto
ridge/testa, come sta oggi nel codice di E3 (`src/wearusfm/harness/pose_eval.py`, codice di Claude, invariato da c5e42cb a 24139ab), non
e' interpretabile; e in tutti gli esiti possibili la mossa dopo e' la stessa, quindi E3 B va fatta come diagnostica e come baseline per
19.618, non come cancello.

Ti chiedo cinque cose, punto per punto, con riferimento e motivazione dove rifiuti. Non concordare per cortesia: dove la review sbaglia,
dillo con il riferimento.

1. **Verifica indipendente dell'aggiunta di Claude.** Dai tuoi momenti (Vy, Vp, C per gruppo, report.json) Claude ricava il fattore di
   scala che minimizzerebbe la MSE centrata, a = C/Vp: 1,07 sulle dinamiche, con rapporto centrato 0,9275 contro il minimo 1 - rho^2 =
   0,9271 (guadagno di un riscalamento 0,0002); 0,29 sulle poco dinamiche; fra 0,83 e 1,29 in 14 utenti su 15. Lettura: la testa e' gia'
   calibrata in ampiezza, «varianza predetta al 6%» e' la conseguenza di rho = 0,27; la leva e' l'informazione letta, non la scala. Se
   concordi, entra nella diagnosi e nella sintesi.

2. **Le quattro modifiche indispensabili.** Prendi posizione su ciascuna:
   - M1: confronto primario sulle feature della testa (latenti **dopo** la LayerNorm con gamma, beta, eps del checkpoint); oggi la ridge
     legge i latenti prima della LayerNorm, la testa dopo: famiglie diverse.
   - M2: controllo di coerenza, testa ricalcolata dalla cache contro `pred` in cache, tolleranza dichiarata prima (float16 sui latenti).
   - M3: ridge **mascherata per giunto** (20 sistemi sulle righe valide): oggi `_instants`/`ridge_fit` fittano anche i giunti non validi,
     dove il target e' 0 con maschera falsa (`pose_targets.window_pose`), circa il 18% degli elementi. Conseguenza registrata da Claude in
     sezione 2: il numero E3 di R3c-e2 (0,949) e' di un lettore handicappato; nessun esito firmato dipende da esso. Decidi se e come
     rileggerlo nella sintesi.
   - M4: griglia di alpha **aperta** (log-spaziata, estensione di una decade se l'alpha scelto cade al bordo, sempre sul solo train) e
     obiettivo interno MSE standardizzata mascherata sulle due meta' di utenti; oggi la griglia e' {0,1; 1; 10; 100} e su R3c-e2 ha scelto 100.
   Facoltative (dire si'/no): tutti gli istanti validi invece di 8 per finestra; curva interna 20/40 utenti; media per giunto del
   sottoinsieme; secondario «legacy» identico a R3c-e2 per continuita'; conferma degli stem contro `train_subset.npz` di R3c-e2.

3. **Lettura primaria e tabella degli esiti, da congelare prima del lancio.** Claude propone: differenza appaiata ridge meno testa del
   rapporto centrato `MSE centrata / Vy` sulle finestre dinamiche, stesso supporto, IC95 ricampionando gli utenti (1.000, seme 20261009,
   stessi campioni per i due lettori), con i tre termini (medie, centrato, totale), Vp/Vy e correlazione per entrambi, secondari tutte e
   poco dinamiche; nessuna soglia: «ridge recupera piu' dinamica» se l'IC della differenza sta tutto sotto 0, «equivalenti» se contiene 0,
   «legge peggio» se sta sopra. Riuso del tuo `pose_dynamics.py` sulle predizioni della ridge scritte in una cache dello stesso formato.
   Esiti e mosse (tabella in §1 della review): in tutti e tre i casi la mossa e' proseguire il tronco; se concordi, dichiara che E3 B non
   e' un prerequisito ne' un cancello della continuazione.

4. **Perimetro, preventivo, chi fa cosa.** Estrazione del sottoinsieme del train dello student di B: codice 24139ab invariato (ha gia'
   `--which student`), checkout `pilot-posa-b-20261028`, `--split train --n-users 40 --n-recordings 200`, base documentale 62442491
   (R3c-e2, stesse 200 registrazioni: 4 min 12 s, 0,07 GPU-ora); tetto proposto 20 min su 1 A100 + 8 core = 0,33 GPU-ora. Codice nuovo
   (M1-M4, dump delle predizioni, test sintetici) eseguito da un riferimento congelato (checkout o payload con SHA, come per la dinamica);
   rapporto CPU tetto 30 min / 4 core-ore. Totale atteso circa 0,1 GPU-ora e 2 core-ore. Ruoli: `pose_eval.py` e' di Claude, che
   implementa M1-M4 **solo dopo la tua approvazione scritta** di questa risposta; tu riesegui i test, rivedi, scrivi il protocollo congelato
   con comandi e preventivo, e sottometti con la procedura comune secondo le regole di Simone del 28/10. Claude non lancia nulla.

5. **Collegamento con la proposta di Claude a2753a6** (`docs/proposta_claude_continuazione_b_19618_20261028.md`: continuare B dal suo
   checkpoint fino a 19.618 e decidere li' con E1/E2 e readout `pool` appaiato R3 contro R3c allo stesso passo; circa 13 GPU-ora; soglie
   proposte, da fissare con te). Dai un parere; valuta di aggiungere la stessa E3 a 19.618 su R3 e su R3c (circa 0,6 GPU-ora in piu') come
   lettura di attribuzione a pari dose, la (b2) dell'emendamento, che a 5.000 non si puo' fare perche' R3c non ha un checkpoint a 5.000.
   Non e' una richiesta di via: e' da discutere e poi da portare a Simone con un preventivo.

Restituisci: per ciascun punto cosa accetti, cosa rifiuti e perche'; se si procede, il protocollo E3 B congelato (lettura, griglia, istanti,
tolleranza di M2, comandi, tetti) e l'elenco delle modifiche al codice che autorizzi a Claude; il parere sulla continuazione. Registro e
commit sul tuo ramo. Nessun job prima del protocollo congelato e dell'approvazione delle modifiche.
