# Review di Claude della proposta E3 sul checkpoint B di GPT (28/10/2026)

Richiesta: `docs/prompt_claude_review_e3_b_20261028.md` (GPT, commit 23cf073 su `origin/codex/r3-long-collaudo-20261028`), consegnata da
Simone: «Ti chiedo solo una review scientifica della proposta E3 di GPT: valuta se sia il prossimo passo piu' utile, se il confronto
ridge/testa sia interpretabile e quali correzioni siano indispensabili. Nessuna implementazione, estrazione o esperimento in questa fase.»
Letti in sola lettura: la sintesi (`docs/reviews/sintesi_scientifica_20261028.md`, 9142ca6), `dinamica_b_results_20261028.md`,
`docs/dinamica_b_20261028.md`, `b_e1e2_results_20261028.md`, `results/infra/pilot_posa/b_dynamics_20261028/final/report.json` e
`analysis/analysis.json`, `scripts/pose_dynamics.py`; dal mio worktree `src/wearusfm/harness/pose_eval.py` (identico in 24139ab),
`src/wearusfm/model/pose_head.py`, `scripts/pose_eval.py`, `docs/emendamento_pilot_posa_v1.md`, il rapporto E3 di R3c
(`results/infra/pilot_posa/pose_eval_R3c-e2_62443474.json`) e il JSON locale di E1/E2 di B. **Nessun job, nessuna estrazione, nessun
file di GPT modificato.** I calcoli nuovi qui sotto sono aritmetica sui momenti gia' pubblicati da GPT, fatta sul Mac.

## Verdetto: concordo con modifiche

E3 sullo student di B a 5.000 e' utile, costa poco (circa 0,1 GPU-ora attese) e completa la diagnosi a 5.000 (testa contro encoder). Ma
**come sta oggi nel codice il confronto ridge/testa non e' interpretabile**, per tre motivi concreti e verificati: (i) la ridge legge i latenti
PRIMA della LayerNorm, la testa DOPO: famiglie diverse, non lo stesso lettore con un fit diverso; (ii) la ridge fitta anche i giunti non
validi, dove il target e' 0 con maschera falsa (circa 18% degli elementi): un lettore handicappato verso la costante; (iii) la griglia di
alpha si e' chiusa al bordo (100 su {0,1; 1; 10; 100}) gia' su R3c: la selezione non ha finito di cercare. Con le quattro correzioni
indispensabili del §7 il confronto diventa pulito. In piu': E3 B non cambia la mossa successiva (in tutti gli esiti il passo dopo e' lo
stesso, altri passi del tronco) e quindi va fatto come lettura diagnostica e come **punto di partenza** per la stessa lettura a 19.618 su
R3 e su R3c, dove esiste il controllo appaiato allo stesso passo, non come cancello della continuazione.

## 0. La sintesi di GPT: numeri verificati, diagnosi condivisa, un'aggiunta

Tutti i numeri della sintesi coincidono con le fonti: E1 0,937552 [0,917038; 0,952262], mediana 0,947380, 15/15; E2 gap 0,017654
[0,013260; 0,023309], 14/14, disallineato/costante 1,007443 (JSON di E1/E2); dinamiche centrate 0,927475 [0,912480; 0,939174], Vp/Vy
0,063240, correlazione 0,269936, guadagni 0,044640 / 0,038692, poco dinamiche 1,059447 con 0/15 (report.json); B 0,883 = -11,7% e 23/32
(verdetto); audit 20 registrazioni di 10 utenti, 100/100; E3 di R3c 0,949 = 5,07%. La ridge di E3 sceglie alpha **fra utenti disgiunti**
del train (meta' degli utenti contro l'altra meta', `ridge_accessibility`): il protocollo di GPT «selezione solo fra utenti train» e' gia' nel
codice, anche se la docstring dice «2 meta' per utente» e andrebbe corretta. Condivido la diagnosi: H1 sostenuta, H2 estrema indebolita,
catena dei bersagli meno sospetta, utilita' da FM aperta; e condivido le cautele (oracolo della centratura, nessun confronto fra metriche
diverse, nessun tuning su val).

**Aggiunta: la testa e' gia' calibrata in ampiezza.** Dai momenti pubblicati (Vy, Vp, C per gruppo) si ricava il fattore di scala a che
minimizzerebbe la MSE centrata se si moltiplicassero le fluttuazioni predette (a = C/Vp), e il minimo raggiungibile, Vy x (1 - rho^2):

| Gruppo | Vy | Vp | C | rho | Vp/Vy | a ottimo | centrata/Vy | 1 - rho^2 | guadagno se riscalata |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dinamiche | 0,5335 | 0,0337 | 0,0362 | 0,270 | 0,063 | **1,07** | 0,9275 | 0,9271 | 0,0002 |
| Poco dinamiche | 0,1021 | 0,0145 | 0,0042 | 0,109 | 0,142 | **0,29** | 1,0594 | 0,9881 | 0,0073 |
| Tutte | 0,3128 | 0,0239 | 0,0198 | 0,230 | 0,076 | 0,83 | 0,9495 | 0,9473 | 0,0007 |

Letture: (a) sulle dinamiche la testa sta gia' all'ottimo di MSE per la correlazione che ha (0,9275 contro 0,9271): «la varianza predetta e'
il 6% del target» e' la conseguenza di rho = 0,27, non un difetto di ampiezza; un guadagno globale darebbe 0,0002 di MSE, cioe' niente.
Rafforza con un numero la frase di GPT «non si deduce che basti amplificare le uscite»: il numero da muovere e' la correlazione, cioe'
l'informazione letta, non la scala. (b) Sulle poco dinamiche a = 0,29: la testa emette fluttuazioni troppo grandi per cio' che sa; in
assoluto Vp si dimezza (0,034 -> 0,015) mentre Vy cala di cinque volte (0,53 -> 0,10): la testa modula l'ampiezza con l'attivita', ma meno
del bersaglio. (c) Per utente (dinamiche) a sta fra 0,83 e 1,29 in 14 utenti su 15 (0,42 per l'utente con 7 finestre) e rho fra 0,21 e 0,37:
calibrata per tutti, nessun utente anomalo. Conseguenza per E3: la domanda giusta e' se un altro lettore lineare, **sulle stesse feature della
testa**, raggiunge una correlazione centrata piu' alta; per questo la scomposizione deve stare dentro E3 (punto 4 di GPT) e il confronto deve
avvenire nella stessa famiglia (punto 2).

## 1. Utilita' del prossimo esperimento

E3 B (ridge sui latenti congelati dello student a 5.000, alpha scelto nel train, lettura su val) risponde a «c'e' informazione sul movimento
accessibile a un lettore lineare che la testa non recupera?». Cosa cambierebbe, scritto prima:

| Esito (primario: rapporto centrato sulle dinamiche, ridge contro testa, differenza appaiata per utente) | Lettura | Mossa |
|---|---|---|
| Ridge nettamente migliore (IC95 appaiato della differenza tutto sotto 0) | i pesi della testa non hanno ancora estratto cio' che i latenti portano: limite di dose/ottimizzazione della testa | proseguire il tronco; la testa va solo lasciata imparare (nessun cambio di testa ora) |
| Equivalenti (IC contiene 0) | a 5.000 la testa legge tutto cio' che e' linearmente accessibile: il limite e' nei latenti a questa dose | proseguire il tronco; solo altri passi (o un altro obiettivo) possono muovere la correlazione; la prossima E3 misura se l'accessibilita' cresce con la dose |
| Ridge peggiore | il disegno della ridge (sottoinsieme, regolarizzazione) e' piu' debole della testa: non informa sull'encoder | proseguire il tronco; E3 non dice nulla di nuovo |

In tutti e tre i casi la mossa e' la stessa. Quindi E3 B **non e' un cancello** e non deve condizionare la decisione sulla continuazione
(`docs/proposta_claude_continuazione_b_19618_20261028.md`, a2753a6, da discutere con GPT). Il suo valore e' un altro: (1) chiude la diagnosi
a 5.000 a costo trascurabile; (2) e' la **baseline** per la stessa lettura a 19.618, su R3 continuato e su R3c (`checkpoint_step0019618.pt`
esiste): la differenza appaiata «R3 meno R3c allo stesso passo» sul rapporto centrato della ridge e' la lettura (b2) dell'emendamento («la posa
entra nella rappresentazione grazie al termine») a pari dose, che a 5.000 non si puo' fare perche' R3c non ha un checkpoint a 5.000.
Controllo piu' semplice gia' disponibile: la calibrazione del §0 (gratis) esclude l'ampiezza come leva. Ragione per proseguire direttamente il
training: nessuna contro E3 B, purche' E3 B non diventi un prerequisito; i due possono andare in parallelo.

## 2. Confronto con la testa: oggi non e' nella stessa famiglia

- **Testa** (`pose_head.py`): `LayerNorm(d)` con gamma e beta appresi, poi `Linear(d, 20)`, sull'ingresso `latent_mean(enc.z)` (media sui K
  latenti, per istante). **E3 oggi** (`ridge_accessibility`): ridge con intercetta su `lat`, che e' quella stessa media **prima** della
  LayerNorm (`extract`: `z = latent_mean(enc.z)`, `p = model.pose_head(z)`; in cache finisce `z`). La LayerNorm e' una normalizzazione per
  istante (non lineare nelle feature): una differenza ridge-testa puo' venire dalla famiglia, non dall'informazione.
- **Indispensabile (M1): confronto primario sulle feature della testa.** Ridge sui latenti dopo la LayerNorm della testa congelata (gamma,
  beta, eps dal checkpoint). Cosi' ridge e testa sono due punti della stessa famiglia lineare sullo stesso ingresso: la ridge e' l'ottimo in
  forma chiusa dell'obiettivo della testa (MSE standardizzata, con intercetta) sul sottoinsieme del train; la testa e' il punto raggiunto da
  5.000 passi di SGD. La differenza misura il fit, cioe' dose/ottimizzazione, non la famiglia. La LayerNorm si applica sul CPU dalla cache
  (per istante, costo nullo); gamma, beta ed eps si scrivono nel npz all'estrazione (o si leggono dal checkpoint nel job del rapporto, con lo
  stesso SHA verificato dalle guardie di GPT).
- **Indispensabile (M2): controllo di coerenza incorporato.** Ricalcolare sul CPU la predizione della testa dalle feature in cache
  (LayerNorm + W, b del checkpoint) e confrontarla con `pred` in cache (calcolata sulla GPU dai latenti non arrotondati): lo scarto massimo
  atteso e' dell'ordine di 1e-3 in unita' standardizzate (float16 sui latenti), da dichiarare prima come tolleranza; uno scarto grande
  significa feature, LayerNorm o pesi sbagliati e invalida il confronto. Costa niente e valida tutta la catena.
- **Indispensabile (M3): fit mascherato per giunto.** Oggi `_instants` prende gli istanti con almeno un giunto valido e `ridge_fit` usa
  l'intera riga di 20 giunti; nei giunti non validi il target e' 0 (`pose_targets.window_pose`: `target = zeros`, `mask[has] = True`).
  Su val le dinamiche hanno 21.750.080 elementi validi su 26.652.800 (81,6%): circa il 18% dei bersagli della ridge sono zeri che non
  sono pose, e spingono il lettore verso la costante. Soluzione esatta e banale: 20 soluzioni separate, una per giunto, ciascuna sulle sole
  righe valide di quel giunto (d = 384: 20 sistemi 384 x 384). Vale a prescindere da come la testa e' allenata (la sua loss va comunque
  ricontrollata nel codice di training prima dell'implementazione: se mascherata, l'asimmetria oggi e' tutta a sfavore della ridge).
- Intercetta e baseline: l'intercetta della ridge e' la media dei target del sottoinsieme (40 utenti) meno `mx @ w`; la costante del
  criterio e' 0 (media del train intero). Il termine delle medie della scomposizione puo' quindi differire per motivi che non c'entrano
  col movimento: per questo il primario e' il termine centrato e le medie si riportano accanto, non si sommano (coerente col punto 4 di GPT).
  Facoltativo: riportare la media per giunto dei target standardizzati del sottoinsieme (se lontana da 0, il sottoinsieme non e' rappresentativo).
- Pesi e supporto: la ridge pesa gli istanti alla pari (oggi 8 istanti per finestra, campionati); la lettura su val usa tutti gli istanti
  validi; la testa ha visto il train intero (158 utenti; 160.000 finestre dello stream in 5.000 passi, una parte con posa) nella vista mascherata. L'asimmetria rende **informativo** un
  esito «ridge migliore» (meno dati, forma chiusa, eppure legge di piu') e **ambiguo** un esito «ridge peggiore» (puo' essere il sottoinsieme).
  Facoltativo ma utile e gratis: curva interna 20 contro 40 utenti sul train (se l'errore interno scende ancora, il sottoinsieme e' piccolo).
- Disegno minimo: **un solo primario** (post-LayerNorm, mascherato, tutti gli istanti validi, griglia aperta). **Un secondario** solo se GPT
  vuole la continuita' col numero di R3c-e2 (0,949): la E3 «legacy» identica a quella di R3c, dichiarata col suo handicap; altrimenti si
  lascia e si dichiara che il 0,949 non e' confrontabile.

## 3. Separazione train/val e riuso delle cache (dal codice)

Verificato: stesso checkpoint e stessa vista (`--which student`, `encode(inp, None)` senza maschera, stesso preprocessing del loader e
scala di sessione stimata per registrazione); statistiche di standardizzazione uniche del train (`pose_stats` 757f2aee..., SHA nel config
della cache e verificato dal wrapper di GPT); sottoinsieme del train **fisso e deterministico** (`recordings_for`: seme 20261026, prima 40
utenti, poi 200 registrazioni spalmate sugli utenti; stesso CSV e stesso albero processato di R3c, quindi le stesse 200 registrazioni, da
confermare confrontando gli stem con il `train_subset.npz` di R3c-e2 gia' su $SCRATCH); split interno per utenti (meta' pari / meta' dispari
dell'elenco ordinato); i 15 utenti di val sono disgiunti dai 158 del train per costruzione del CSV. La cache di val di B (2,48 GB,
SHA 70bb5513...) contiene gia' `lat` (float16), `pred`, `target`, `mask`, `std`, `pose_var`: **non va ri-estratta**; manca solo il
sottoinsieme del train dello student di B.
Limiti e controlli, distinti: (i) **float16** sui latenti: arrotondamento relativo 5e-4, trascurabile rispetto a effetti dell'ordine dell'1%,
ma la LayerNorm su valori arrotondati va validata col controllo M2, non assunta; (ii) **istanti**: 8 per finestra scelti a caso fra quelli con
almeno un giunto valido; con d = 384 si possono usare tutti gli istanti validi (circa 320.000 righe, 0,5 GB in float32, equazioni normali
accumulate a blocchi): una scelta sola, dichiarata prima; (iii) **giunti non validi**: difetto dimostrato (M3), non un controllo da fare;
(iv) il preprocessing della ridge e' la sola centratura (`mx`, `my`) stimata sul fit interno: giusto; nessuna standardizzazione delle feature,
giusto per le feature post-LayerNorm. Nessuna selezione dipendente da val in tutta la catena: confermato.

## 4. Metriche e selezione, da fissare prima

- **Obiettivo interno** per alpha: MSE standardizzata **mascherata** per elemento sulle due meta' di utenti del train (non la MAE non
  mascherata di oggi): e' la loss della testa e l'unita' della scomposizione; la ridge minimizza MSE, scegliere alpha su un'altra metrica non
  ha motivo. **Griglia aperta** (M4): log-spaziata, per esempio 1e-2 ... 1e4 (7 valori), fissata prima; se l'alpha scelto cade a un bordo si
  estende di una decade da quel lato e si risceglie, **sempre sul solo train**, prima di toccare val; l'alpha scelto si riporta.
- **Confronto primario**: differenza appaiata (ridge meno testa) del rapporto centrato `MSE centrata / Vy` sulle finestre dinamiche, sullo
  stesso supporto (stesse finestre, stesse celle n >= 2), con IC95 ricampionando gli utenti (1.000, seme 20261009, stessi campioni per i due
  lettori: appaiato per costruzione); accanto, per entrambi i lettori, termine delle medie, totale, Vp/Vy e correlazione centrata, con
  i secondari predefiniti (tutte, poco dinamiche). Lettura dichiarata prima, descrittiva e senza soglia: «la ridge recupera piu' dinamica» se
  l'IC della differenza sta tutto sotto 0; «equivalenti» se contiene 0; «legge peggio» se sta sopra. Un guadagno sulle medie o sull'intercetta
  non si legge come inseguimento del movimento: resta nel suo termine.
- **Riuso**: `scripts/pose_dynamics.py` di GPT gia' calcola somme per utente e bootstrap da una cache con `pred`; basta che il rapporto E3
  scriva le predizioni della ridge in una cache dello stesso formato (campo `pred`, piu' `pred_head` conservato) e si esegua `pose_dynamics.py`
  invariato su di essa; la differenza appaiata si calcola da due file di somme per utente con uno script piccolo. Il rapporto E3 «rapporto
  MAE/costante sulle dinamiche» resta riportato per continuita' con l'emendamento ((b1)-(b3)), senza la soglia 0,9 come verdetto.

## 5. Limiti della conclusione

Lettura descrittiva a 5.000: non si applicano le soglie firmate per 49.046 (E1 0,9 / 0,95), non si inventa un cancello dopo i numeri. I 15
utenti di val sono stati letti tre volte (E1/E2, dinamica, ora E3): legittimo perche' nulla e' stato scelto su val, e cosi' deve restare:
ogni iperparametro (alpha, istanti, griglia) dal train; dopo E3 nessun'altra lettura di val a 5.000 se non pre-registrata; la prossima
lettura di val sara' a 19.618 se si continua. Un guadagno della ridge non prova un unico collo di bottiglia; un esito debole non prova un
limite assoluto dell'encoder (asimmetria del §2). R0 confronta ricette (cambia anche il JEPA): non attribuisce; l'attribuzione alla loss di
posa e' R3 contro R3c **appaiato allo stesso passo**, possibile a 19.618 e all'endpoint, non a 5.000. Il numero E3 di R3c-e2 (0,949) va
riletto come «lettore handicappato» (M3, griglia al bordo): la frase «sotto il criterio del 10%, non assenza di informazione» resta vera ma
l'accessibilita' vera di R3c potrebbe essere piu' alta; nessun esito firmato dipende da quel numero. Registrato nel registro, sezione 2.

## 6. Perimetro minimo e fattibilita' (nessuna implementazione qui)

Cosa manca per un protocollo eseguibile, in ordine:
1. **Codice (piccolo, in `pose_eval.py` o in uno script nuovo `pose_e3_b.py`, con test sintetici):** ridge mascherata per giunto; feature
   post-LayerNorm da gamma/beta/eps del checkpoint; griglia aperta con regola di estensione; obiettivo interno MSE mascherata; tutti gli
   istanti validi; controllo M2 con tolleranza dichiarata; scrittura delle predizioni della ridge in una cache in formato `pose_dynamics`;
   `save_npz` che salva gamma/beta/eps quando c'e' la testa. Test: bersaglio lineare esatto recuperato, giunti mascherati ignorati nel fit,
   LayerNorm identita' quando gamma = 1 e beta = 0, alpha al bordo rilevato.
2. **Estrazione del sottoinsieme del train** dello student di B: codice di estrazione invariato (24139ab ha gia' `--which student`), checkout
   `pilot-posa-b-20261028`, wrapper con le guardie di GPT, `--split train --n-users 40 --n-recordings 200`, output nuovo su $SCRATCH accanto
   alla cache di val. Base documentale del costo: R3c-e2, stesse 200 registrazioni, job 62442491 COMPLETED in 4 min 12 s (0,07 GPU-ora,
   0,56 core-ore); R0-e2 62478387 in 3 min 04 s; la val di B (1.950 registrazioni) in 26 min 07 s. Attesi circa 5 min; tetto proposto 20 min
   su 1 A100 + 8 core = 0,33 GPU-ora, 2,7 core-ore. Non «quattro minuti» per assunzione: e' la misura di ieri sulle stesse registrazioni.
3. **Rapporto CPU** (seriale, 8 core, 30 GB per la QoS): legacy 48 s; con 20 sistemi per giunto x 7 alpha x 2 meta' su 320.000 righe,
   qualche minuto; tetto proposto 30 min = 4 core-ore. Poi `pose_dynamics.py` sulla cache della ridge (13 s ieri) e la differenza appaiata.
   Il codice nuovo va eseguito da un riferimento congelato (checkout nuovo da un job CPU, 1/3 core-ora, oppure payload su stdin con SHA
   come ha fatto GPT per `pose_dynamics`).
4. **Totale**: circa 0,1 GPU-ora e 2 core-ore attese; tetti circa 0,33 GPU-ora e 7 core-ore. Sotto la soglia di autonomia di GPT (regola di
   Simone del 28/10 per GPT, recepita in CLAUDE.md sul ramo di GPT). Per me resta la regola della notte del 28/10: nessun job senza
   approvazione di GPT e via esplicito di Simone; Simone a GPT: «lui non fa esperinenti adesos, e' solo per fare un review».
5. **Pre-registrazione**: letture del §4, griglia e regola di estensione, politica degli istanti, tolleranza di M2, nel protocollo e nel
   registro prima della sottomissione; il giro del `critico` sulla lettura, come GPT ha fatto per la dinamica.
6. **R0 e J2 bis**: non necessari ora, concordo: R0 non attribuisce; J2 bis senza una nuova prova di disallineamento resta dopo.
7. **Collegamento alla continuazione**: se si continua a 19.618, aggiungere la stessa E3 (stesso codice, stesse 200 registrazioni, gamma/beta
   del checkpoint relativo) a R3@19.618 e a R3c@19.618, a circa 0,6 GPU-ora in piu' (val e train di R3c, train di R3; la val di R3 viene gia'
   da E1/E2): e' la lettura di attribuzione a pari dose e la misura
   di «l'accessibilita' cresce con la dose». Propongo di inserirla nel §3a della mia proposta a2753a6 quando GPT la discute.

## 7. Riepilogo delle modifiche

**Indispensabili** (senza, il confronto non si interpreta):
- **M1** confronto primario sulle feature della testa (post-LayerNorm con gamma/beta/eps del checkpoint). Riferimento: `pose_head.py`
  (`self.out(self.norm(z_mean))`), `pose_eval.extract` (cache di `z` pre-LayerNorm). Effetto: senza, la differenza puo' essere la famiglia.
- **M2** controllo di coerenza testa-dalla-cache contro `pred`, tolleranza dichiarata prima. Effetto: valida feature, LayerNorm e pesi.
- **M3** ridge mascherata per giunto (20 sistemi sulle righe valide). Riferimento: `_instants` / `ridge_fit`, `pose_targets.window_pose`
  (zeri con maschera falsa, circa 18% degli elementi). Effetto: senza, la ridge e' tirata verso la costante e «ridge peggiore» non e' leggibile.
- **M4** griglia di alpha aperta con regola di estensione sul solo train e obiettivo interno MSE mascherata. Riferimento: `RULE["ridge_alphas"]`
  = {0,1; 1; 10; 100}, alpha = 100 scelto su R3c-e2 (bordo). Effetto: senza, la ridge e' sotto-ottimizzata per costruzione.
- Lettura primaria = differenza appaiata del rapporto centrato sulle dinamiche con IC per utente (§4), tre termini riportati, nessuna soglia.

**Facoltative** (migliorano, non cambiano l'interpretazione): tutti gli istanti validi invece di 8; curva interna 20/40 utenti; media per giunto
del sottoinsieme; secondario «legacy» per continuita' con R3c-e2; conferma degli stem del sottoinsieme contro `train_subset.npz` di R3c-e2.

**Decisione che chiedo di fissare insieme a E3 B, non dopo:** E3 B e' diagnostica e baseline, non un prerequisito ne' un cancello della
continuazione del tronco; la stessa E3 si ripete a 19.618 su R3 e R3c. Le scelte restano di GPT e le firme di Simone.
