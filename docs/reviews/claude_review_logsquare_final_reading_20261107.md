# Review Claude: lettura finale di T100-LQ (linea 3), 07/11/2026

Richiesta di Simone, 07/11: «se è finito tutto fammi l'analisi e la tua valutzione scientifica», dopo «dovremmo aver finito ua,
verifica e fammi una lettura dei risultati, se non devi lanciare nessun task». Nessun job lanciato da me: solo letture da Leonardo
(`sacct`, `squeue`, `find`, `cat`, `sha256sum`) e dal ramo di GPT `origin/codex/tokenizer-logcontrast-20261104`, punta
`a8ea8b5c5902e8bda3708a5081f1cfa30b5f4bf3` (07/11 16:36, invariata alle 17:43). GPT resta responsabile scientifico e operatore
della linea 3: questa è la mia lettura indipendente, richiesta da Simone; la lettura di GPT non è ancora pubblicata.

Evidenze: `claude_review_logsquare_final_reading_20261107_numbers.txt` (hash dei 16 file letti, numeri estratti, ricalcoli, sacct),
`..._pose.py` e `..._probe.py` (ricalcoli indipendenti). Criteri: protocollo `docs/tokenizer_logsquare_protocol_20261104.md`
(approvato nella mia review 6c0ce65 del 04/11): primario LQ meno T100-L su E_dyn della posa e su EPN612; secondari LQ meno T25-L e
LQ meno T100-G; IC95 appaiati con 2.000 estrazioni di utenti/soggetti, seme 20261102; soglia pratica posa 2% di riduzione relativa
dell'errore con IC sotto zero; non inferiorità P1/P2 con margine 1 pp (limite inferiore oltre -1 pp).

## 1. Esito in breve

- Tutto finito e integro: 35 job `COMPLETED 0:0` (training 63593128 a 19.618 passi, 4 estrazioni, benchmark, diagnostiche,
  18 job CPU di lettura del 07/11). I tre confronti finali (63775886 sonde, 63786157 posa 5.000, 63786158 posa 19.618) sono completi.
- Ho ricalcolato i contrasti dal per-utente (posa) e dai conteggi per soggetto (classificazione): identici a quelli dei rapporti.
- **Posa (primario): nulla.** A 19.618 passi E_dyn di LQ 0,8441 contro 0,8434 di T100-L; riduzione relativa dell'errore -0,09%,
  IC95 [-0,32%; +0,14%]. Un guadagno del 2% è escluso. A 5.000: -0,25% [-0,54%; +0,03%].
- **EPN612 (primario di classificazione): +1,53 pp** [+1,01; +2,01] su T100-L, ma T100-G fa lo stesso (+1,48 pp) e LQ meno T100-G
  vale +0,06 pp [-0,51; +0,60]. P1 e P2: non inferiorità a 1 pp sostenuta verso T100-L e T25-L; verso T100-G irrisolta su P2.
- Mia valutazione (sezione 7): il ramo log-quadratico non danneggia ma non porta nulla sulla posa e nulla oltre GELU sulla
  classificazione; non lo adotterei come tokenizer di riferimento. Decisioni a Simone, lettura scientifica a GPT.

## 2. Cosa ho verificato

| Controllo | Esito |
|---|---|
| `sacct` dal 04/11 18:00 al 07/11 17:51, nomi `LOGSQ-*` | 35 job con esecuzione, tutti `COMPLETED 0:0` tranne il tentativo 63532335 del 05/11 (`FAILED 1:0`, 21 s, già noto) |
| `squeue` alle 17:43 | nessun job LOGSQ in coda; resta solo il training della linea 1 (63696864, PENDING) |
| Ledger `~/wearusfm_local/jobs/ledger.tsv` | 29 ID logici LOGSQ dal 05/11 15:00, tutti `sottomesso` da `gpt`; un solo `rifiutato` (aggregato P2, limite di 10 invii) poi riaccettato |
| Uscite su `$WORK/wearusfm_runs/results/logsquare_cpu_20261107/` | 39 file; letti i 16 JSON (1,75 MB) con sha256 ricalcolato al trasferimento, 16 su 16 uguali |
| Pin dei confronti | i rapporti ridge puntati dai compare (`5ed2e16a…`, `2c2c5924…`) coincidono con i `report.json.gz` pubblicati da GPT in `completed_ridge/`; gli aggregati puntati dal confronto sonde (`9bf14200…`, `286d11e8…`) coincidono con i file letti |
| Convergenza | ridge: alpha 1e5 per i quattro bracci a 19.618, minimo interno, `boundary_unresolved: false` (a 5.000: 1e6 per LQ, T25-L, T100-G e 1e5 per T100-L, sempre interno); sonde: 84 fit P1 e 56 P2 per braccio, 0 avvisi, 0 al limite, `reading_status: solver_converged_by_criterion`, `decision_valid: true` |
| Appaiamento | stessi utenti (15), stesse finestre selezionate (8.329 dinamiche, 18.548 totali), stessi denominatori; stessi 314 + 32 + 8 + 8 + 19 soggetti e conteggi per classe; `draws_sha256` uguali fra gruppi |
| Ricalcolo indipendente | posa: tre contrasti per tre gruppi per due dosi, identici; sonde: celle, contrasti ed etichette NI identici |

Non riverificato: il fit ridge e le predizioni (richiederebbero cache e target su Leonardo); i ricalcoli partono dalle somme
sufficienti e dai conteggi pubblicati nei rapporti, che il codice `compare` rilegge dagli artefatti con hash.

## 3. Training e costi

| Voce | Valore |
|---|---|
| Training 63593128 | sottomesso 05/11 15:18:42, partito 07/11 09:06:13 (41 h 47 min in coda), finito 12:15:41; 11.368 s su 14.400; `max_steps` 19.618; 28.386.589 parametri; memoria GPU max 15,83 GiB |
| Estrazioni a 19.618 | posa 63772648 1.845 s (tetto 3.000), P1 63772649 832 s (2.100), P2 63772650 1.719 s (3.000); posa 5.000 63615952 1.866 s |
| Lettori CPU del 07/11 | 18 job (checkout A e B compresi): 13,14 core-ore su un tetto di 51; fit ridge 55 e 61 s; sonde da 524 a 1.645 s |
| Campagna T100-LQ | 35 job: **19,06 GPU-ore** e 165,75 core-ore allocate (training 17,07 su tetto 22; estrazioni e benchmark 1,77; preflight e diagnostica 0,23); riferimento del protocollo 26,75 GPU-ore |

## 4. Posa

E = SSE centrata per finestra/giunto divisa per la varianza supportata dei target (più basso è meglio; 1 = baseline della media di
finestra). Dinamiche: `pose_var` originale della finestra intera oltre 0,2574. `delta` = (SSE_LQ meno SSE_controllo) / somma delle
varianze; negativo = LQ meglio. IC95 dal bootstrap appaiato sugli utenti.

### 19.618 passi, finestre dinamiche (primario), 15 utenti, 8.329 finestre

| Braccio | E_dyn | IC95 | corr | var pred / var target | MAE (rad) |
|---|---:|---|---:|---:|---:|
| T100-LQ | 0,8441 | [0,8234; 0,8639] | 0,3949 | 0,162 | 0,30037 |
| T100-L | 0,8434 | [0,8223; 0,8632] | 0,3958 | 0,156 | 0,30037 |
| T25-L | 0,8520 | [0,8305; 0,8722] | 0,3847 | 0,148 | 0,30422 |
| T100-G | 0,8389 | [0,8183; 0,8585] | 0,4014 | 0,169 | 0,30007 |

| Contrasto | delta | IC95 | riduzione relativa | segnale 2% |
|---|---:|---|---:|---|
| LQ meno T100-L (primario) | +0,000726 | [-0,001190; +0,002724] | -0,09% [-0,32%; +0,14%] | no |
| LQ meno T25-L | -0,007887 | [-0,010246; -0,005484] | +0,93% | no |
| LQ meno T100-G | +0,005156 | [+0,003040; +0,007297] | -0,61% | no |
| T25-L meno T100-L (contesto, mio, stessi draw) | +0,008613 | [+0,006726; +0,010603] | T100-L meglio dell'1,02% | |
| T100-G meno T100-L (contesto, mio) | -0,004430 | [-0,007286; -0,001475] | T100-G meglio dello 0,53% | |

Per utente, primario: LQ meglio in 6 su 15, peggio in 9. I due secondari ricalcano gli effetti già letti il 04/11 (finestra 100 ms
+1,01%, GELU +0,53%): il vantaggio di LQ su T25-L è l'effetto finestra, lo svantaggio verso T100-G è l'effetto GELU.
Tutte le finestre: delta +0,001340 [-0,000336; +0,002885], riduzione -0,15% [-0,33%; +0,04%]. Statiche: +0,004429
[+0,000934; +0,008211], LQ peggiore dello 0,43% [0,09%; 0,81%]; E sopra 1 per tutti i bracci (1,020-1,025).

### 5.000 passi, finestre dinamiche

| Braccio | E_dyn | IC95 |
|---|---:|---|
| T100-LQ | 0,9129 | [0,8966; 0,9263] |
| T100-L | 0,9106 | [0,8925; 0,9259] |
| T25-L | 0,9215 | [0,9069; 0,9334] |
| T100-G | 0,8998 | [0,8824; 0,9157] |

LQ meno T100-L: +0,002238 [-0,000242; +0,004880], riduzione -0,25% [-0,54%; +0,03%]; LQ meglio in 3 utenti su 15. LQ meno T25-L
-0,008607 [-0,011759; -0,005579] (+0,93%); LQ meno T100-G +0,013035 [+0,007875; +0,018795] (-1,45%). Tutte le finestre:
-0,002467 [-0,004713; -0,000172], +0,26% [+0,02%; +0,50%]. Statiche: -0,026132 [-0,032046; -0,020293], +2,44% [+1,90%; +3,00%],
con `signal_2percent_and_ci_below_zero: true`: vedi N-1, è un'anomalia di T100-L a quella dose, non un effetto LQ.

Dose: da 5.000 a 19.618 E_dyn scende da 0,913 a 0,844 per LQ e da 0,911 a 0,843 per T100-L: stessa traiettoria, rapporto
LQ/T100-L 1,002 e 1,001.

### W_log (lettura descrittiva R1-b, teacher)

| Passo | colonne min-max | Fourier (24) media, quota norme² | spline (24) | MLP (16) | somma norme² |
|---|---|---|---|---|---:|
| 5.000 | 0,041-0,202 | 0,138, 61,5% | 0,066, 14,0% | 0,107, 24,4% | 0,766 |
| 19.618 | 0,124-0,393 | 0,268, 39,9% | 0,195, 21,4% | 0,326, 38,7% | 4,453 |

Le 64 colonne restano tutte positive e crescono di circa cinque volte in norma quadrata; la famiglia MLP guadagna quota. Il ramo
log-quadratico viene usato sempre di più dal modello, ma le letture a valle non cambiano: descrittivo, nessuna conclusione causale.

## 5. Classificazione a 19.618 (balanced accuracy %, IC95 appaiato per soggetto)

| Dataset (soggetti, classi) | T25-L | T100-L | T100-G | T100-LQ | LQ meno T100-L | LQ meno T25-L | LQ meno T100-G |
|---|---:|---:|---:|---:|---|---|---|
| EPN612 (314, 6), primario | 81,60 | 81,10 | 82,58 | 82,63 [81,42; 83,88] | **+1,53 [+1,01; +2,01]** | +1,03 [+0,51; +1,57] | +0,06 [-0,51; +0,60] |
| UCI (19, 6) | 77,44 | 76,34 | 78,46 | 76,88 [69,15; 83,67] | +0,54 [-2,30; +3,52] | -0,56 [-2,69; +2,07] | -1,58 [-5,31; +2,40] |
| NinaPro DB2 (32, 49) | 18,56 | 18,68 | 18,22 | 18,63 [16,64; 20,67] | -0,05 [-0,82; +0,73] | +0,07 [-0,52; +0,64] | +0,41 [-0,38; +1,09] |
| NinaPro DB3 (8, 49) | 3,15 | 2,83 | 3,38 | 2,43 [1,93; 3,13] | -0,41 [-1,22; +0,36] | -0,73 [-1,48; -0,12] | -0,95 [-1,80; -0,23] |
| NinaPro DB6 (8, 7) | 34,82 | 34,14 | 33,75 | 33,20 [31,55; 35,07] | -0,94 [-2,73; +0,79] | -1,62 [-4,23; +1,16] | -0,55 [-2,59; +1,36] |

| Compito | T25-L | T100-L | T100-G | T100-LQ | LQ meno T100-L | NI 1 pp | LQ meno T25-L | NI | LQ meno T100-G | NI |
|---|---:|---:|---:|---:|---|---|---|---|---|---|
| P1 (pesi 8/13, 3/13, 2/13) | 17,51 | 17,40 | 17,19 | 17,13 [15,87; 18,43] | -0,27 [-0,85; +0,31] | sostenuta | -0,37 [-0,91; +0,21] | sostenuta | -0,05 [-0,66; +0,50] | sostenuta |
| P2 (pesi 0,5 e 0,5) | 79,52 | 78,72 | 80,52 | 79,76 [75,77; 83,24] | +1,04 [-0,40; +2,49] | sostenuta | +0,24 [-0,89; +1,60] | sostenuta | -0,76 [-2,58; +1,18] | irrisolta |

Contesto (mio, stessi draw): su EPN612 T100-G meno T100-L +1,48 pp [+0,83; +2,12] e T25-L meno T100-L +0,50 [-0,11; +1,07];
su P2 T100-G meno T100-L +1,80 [-0,35; +3,64]. Per soggetto su EPN612, LQ contro T100-L: meglio 185, uguale 18, peggio 111 su 314.
DB3 è al caso per tutti i bracci (caso 2,04%): le differenze lì non si leggono.

## 6. Note

- **N-1, bandiera statica a 5.000.** `signal_2percent_and_ci_below_zero: true` per LQ meno T100-L nel gruppo statico a 5.000 (+2,44%).
  In quel gruppo T100-L è l'anomalia (E 1,0695 contro 1,043-1,046 degli altri tre bracci): anche T25-L e T100-G lo battono del 2,25%
  e 2,33% con gli stessi draw. A 19.618 i quattro bracci stanno fra 1,020 e 1,025 e LQ è il peggiore. Il gruppo statico non è il
  primario e la bandiera non va riportata come segnale di LQ.
- **N-2, archiviazione.** I tre confronti finali e i rapporti delle sonde stanno solo su `$WORK/wearusfm_runs/results/logsquare_cpu_20261107/`
  (persistente): GPT ha interrotto i controlli dopo gli ultimi invii, come chiesto da Simone, e nel repo mancano copie e hash di
  `probe/compare.json` (878.936 byte, `3517fc27…`), `pose/5000/compare/report.json` (263.400, `8b21ea16…`) e
  `pose/19618/compare/report.json` (263.158, `e2439d88…`). Cache delle estrazioni e checkpoint LQ (5.000 e 19.618, 454,8 MB ciascuno)
  stanno su SCRATCH: cancellazione automatica verso metà dicembre; conservarli o no è una decisione di Simone.
- **N-3, evidenza esplorativa.** Un seme di pretraining, validazione di sviluppo, nessuna correzione per molteplicità (18 contrasti di
  posa, 21 di classificazione). Il controllo di ripetibilità T25-L contro T100-L mostra differenze di 0,5-1 pp su EPN612 e dell'1% in
  E_dyn: differenze di quest'ordine fra bracci non si distinguono dal rumore di un singolo pretraining.

## 7. La mia valutazione scientifica

Interpretazioni mie, per Simone; la responsabilità scientifica della linea 3 resta a GPT.

1. **Sulla posa il risultato è un nullo preciso, non un'assenza di potenza.** Con 15 utenti e 8.329 finestre dinamiche l'IC95 del
   contrasto primario è largo circa mezzo punto percentuale in termini relativi, e sta centrato sullo zero a entrambe le dosi.
   Correlazione, rapporto di varianze e MAE di LQ e T100-L coincidono alla terza cifra. Le risposte log-quadratiche non aggiungono
   informazione utile alla lettura lineare della cinematica dal teacher, almeno a 30M, seme 0, 19.618 passi.
2. **Su EPN612 il guadagno c'è ed è netto (+1,5 pp, 185 soggetti su 314 migliorano), ma non è specifico di LQ:** T100-G ottiene lo
   stesso +1,5 pp e il contrasto LQ meno T100-G è nullo. La lettura più semplice è che una non linearità nel tokenizer aiuta la
   classificazione a 200 Hz, e la forma log-quadratica non fa meglio della GELU. Come dice il protocollo, un vantaggio EPN non
   dimostra un effetto della sola frequenza; qui non dimostra nemmeno un effetto della forma log-quadratica.
3. **Nessun danno misurabile:** non inferiorità a 1 pp sostenuta su P1 e P2 verso T100-L; su DB3 e DB6 i segni sono negativi ma gli IC
   coprono lo zero e DB3 è al caso per tutti. Il costo d'inferenza del ramo è piccolo (+0,3% posa, +1,3% classificazione dal benchmark).
4. **Conseguenza, se dovessi scegliere io:** non adotterei T100-LQ come tokenizer di riferimento. Non migliora il primario di posa, e
   il vantaggio su EPN612 si ottiene con T100-G, più semplice. Non vedo motivo di spendere altre dosi o semi sul ramo log-quadratico
   così com'è. Resta vero, come scrive il protocollo, che un nullo su questa forma non esclude energia, non linearità o approcci LEAF
   in generale: la «seconda domanda» (contrastiva) spetta a GPT formularla e a Simone decidere se affrontarla.
5. Costi: la campagna è chiusa a 19,06 GPU-ore contro un riferimento di 26,75, con tutti i tetti rispettati.

## 8. Prompt per GPT

Testo da incollare nella chat GPT logcontrast (linea 3), se Simone vuole:

> Claude ha letto i risultati finali di T100-LQ su richiesta di Simone (review `docs/reviews/claude_review_logsquare_final_reading_20261107.md`
> sul ramo `worktree-claude-training-engine`). Tutti i job sono COMPLETED 0:0 e i contrasti ricalcolati in modo indipendente dal per-utente e
> dai conteggi per soggetto coincidono con i tuoi rapporti. Esito: posa E_dyn a 19.618 LQ 0,8441 contro T100-L 0,8434, riduzione relativa
> -0,09% [-0,32%; +0,14%], nulla a entrambe le dosi; EPN612 +1,53 pp [+1,01; +2,01] su T100-L, ma T100-G fa +1,48 e LQ meno T100-G vale
> +0,06 [-0,51; +0,60]; P1 e P2 non inferiori a 1 pp verso T100-L e T25-L, irrisolto verso T100-G su P2. Valutazione di Claude: LQ non
> migliora la posa e non aggiunge nulla a GELU sulla classificazione; non lo adotterebbe come riferimento. Tre note: N-1 la bandiera
> `signal_2percent_and_ci_below_zero` del gruppo statico a 5.000 è un'anomalia di T100-L a quella dose (anche T25-L e T100-G lo battono
> del 2,2-2,3%), non un segnale LQ, e non va riportata come tale; N-2 i tre confronti finali e i rapporti delle sonde sono solo su
> `$WORK/wearusfm_runs/results/logsquare_cpu_20261107/`, da archiviare nel repo con hash, mentre cache e checkpoint su SCRATCH scadono a
> metà dicembre, decisione di Simone; N-3 un seme, validazione di sviluppo, nessuna correzione per molteplicità. Ti chiede di pubblicare la
> tua lettura scientifica sul tuo ramo con la stessa tabella dei contrasti, di archiviare i tre confronti, e di formulare, se la ritieni
> ancora utile, la seconda domanda (contrastiva) come proposta per Simone. Nessun nuovo job è autorizzato da questa review.
