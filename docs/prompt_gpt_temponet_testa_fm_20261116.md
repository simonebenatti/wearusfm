# Prompt per GPT coordinamento: TEMPONet come testa di regressione della posa nel FM, proposta di protocollo (16/11/2026)

Simone, 16/11, a Claude: «ok, adesso prepapariamo unprompt per GPT in cui gli facciamo provare ad usare temponet come testa per
la regressine nel FM». E' la domanda che aveva fatto a Claude il 06/11 sera, nella chat di coordinamento di Claude (non in una
chat GPT): «Le teste devono necessriamente essere tutte omogenee? [...] Ad esempio, una versone di temponet come testa per la
regressione come la vedi? troppo grossa ?». Alla domanda di Claude «La porto a GPT come ipotesi per il punto 6c?» aveva risposto
«Dopo la lunga». La valutazione lunga e' chiusa: conclusioni 1-4 firmate il 07/11 («Sì, firmo»). La voce del 06/11 (commit
`112ab60` e `ff8305a`) sta solo sul ramo di Claude `worktree-claude-training-engine` (`0ab9d98`, `docs/claude_GPT_shared`, righe
7512-7541): non e' fra le fonti della tua sintesi del 16/11.

Ruoli (CLAUDE.md su main `e114a78`). Ti scrivo come autore della sintesi `8eba5e4`. GPT coordinamento integra su main e coordina
i responsabili delle linee, senza diventare un secondo operatore. La responsabilita' scientifica (protocolli, bracci, criteri,
letture e piani) e' di GPT, per la firma di Simone. Dal 28/10 il lavoro affidato a GPT lo implementa e lo invia GPT, e Claude
rivede; per le linee 1 e 2 prevalgono le disposizioni del 05/11 e del 06/11. Nella tabella delle linee questa attivita' non c'e'
(lettura di Claude): affidamento del codice, operatore e budget li decide Simone. **Questo prompt non autorizza job, deploy o
codice: chiede una proposta.** Lettura di Claude: il «provare» di Simone passa prima da un protocollo firmato, con criteri
congelati prima di guardare.

## 1. Fatti

Codice e cifre da main `e114a78`, salvo dove indicato. Il punto di partenza e' la tua sintesi del 16/11 (`8eba5e4`,
`docs/reviews/regression_synthesis_20261116.md`): «La TEMPONet sui latenti è ancora una domanda aperta», con il «Primo controllo
economico possibile» a encoder congelato. Qui c'e' cio' che serve in piu' per una testa temporale.

**La sequenza che la testa leggerebbe.**
- Il backbone da' `z` di forma (S, P, K, d): P patch da 25 ms (40 al secondo), K = 64 slot per istante, d = 384 nel 30M e 896 nel
  300M (`src/wearusfm/model/fm.py:135`, `model/perceiver.py:44`).
- Solo il backbone mescola le patch fra loro: attenzione bidirezionale su tutta la finestra, RoPE sul tempo, nessuna posizione fra
  gli slot (`model/backbone.py:1-12, 74-81`). Encoder locale e Perceiver lavorano patch per patch (`model/local_encoder.py:19-20`,
  `model/perceiver.py:1-4`). Prima del backbone l'unico contesto temporale e' il supporto del tokenizer: circa 300 ms con T100, cioe'
  100 ms di finestra piu' 100 ms di contesto per lato (`docs/tokenizer_window_nonlinear_protocol_20261102.md:29-33, 67-69`).
- Una testa causale messa sopra non rende causale il sistema.

**La testa di posa di oggi e come si allena.**
- `PoseHead`: LayerNorm(d) → Linear(d, 20), inizializzata a zero, a ogni patch sulla media dei 64 slot
  (`src/wearusfm/model/pose_head.py`). Con d = 384 ha 8.468 parametri (conto di Claude).
- Perdita: errore quadratico sugli angoli standardizzati, media per finestra, peso 1,0 accanto alle ancore 0,2 (JEPA 0)
  (`src/wearusfm/training/jepa.py:173-209, 547-556`).
- La testa legge l'encoding mascherato dello studente. Ratio 0,5 dei token validi: maschere temporali corte di 1-2 patch (15%),
  medie di 4-12 (25%) e lunghe di 20-80 patch, al piu' meta' finestra, cioe' fino a 2 s (20%); maschere spaziali per tutta la
  finestra (40%). Meta' delle medie e delle lunghe sono slab su tutti i canali, nelle sessioni con l'ancora RVQ accesa; le altre
  nascondono un gruppo spaziale. Con gli slab il 12,5% delle patch di emg2pose non ha canali visibili, ma entra lo stesso nel
  termine di posa (`results/infra/tokenizer_execution_20261103/19618/T100-G/config.json:44-63`;
  `src/wearusfm/training/masking.py:10-23`; `jepa.py:173-177`).
- Il teacher e' l'EMA 0,996 di tutti i parametri, testa compresa; i buffer si copiano (`config.json:30`; `jepa.py:121-126`;
  `model/fm.py:180-182`). Cache e letture usano il teacher senza maschera (`src/wearusfm/harness/backbone_depth.py:15, 26`).
- Nel pretraining ogni finestra dura 4 s, o meno (almeno 1 s) se il tratto finisce prima, anche in emg2pose
  (`src/wearusfm/data/pretraining_loader.py:285, 332-342`). Le letture usano finestre da 4 s.
- Bersaglio per patch: media dei blocchi a 200 Hz con centro nella patch, ritardo zero; solo sullo split train di emg2pose, 158
  utenti (`src/wearusfm/data/pose_targets.py`; `docs/pilot_posa_v1.md:53-91`).
- Il ritardo zero e' la parte (b) della deroga a v10 §6.5 e §3.7, firmata da Simone il 26/10 insieme a quella a §6.1; la parte
  (a) e' il peso di loss del termine di posa (`docs/pilot_posa_v1.md:3, 97-103`). Motivo scritto: il backbone e' bidirezionale, e
  «un lead time appreso ha senso per un decoder causale, che non e' questo». La «testa lineare, non MLP» invece non e' una deroga:
  e' una scelta di Claude dentro R3, per la leggibilita' lineare del readout medio (righe 108-110).
- Il v10 §6.5, sulle teste supervisionate del pretraining: «Piccole: troppa capacità assorbe l'adattamento che si vorrebbe nel
  backbone. Le teste di regressione portano il lead time vincolato di §3.7.» (`docs/fm_emg_reference_v10.md:1013-1017`).

**Supporto e metrica delle letture.**
- Fino a 10 finestre da 4 s (160 patch) per registrazione, senza sovrapposizione (`src/wearusfm/harness/pose_eval.py:98-123`).
- Pannello di adattamento: 40 utenti train, 200 registrazioni selezionate (198 con finestre), 1.907 finestre, circa 2,1 h di segnale.
  Sono fra i 158 utenti del pretraining, con la posa supervisionata: per l'encoder sono dati di training.
- Lettura sui 15 utenti val: 1.950 registrazioni dello split (1.618 in fasi note, 332 in fasi nuove; 1.923 con finestre), 18.548
  finestre; gia' usati come sviluppo (`scripts/tokenizer_pose_readout.py:32-33`; conteggi effettivi in `docs/decisioni.md:4948-4949`;
  split in `docs/temponet_emg2pose_plan.md:38-46` sul ramo TEMPONet).
- I 15 utenti val sono lo split `val` ufficiale (`docs/emendamento_pilot_posa_v1.md:23`). FM e TEMPONet raw lo prendono dallo stesso
  CSV dei metadati, e i 15 identificativi coincidono. TEMPONet raw li ha gia' usati per l'alpha dell'EMA e il cancello, su un altro
  supporto: 1.609 registrazioni e 12.468 finestre ufficiali.
- Ridge attuale: istantanea su LayerNorm senza stato della media K, teacher, crop6 (6 ≤ t e t+6 < 160: 148 centri). Alpha 10^5,
  scelto con due fold di utenti train (`scripts/tokenizer_pose_readout.py:41-51, 232-271`).
- `E_dyn` (`scripts/pose_dynamics.py`) sulle finestre dinamiche: pose_var ≥ 0,2574231028556824, 8.327 supportate con crop6.
- Bootstrap appaiato sugli utenti: 2.000 ricampionamenti, seme 20261102 (`scripts/tokenizer_pose_readout.py:30`, ripreso da
  `scripts/consolidation_long_pose.py:26`). Il default di `pose_dynamics.py` (1.000, seme 20261009) e' quello storico di E1.
- Criterio pratico: almeno -2% relativo, con IC favorevole (`docs/tokenizer_window_nonlinear_protocol_20261102.md:191-197`).

**Numeri gia' misurati** (finestre dinamiche, T100-G 30M teacher; `E_dyn`: minore e' meglio).

| Lettura | `E_dyn` | Correlazione centrata | Varianza predetta / bersaglio |
|---|---|---|---|
| 19.618, ridge dopo il backbone | 0,838933 | 0,401 | 16,9% |
| 49.046, ridge dopo il backbone | 0,785902 | 0,463 | 22,1% |
| 49.046, testa allenata | 0,778629 | 0,471 | 23,9% |
| ridge prima del backbone, 19.618 → 49.046 | 1,035 → 1,032 | 0,074 → 0,079 | — |

Fonti: `docs/reviews/consolidation_long_results_20261107.md` e `scientific_readout.json` a `cfa6462`. Il lettore lineare con
t±100 ms (B student, crop4) ha dato -0,49% relativo e ha abbassato la varianza predetta da 0,140 a 0,108 (`ec5b309`). Simone ha
firmato il 07/11 il 49.046 come riferimento esplorativo per la posa.

**TEMPONet gia' fatta, su EMG grezzo.** Ramo `claude/temponet-db8-20261109`, punta `da5e820`; conclusioni firmate il 10/11 (DB8)
e il 12/11 (emg2pose).
- `src/wearusfm/model/temponet.py`, solo torch. Tre blocchi, ciascuno con due convoluzioni dilatate (kernel 3, dilatazioni 2/4/8)
  e una con kernel 5 e stride 1/2/4, seguita da AvgPool 2; canali 16/32/64; poi FC 64-32. BatchNorm e ReLU dopo ogni strato
  tranne l'uscita lineare; dropout 0,5 su FC I e FC II. Da' un'uscita per finestra, non per patch.
- emg2pose: 16 canali × 256 campioni (128 ms), solo passato fino all'istante predetto; EMG nelle unita' native, senza il filtro del
  loader del FM. 71.412 parametri; L1, Adam 1e-4, batch 64, 19 epoche; un modello per 158 utenti, circa 21 milioni di finestre per
  epoca. Training 3,079 GPU-ore, attivita' intera 3,513 (`docs/temponet_emg2pose_plan.md:59-70` per la configurazione, righe 187 e
  237 per i costi).
- Test ufficiale gia' visto (MAE nella tua sintesi: 14,55 / 17,67 / 17,60); cancello in val 0,831.
- Non si confronta con `E_dyn` (tua sintesi `8eba5e4`). Le conclusioni firmate il 12/11 dicono: «Conviene fissare un solo insieme
  di finestre e di istanti per tutti i modelli, per esempio quello del contesto piu' lungo, oppure dichiarare la differenza»
  (`docs/temponet_emg2pose_plan.md` a `255fc8e`, righe 304-305).
- Il ramo non e' su main (42 commit oltre `e114a78`). `tests/cpu/test_temponet.py` prova il modello intero solo con 16 × 256:
  nessun test con 384 o 896 canali o con T diverso da 256.

**TEMPONet sui latenti cosi' com'e'** (conti di Claude con `temponet.py`).
- `forward` accetta solo un ingresso (batch, C, T) di lunghezza esatta. Con gli stride di default serve T ≥ 38 (0,95 s).
- Dopo i tre blocchi restano 1 posizione temporale per T da 38 a 101 e 2 per T = 160; la FC le appiattisce, quindi l'uscita resta
  una per finestra. Per avere un'uscita per patch servono altre scelte. Con crop6 una finestra scorrevole di almeno 38 patch non
  copre i centri vicini ai bordi senza padding.
- Campo ricettivo 378 passi: sulla griglia di 25 ms sono 9,45 s, piu' della finestra di 4 s. I 128 ms di TEMPONet sono circa 5
  patch.
- Il primo blocco ha 16 canali: con d = 384 la prima convoluzione comprime 384 → 16, meno dei 20 angoli.
- Ogni strato tranne l'uscita ha BatchNorm, con le statistiche dei soli 40 utenti del pannello; la ridge usa una LayerNorm senza
  stato, per istante. TEMPONet non ha una normalizzazione d'ingresso.
- Parametri con 20 uscite: 76.788 (d = 384, T da 38 a 101), 80.884 (d = 384, T = 160), 105.460 (d = 896, T = 160). Sono lo 0,3%
  del 30M (28.362.013) e lo 0,04% del 300M, ma circa 9-10 volte la `PoseHead` del 30M.
- Quindi «TEMPONet come testa» richiede un adattamento, e ogni scelta cambia il modello rispetto alla TEMPONet di [Z21] (Zanghieri
  et al., IEEE COINS 2021; `docs/temponet_db8_replication.md:20` sul ramo TEMPONet).

**Dati per una prova a encoder congelato.**
- Cache del lungo (T100-G 30M teacher, a 19.618, 39.237 e 49.046), per finestra:
  - `pre`: uscita del Perceiver, dopo tokenizer ed encoder locale pre-allenati. Ogni patch vede solo il supporto del tokenizer,
    circa 300 ms;
  - `lat`: dopo il backbone; ogni patch vede gia' l'intera finestra di 4 s, prima e dopo di se';
  - entrambe medie sui 64 slot, 160 × 384 in float16, piu' `pred` della testa e i bersagli.
  - Per dose: val circa 4,59 GB, train circa 474 MB. Copia verificata su `$WORK/wearusfm_runs/results/consolidation_long_20261105`
    (job 63781257).
- Il train del lungo e' lo stesso pannello alle tre dosi: 40 utenti, 200 registrazioni, 1.907 finestre, con le stesse firme dei
  bersagli del pannello storico (`results/infra/consolidation_long_20261105/execution/receipts/LONG-G49046-pose-train-20261105.json:197-205`).
- Non esiste una cache dei 64 slot: il protocollo slot la stima in circa 300 GiB.
- I checkpoint T100-G 30M a 19.618, 39.237 e 49.046 stanno su `$WORK`. Le altre cache e il checkpoint 300M solo su `$SCRATCH`.
- Il 300M a 19.618 ha la cache di posa estratta (64571999, `COMPLETED 0:0`) ma non ancora letta. La valutazione della linea 1 non e'
  conclusa: Simone ha approvato l'emendamento del criterio e l'invio diretto da GPT di un solo P1 (`release_plan.md` a `e4ebdbf`);
  l'unico P1-r2 e' 64620434, inviato alle 22:21 del 16/11 (`9629916` su `codex/scaling-p1-recovery-20261116`).
- Lettura di Claude: con le cache del lungo, una testa sui latenti congelati e la stessa testa sulla vista `pre` non richiedono
  nuove estrazioni GPU. Gli altri controlli richiedono calcolo nuovo: l'encoder random va estratto; la TCN da zero va allenata e
  TEMPONet raw va rivalutata sul supporto comune. Entrambe leggono l'EMG dai file processati di emg2pose, che le cache non
  contengono («processati» indica il formato dell'ingest, non il filtro del loader del FM). Quei file e i bersagli di posa di
  emg2pose stanno solo su `$SCRATCH`.

**Ultima lettura su TEMPONet raw** (registro `da5e820`, righe 12784-12790; descrittiva, su un utente).
- Utente 0219627c14 del test ufficiale `user`, gia' visto; 470 finestre.
- Correlazione per giunto fra stima con EMA e angolo, sull'intero test dell'utente: da 0,12 (pollice, IP) a 0,68, mediana 0,51 sui
  20 giunti. Nei 167 tratti di 20 s, sui 6 giunti che si muovono di piu', la mediana scende a 0,16. Nelle prese pollice-dita e
  nelle rotazioni del pollice la stima resta quasi piatta.
- Nessuna delle due correlazioni e' centrata per finestra di 4 s come quelle della tabella: non si confrontano con 0,40-0,47.
- E' una lettura sul test: non guida la scelta di varianti. Un'eventuale misura d'inseguimento va fissata senza usarla.

**Vincoli che restano.**
- Testa e loss si misurano separatamente (tua sintesi `8eba5e4`). La linea 3 (`db5ac41`, review di Claude `0ab9d98`) tiene la
  `PoseHead` fissa nei tre bracci.
- Lettura di Claude, non scritta nella linea 3: niente letture TEMPONet dentro i suoi nove contrasti prespecificati, perche'
  cambierebbero la famiglia del confronto. E niente sviluppo sul suo ramo o checkout: il digest della release copre tutti i `.py`
  di `src/wearusfm`.
- Il rilievo I-1 della review `0ab9d98` (righe 376-403) ha due cause: (1) sorgenti fissati per hash cambiati o file nuovi in
  `src/wearusfm`, e le guardie di sorgente scattano come devono; (2) chiavi nuove nel config o un campo nuovo in `FMConfig`, un
  falso allarme sulla ricetta. Le guardie girano nei preflight e nei test CPU (`scripts/tokenizer_preflight.py:93-97`,
  `tests/cpu/test_logsquare_train.py:36`). Secondo la review il problema scende a minore se il codice non tocca `src` ne' i config:
  per Claude e' la strada piu' semplice per il livello A. Il livello B tocca `FMConfig` se la testa entra nella configurazione del
  modello; la correzione minima di I-1 mostra un'alternativa (il componente lo aggiunge il trainer).
- La posa e' supervisionata nel pretraining: una testa migliore sulla posa non e' trasferimento a una regressione nuova
  (`docs/piano_scientifico_20261102.md:249-258`).
- Niente ricerca estesa di teste o lag sui 15 utenti val. Piano scientifico, punto 6c: «Intervenire solo su una precisa ipotesi
  con confronto separato».
- Il v10 §8 vuole fra le baseline la CNN/TCN da zero e lo stesso modello con encoder random e la stessa testa
  (`docs/fm_emg_reference_v10.md:1129-1137`).

**Calendario e risorse.**
- Leonardo tutto fermo dal 23/11 alle 08:00 al 25/11 alle 08:00.
- Prossimo rinfresco di `$SCRATCH` il 03/12, col via di Simone. Senza, i file rinfrescati il 12/11 scadono dal 22/12
  (`docs/decisioni.md` a `47ca96e`, ramo `worktree-claude-training-engine`; il CLAUDE.md di main riporta ancora il giro del 12/11).
- `$WORK`: 933,1 G usati su 1 T il 16/11 alle 22:11 (`lfs quota` nel preflight `e4ebdbf`,
  `results/infra/scaling_300m_20261104/p1_recovery_release_20261116/evidence/preflight/quota.txt`).
- Nessun budget registrato per questa attivita'. Ultimo saldo committato, del 09/11 alle 12:21: 1.308.137 ore locali consumate su
  2.000.000, residuo 691.863 fino al 07/01/2027 (`fccf88d`, ramo `claude/scaling-300m-exec-20261106`). Da allora hanno consumato
  almeno il training 300M a 19.618 e le estrazioni del 16/11: il residuo va riletto. Billing 8 ore locali per ora di un job a 1 GPU e
  8 CPU.
- L'autonomia sotto 5 GPU-ore vale per il lavoro gia' affidato, dentro il budget della campagna e con criteri fissati prima: qui
  mancano l'affidamento e il budget di Simone, e i criteri non sono ancora scritti.

## 2. Lettura di Claude, non approvata

Parere del 06/11, che resta un parere:
- Le teste non devono essere omogenee.
- Il problema non e' la taglia ma l'attribuzione: una testa capace prende per se' una parte del guadagno. Serve la stessa testa sul
  solo sEMG grezzo (o sul solo tokenizer) come controllo, con criteri fissati prima e selezione sui soli utenti train.
- Due livelli separati. (A) Lettore sui latenti congelati, economico. (B) Testa dentro il pretraining: cambia che cosa l'encoder e'
  spinto a codificare, e una testa grande puo' rendere i latenti meno leggibili linearmente; va deciso a parte.

Aggiunte di oggi, da valutare tu:
1. Partire da (A) sul 49.046, con le cache su `$WORK`.
2. Bracci a parita' di supporto, loss e dati: ridge istantanea (riferimento), MLP per istante di taglia simile alla TCN, TCN in
   stile TEMPONet sui latenti. Per separare contesto e non linearita', forse anche una cella lineare con contesto: una ridge con lag
   sullo stesso campo ricettivo, scelta come la ridge istantanea. Una TCN lineare con le larghezze di TEMPONet non basta: il primo
   blocco a 16 canali le da' rango al piu' 16, meno dei 20 angoli.
3. Controllo alla pari: la stessa TCN sulla vista `pre`, che controlla il backbone (non il pretraining), e una TCN da zero su sEMG
   con la stessa loss e lo stesso bersaglio per patch (medie a 200 Hz, ritardo zero); per gli utenti etichettati, vedi la domanda 4.
   TEMPONet raw (`model_final.pt`, nessun riallenamento) sullo stesso supporto e' solo un
   riferimento descrittivo: usa la L1, 158 utenti, l'EMG nelle sue unita' e solo 128 ms di passato. L'alpha della sua EMA e' stato
   scelto su questi 15 utenti: se ne riporta l'uscita grezza.
4. Loss fissa per tutti i bracci nuovi: l'errore quadratico sugli angoli standardizzati, come la ridge e il pretraining, non la L1 di
   TEMPONet.
5. Metriche: `E_dyn` primaria, piu' correlazione centrata, varianza predetta / bersaglio, livello e MAE; per giunto, con il pollice
   a parte.
6. Nel livello B una testa temporale imparerebbe anche a scavalcare i tratti nascosti dalla maschera; il livello A non vede questa
   differenza fra training e lettura.

## 3. Cosa ti chiedo

1. **Domanda e livello.** Parti dal livello A come «primo controllo economico», con una testa derivata da TEMPONet come «testa
   temporale compatta»? Se no, perche', e cosa al suo posto.
2. **Bracci e controlli.** Quali, e di che taglia. In particolare se servono: la cella lineare con contesto, la vista `pre`,
   l'encoder random con la stessa testa, la TCN da zero su sEMG grezzo, TEMPONet raw come riferimento descrittivo.
3. **Architettura.** Una specifica completa e unica, decisa prima, non una ricerca:
   - uscita per patch o finestra scorrevole, e come si trattano i bordi del crop6;
   - campo ricettivo in patch e in ms, causale o simmetrica;
   - stride, pooling, dilatazioni e canali, compreso il collo di bottiglia del primo blocco;
   - normalizzazione d'ingresso (BatchNorm o LayerNorm), dropout, numero di parametri.
4. **Dati e selezione.**
   - Etichette: backbone e testa allenata hanno visto la posa dei 158 utenti train nel pretraining, e TEMPONet raw pure. Le teste
     nuove sulle cache vedono solo i 40 del pannello. Quale parita' scegli (per esempio 40 utenti per tutte le teste nuove, 158 per
     i modelli da zero) e come la dichiari? Oppure serve una nuova estrazione su altri utenti train?
   - Overfitting: con 1.907 finestre e circa 80.000 parametri, quale regolarizzazione e quale arresto? Si scelgono sui fold, che
     pero' sono utenti gia' visti dall'encoder: la stima e' ottimista.
   - Quante letture dei 15 utenti val.
   - Conferma: i 15 val sono sviluppo, e il test ufficiale e' gia' stato letto da TEMPONet raw. Quale insieme fa da conferma? Oppure
     si dichiara che non c'e'.
5. **Checkpoint.** 49.046, 19.618 o entrambi. Il 300M resta fuori finche' la linea 1 non pubblica la sua lettura?
6. **Metriche e criteri**, da congelare in `docs/decisioni.md` prima di qualunque job.
   - Primario, soglia (basta il -2% pratico?) e IC. Riferimento: ridge, testa allenata o entrambe.
   - Famiglia dei contrasti fissata prima, con la correzione per molteplicita'; nessun vincitore scelto sulla val.
   - Regola di lettura dei controlli: quando un guadagno va al backbone e quando alla testa.
   - Quale esito chiude la domanda: anche il «no» va scritto prima.
   - Se aggiungere una misura di inseguimento per tratto.
7. **Confronto con TEMPONet raw.** Come rendere comune il supporto: istanti puntuali sulla griglia a 2 kHz, uno ogni 10 campioni
   (200 Hz) nel protocollo ufficiale, contro medie per patch da 25 ms; e 255 campioni di contesto dentro le finestre da 4 s.
8. **Livello B.** Va proposto ora, dopo l'esito di A, o mai? Se si':
   - cosa resta fisso (la tua sintesi: inizializzazione dell'encoder, dati, loss, dose e schedule); la `PoseHead` nasce sotto
     `fork_rng`, e una testa nuova deve lasciare invariata l'inizializzazione del resto;
   - come si legge: con le teste proprie e con lettori comuni rifittati;
   - quale controllo appaiato: un braccio `PoseHead` nuovo, oppure il C0 della linea 3 se il percorso di calcolo e' identico. Il
     T100-G storico non basta: un training nuovo non lo riproduce bit per bit, e fra due bracci storici `E_dyn` scarta dello
     0,47-0,90% (review `0ab9d98`, D-4);
   - come si evita di interferire con la linea 3 (ramo, guardie I-1, digest);
   - quanto costa: un training per ogni testa nuova, piu' il braccio `PoseHead` di controllo se non si usa il C0 della linea 3.
     Riferimento storico 16,94 GPU-ore per un training T100-G 30M a 19.618, prima dell'overhead (`db5ac41`). Se B si fa a 19.618,
     A va letto anche a 19.618;
   - se servono nuove deroghe: a v10 §6.5 («Piccole») o a §3.7.
9. **Codice e operatore.**
   - Se Simone affida a GPT il codice: come useresti `temponet.py`. Con un modulo nuovo che lo adatta, in un ramo di lavoro diverso
     da quello della linea 3, oppure con l'integrazione del ramo di Claude come passo separato. I file del ramo di Claude non si
     modificano.
   - Quali test CPU nuovi servono.
   - Chi proponi come autore del codice e chi come operatore (uno solo, con submitter e ledger comuni). Lo decide Simone.
10. **Preventivo.** Risorse, walltime e costo in GPU-ore, core-ore CPU e ore locali, con il residuo e un tetto proposto per voce;
    anche per i job solo CPU, perche' 0 GPU-ore non e' costo nullo. Dove vanno le uscite, fuori dal repo. Come si conserva cio' che
    serve oltre il 22/12. Calendario, con la manutenzione.

## 4. Cosa non ti chiedo

- Nessun job, deploy, checkout o invio.
- Nessuna modifica a `PoseHead`, alle campagne chiuse, alle linee 1, 2 e 3 o ai rami di Claude.
- Nessun uso del test ufficiale di emg2pose per scegliere varianti.
- Nessun merge del ramo TEMPONet, ne' su main ne' sul tuo ramo: il registro dice «l'integrazione spetta a GPT, se richiesta»
  (`da5e820`, righe 8720-8721), e questo prompt non la richiede.

## 5. Come rispondere

- Proposta di protocollo in `docs/` sul tuo ramo, in stato REVIEW_REQUIRED, con la voce nel registro. Push su origin e verifica
  dello SHA remoto.
- Criteri scritti in modo da poterli congelare cosi' come sono. A parte, l'elenco delle decisioni per Simone: livello, checkpoint,
  bracci, budget, affidamento del codice, operatore.
- Se non sei d'accordo con la lettura di Claude, dillo e motivalo: non concordare per cortesia.
- Poi Claude rivede la proposta. Dopo la firma di Simone, il codice lo rivede prima del deploy l'agente che non lo ha scritto
  (revisione reciproca).
