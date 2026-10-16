# Calibrazione C1-C4: ammissibilita', validation interna e selezione (congelata prima di C1)

**Stato: FIRMATO da Simone il 16/10/2026** («firmo A4 e parità, prepara il preventivo del collaudo su w1cat»; il 16/10 anche «in C3 con un
solo LR ammissibile ci si ferma (come dice l'accordo), salvo diversa firma mia»). Proposta di Claude del 15/10/2026, rivista dal subagent
`critico` il 16/10; GPT la rivede quando rientra. Budget firmati il 16/10: calibrazione 220 GPU-ora, finestra 1 6.000 (`docs/decisioni.md`). L'accordo del
15/10/2026 (`docs/accordo_operativo_finestra1_20261015.md`, «Calibrazione» e prompt GPT punto 5) assegna questo protocollo a GPT. Simone:
«come al solito chat gpt si è mangiato tutti i token , riessci tu a fare i task per C1 C4 ?» e «mi raccomando se trovi sistema tutto».
Nessun file dell'harness e' stato modificato: il codice nuovo sta in file nuovi che usano l'harness in sola lettura.

Le soglie qui sotto si congelano **prima di guardare** (CLAUDE.md): dopo la firma si registrano in `docs/decisioni.md` e si committano prima
di C1; un cambio dopo C1 e' una decisione nuova, mai una correzione silenziosa. Le parti marcate *interpretazione* sono letture mie
dell'accordo, da confermare.

## Gia' fissato dall'accordo (non si decide qui)

Obiettivo A (JEPA target b + ancore 0,2; keep e rami spenti), batch globale 32, un nodo/4 GPU, prefetch, manifest derivato al 100%, WSD D14
con warmup da D = 4 epoche (1.962 passi) e decay sul 20% finale di ogni endpoint; griglia C1 30M 1e-4 / 3e-4 / 1e-3, 1 epoca, seme 0; C2 i due
finalisti, seme 1; C3 100M, 0,5 epoche; C4 300M, 0,25 epoche, il LR proposto. Selezione con validation interna alle partizioni train delle
sonde P1; nessun test ufficiale P1/P2 e nessun benchmark P2; il pannello fisso del trainer e' diagnostica di stabilita', non validation.
Se 3e-4 e' stabile e' fra i finalisti con la migliore alternativa stabile; se no, le due alternative solo se entrambe ammissibili. Meno di
due candidati, una conferma fallita o C4 instabile: ci si ferma e si registra, senza sostituire LR ne' allargare la griglia. La training loss
non sceglie mai.

## Validation interna (`scripts/calibration_probe.py`)

- **Soggetti:** solo quelli di pretraining di NinaPro DB2, DB3, DB6 negli split firmati (32, 8, 8): sono le partizioni train della P1.
  I soggetti di test (8, 3, 2) non si aprono mai: l'estrazione riceve solo i soggetti di pretraining e verifica che coincidano; il codice
  rifiuta uno split con un soggetto in entrambi. Dei test si usano solo i **conteggi**, per i pesi dei dataset.
- **Feature e sonda:** quelle della P1, dallo stesso codice: `probe_p1.dataset_features` (teacher del checkpoint post-decay, feature
  "model", finestre da 1 s, scala per sessione) e `harness.probe.linear_probe` (logistica bilanciata, C sulla griglia fissa scelto sui
  soggetti di validazione, normalizzazione stimata sul train).
- **Fold:** per dataset i soggetti (ordinati) si permutano col seme 20261015 e si distribuiscono a turno in **K = 4 fold** (DB2 8 per fold,
  DB3 e DB6 2). Per il fold k: valutazione sui suoi soggetti; C scelto su `val_frac` = 0,1 dei rimanenti (per eccesso, almeno 1; seme
  [20261015, k]); train sugli altri (DB2 21, DB3 e DB6 5). Ogni soggetto e' predetto una volta da una sonda che non l'ha visto. Stessi fold
  per tutti i candidati: confronto accoppiato.
- **Metrica S:** per dataset l'accuratezza bilanciata delle predizioni fuori fold (classi assenti dal train del fold tolte e contate);
  S = media pesata coi pesi della P1 ("subjects", proporzionali ai soggetti di test: 8/13, 3/13, 2/13); errore standard per bootstrap sui
  soggetti, stratificato per dataset (1.000, seme 0). *Interpretazione:* uso i pesi della P1 perche' S stimi la stessa quantita' della P1
  ufficiale su soggetti mai visti dalla sonda.
- **Descrittivi, non criteri:** rango effettivo e colonne costanti delle feature.

## Ammissibilita' di un run (`scripts/calibration_select.py`; tutte)

| | Criterio | Origine della soglia |
|---|---|---|
| A1 | Completo: nessuno STOP, `summary.json` con `stopped = max_steps` e i passi pianificati; la sonda viene dal suo `checkpoint_postdecay.pt` | regola della guardia, gia' in uso |
| A2 | Ogni valore numerico di `metrics.jsonl` e `fixed_eval.jsonl` finito | gia' in uso (un non finito ferma il run) |
| A3 | Non-collasso: nessun allarme nel run; all'ultima valutazione (passo finale) rapporto di collasso min(studente, teacher) >= 0,05 e rango effettivo dello studente >= 10% della dimensione | **firmate**, decisione 16 (`AlarmRule`) |
| A4 | Stabilita' sul pannello fisso (8 x 32 finestre, ogni **500** passi dal passo 0, senza buchi): mai **3** valutazioni consecutive con L_t > **1,10** x min_{s<=t} L_s; L all'ultima valutazione <= **0,9** x L_0 (L = loss totale) | **firmate da Simone il 16/10/2026** (dalla bozza v2 del 14/10, `docs/regola_stabilita_bozza.md`) |

- A4 richiede il pannello acceso nei run di calibrazione: il pianificatore aggiunge `--fixed-eval-every 500` (8 forward senza gradiente ogni
  500 passi, sotto l'1%). Tre valutazioni consecutive = 1.500 aggiornamenti di regressione persistente; una o due sopra soglia si
  registrano senza escludere.
- *Interpretazione:* «stabile» dell'accordo = ammissibile secondo A1-A4.
- A1 confronta il checkpoint della sonda con quello del run come file, non come stringa: percorsi uguali dopo la normalizzazione (barre doppie
  o finali, `.`) o, se esistono, lo stesso file su disco (link simbolici, `$WORK` contro `/leonardo_work/...`). Correzione del 16/10/2026:
  prima una barra finale in `run_root` rendeva inammissibili tutti i run.
- Limiti noti: il pannello misura la regressione sui dati di training, non la generalizzazione; la parte JEPA ha target mobili (teacher
  EMA), quindi la soglia sta sulla loss totale, dominata dalle ancore.

## Selezione (solo S; confronto accoppiato sui soggetti)

Per due LR, Delta = S(a) - S(b) e il suo errore standard da un bootstrap accoppiato (stesso ricampionamento dei soggetti per i due
candidati; finestre, etichette e soggetti identici, controllato). **|Delta| <= 1 SE = parita' e vince il LR piu' basso** (firmata da Simone
il 16/10/2026). Motivo: nella scala da 30M a 300M il LR ottimo tende a scendere con la taglia; a parita' il LR piu' basso e' la scelta prudente per C3/C4.

- **C1:** se 3e-4 e' ammissibile: finalisti 3e-4 e la migliore fra le alternative ammissibili (se sono due, confronto accoppiato); se
  nessuna alternativa e' ammissibile, stop. Se 3e-4 non e' ammissibile: 1e-4 e 1e-3 solo se entrambi ammissibili, altrimenti stop.
- **C2:** confermati = i finalisti ammissibili al seme 1. Se uno manca, C3 e C4 si fermano (gia' nel pianificatore). S al seme 1 e il
  confronto si registrano, un'inversione dell'ordine non e' un fallimento.
- **C3:** LR proposto per il 300M = il migliore fra i due LR ammissibili al 100M; **meno di due ammissibili (uno o nessuno) = stop**, come
  dice l'accordo: «Se mancano due candidati ammissibili [...] interrompere il passaggio alla fase successiva». Correzione del 16/10/2026
  (Simone: «in C3 con un solo LR ammissibile ci si ferma (come dice l'accordo), salvo diversa firma mia»): la prima versione continuava con
  l'unico LR ammissibile. Continuare con uno solo richiede una firma di Simone e un cambio di `calibration_select.py`.
- **C4:** superata se ammissibile. Un solo punto al 300M non dimostra l'ottimo del rung (accordo): il LR della campagna resta una decisione
  di Simone sui risultati di C1-C4.

Le decisioni vanno nel file del pianificatore (`scripts/calibration_plan.py --decisions`) col comando `calibration_select.py merge`, che
rifiuta fasi riscritte o fuori sequenza e un LR per il 300M che non sia uno dei due di C3: `{"C1": {"finalists": [...]} | {"stop": ...},
"C2": {"confirmed": [...]}, "C3": {"proposed_300m_lr": lr} | {"stop": ...}, "C4": {"lr": lr, "passed": bool}}`.

## Catena di valutazione di una fase (`calibration_plan.py --out`, file `valutazione_<fase>.json`, schema eval-plan-v0)

Per ogni run: guardia del checkpoint post-decay (passi pianificati) -> estrazione (`scripts/slurm/calibration_extract.sbatch`, 1 GPU,
`boost_usr_prod`/`normal`, tetto 30 min al 30M, 1 h al 100M, 2 h al 300M) -> sonda a fold (`calibration_probe.sbatch`, `lrd_all_serial`, 8
core, tetto 3 h); poi la selezione della fase (`calibration_select.sbatch`, 2 core, 30 min) dopo tutte le sonde. Limiti: 8 job GPU, 8 seriali.
Feature su `$SCRATCH`, rapporti e decisioni in `$WORK/wearusfm_runs/results/`; la decisione entra nel repo solo dopo la copia sul Mac, il
`merge` e il commit, e la fase successiva parte solo col via di Simone.

## Preventivo (dal pianificatore; stime delle valutazioni da rivedere con le prime misure)

| Fase | Training GPU-ora attese / tetto | Valutazione GPU-ora attese / tetto | CPU tetto (core-ora) |
|---|---|---|---|
| C1 (3 run, 30M) | 60,3 / 78,0 | 0,9 / 1,5 | 74 |
| C2 (2 run, 30M) | 40,2 / 52,0 | 0,6 / 1,0 | 50 |
| C3 (2 run, 100M) | 39,8 / 52,0 | 1,3 / 3,0 | 66 |
| C4 (1 run, 300M) | 19,9 / 26,0 | 1,5 / 3,0 | 33 |
| **Totale** | **160,2 / 208,0** | **4,3 / 8,5** | **223** |

Totale **~164,5 GPU-ora attese, tetto 216,5** (1.732 ore locali), nel budget firmato di 220; C1 da sola **~61 attese, tetto 79,5** (636 ore
locali). Valutazioni aggiornate il 16/10 con le misure di `w1_selezione_v1` (al 30M; 100M e 300M stimati). Voce della
finestra 1: il budget della finestra 1 non e' ancora fissato, quindi serve una decisione di Simone sul budget prima di C1 (non il residuo
infrastruttura). Basi delle stime di valutazione: estrazione P1 al 30M 13:42 per 61 soggetti, sonde P1 22 min a 8 core
(`docs/step1_downstream_ab.md`); qui 48 soggetti e 4 fold.

## Prima di C1

1. ~~Firma di Simone su A4 e sulla regola di parita'~~: firmate il 16/10/2026 (C3 con un solo LR ammissibile: stop, firmato lo stesso
   giorno). Resta il budget della finestra 1.
2. Deploy di un commit con questi file (`deploy/<nome>`) e checkout di campagna (job CPU), percorsi del manifest derivato al 100% sul cluster
   (SHA del file del cluster), scale e `window_seconds`.
3. Facoltativo, consigliato: collaudo della catena reale (estrazione, sonda, selezione) su un checkpoint gia' esistente (`w1cat`), per
   provare launcher e tempi sul cluster prima di spendere le GPU-ora di C1 (il collaudo controlla la meccanica, non l'ammissibilita').

Test: `tests/cpu/test_calibration_selection.py` (fold, sonda sintetica, estrazione che apre solo i soggetti di pretraining, ammissibilita',
regole di C1-C4 con parita', unione delle decisioni, launcher), `tests/cpu/test_calibration_plan.py` (stop, pannello, piani di
valutazione validati dall'orchestratore).

## Collaudo della catena reale `w1_selezione_v1` (16/10/2026; Simone: «firmo A4 e parità, prepara il preventivo del collaudo su w1cat»)

**Scopo:** provare sul cluster, prima di spendere le GPU-ora di C1, i pezzi nuovi della catena con checkpoint veri: launcher di estrazione,
sonda e selezione, lettura dei file reali del trainer, accoppiamento delle finestre fra due checkpoint diversi (condizione del confronto
accoppiato), tempi e memoria per correggere il preventivo di C1. Controlla la meccanica, non l'ammissibilita'.

**Piano:** i due post-decay del collaudo delle catene, `w1cat-T` (1.400 passi, ultimo job `w1cat-T-j03`) e `w1cat-B` (ramo, 1.375 passi,
`w1cat-B-j02`), 30M. Per ciascuno guardia (passi minimi e SHA-256 gia' misurati nel collaudo `w1val`) -> estrazione (1 GPU, QoS di debug,
30 min) -> sonda a fold (8 core, 3 h); poi la selezione nella fase `collaudo` (candidati con un'etichetta; nessuna scelta, solo gli
ammissibili e i confronti accoppiati descrittivi). Feature in `$SCRATCH/wearusfm_runs/w1sel_1610/`, rapporti in
`$WORK/wearusfm_runs/results/infra/w1sel_1610/`. Codice da un riferimento `deploy/w1sel-20261016` e da un checkout di campagna nuovo.

**Criteri (congelati qui, prima del lancio):**
1. Guardie: entrambe COMPLETED 0:0 col rapporto `ok` e gli SHA-256 attesi (T 15262e10..., B 71af16a2...), sottomesse senza afterok (i
   training sono finiti da ieri secondo sacct).
2. Estrazioni: COMPLETED 0:0 entro i 30 minuti; per ciascun checkpoint tre npz col protocollo `w1-calibrazione-selezione-v1`, i soli
   soggetti di pretraining (32, 8, 8), lo SHA-256 del checkpoint uguale a quello della guardia, 768 colonne (2 x 384).
3. Sonde: COMPLETED 0:0 entro le 3 ore; S ed errore standard finiti, 0 < S < 1; 4 fold per dataset; ogni soggetto di pretraining predetto
   fuori fold.
4. Selezione: COMPLETED 0:0; il registro riporta l'ammissibilita' di entrambi letta dai file reali. **Esito atteso: nessuno dei due
   ammissibile**, per il pannello fisso assente (`w1cat` non l'aveva acceso): un «ammissibile» sarebbe un difetto. Il confronto accoppiato
   descrittivo e' calcolato, quindi finestre, etichette e soggetti coincidono fra i due checkpoint.
5. Rieseguendo la catena dopo la fine: nessuna sottomissione in piu'; nessun file scritto nel checkout.

Un nodo che fallisce ferma quelli a valle (afterok); si analizza, nessun rilancio alla cieca. Le durate e la memoria misurate (sacct)
aggiornano le stime e i tetti delle valutazioni di C1 con un commit prima di C1.

**Risorse e costo:** riferimento di deploy (sola scrittura del ref, preflight in sola lettura) e checkout di campagna (job CPU, 1 core,
pochi minuti); 2 guardie CPU (1 core, tetto 20 min); 2 estrazioni su 1 A100 con QoS `boost_qos_dbg` (tetto 30 min ciascuna); 2 sonde
(8 core, tetto 3 h); 1 selezione (2 core, tetto 30 min). **GPU: attese ~0,4 GPU-ora (3,2 ore locali), tetto 1,0 GPU-ora (8 ore locali)**,
dal budget «preparazione dell'infrastruttura»: 9,26 usate su 25, residuo 15,74, dopo il tetto almeno 14,74. **CPU: attese ~20 core-ora,
tetto ~51 core-ora.**

**Lancio (16/10/2026; Simone: «procedi con tutti e 3 i punti e vai, lancia il collaudo su w1cat»).** Preflight in sola lettura pulito
(nessun hook attivo, HEAD su main, nessun ref o checkout `w1sel`); `deploy/w1sel-20261016` -> 7cf4f08; checkout 60688786 COMPLETED 0:0
(12 s, «checkout di campagna pronto»). Catena sottomessa una volta: guardie 60688822 (T) e 60688823 (B) senza afterok (training finiti
secondo sacct), estrazioni 60688824 (T) e 60688827 (B) su `boost_qos_dbg`, sonde 60688826 e 60688829, selezione 60688830.

**Esito (16/10/2026): SUPERATO** (`results/infra/w1sel_20261016.json`, rapporti in `results/infra/w1sel_1610/`). Codice `7cf4f08`.
1. Guardie 60688822/60688823: `ok`, SHA attesi, senza afterok.
2. Estrazioni 60688824 (17:51) e 60688827 (16:34), entro i 30 minuti: soli soggetti di pretraining (32/8/8), SHA del checkpoint uguale a
   quello della guardia, 768 colonne; memoria ~10 GB.
3. Sonde 60688826 (1 h 44) e 60688829 (1 h 51), entro le 3 ore: S = 15,08 ± 0,53% (T) e 15,13 ± 0,54% (B); DB2 ~15%, DB3 ~2,3% (49
   classi, caso 2%), DB6 ~35% (7 classi); 4 fold, ogni soggetto predetto una volta fuori fold.
4. Selezione 60688830: `{"collaudo": {"admissible": []}}`, entrambi inammissibili per il pannello fisso assente (atteso: `w1cat` non
   l'aveva acceso); confronto accoppiato descrittivo calcolato (B - T = +0,05 punti, errore standard 0,10: parita'), quindi l'accoppiamento
   delle finestre fra checkpoint regge sui dati veri.
5. Seconda chiamata della catena: 7 nodi gia' sottomessi, nessuna sottomissione; checkout in sola lettura senza file nuovi.
**Misure per C1:** estrazione al 30M ~17 min (stimavo 12), sonda ~1,8 h a 8 core (stimavo 1,2); le sonde girano UNA alla volta, perche'
sulla seriale c'e' un limite di core per utente (`QOSMaxCpuPerUserLimit`): con 3 run, ~5,5 h di sonde in fila dopo il training.
**Consuntivo: 0,57 GPU-ora** (due estrazioni da 1 GPU) sul budget infrastruttura, ora 9,83 usate su 25 (residuo 15,17); ~29 core-ora CPU.
