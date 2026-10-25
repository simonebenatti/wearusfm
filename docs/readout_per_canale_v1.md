# Readout per canale sui checkpoint esistenti — protocollo `readout_per_canale_v1`

**Stato: lettura CONGELATA in 63ac0dd (23/10/2026, prima di aprire i rapporti della baseline classica), emendata nel commit successivo
dopo la seconda review del `critico` e dopo l'apertura della baseline (emendamenti elencati in fondo, nessuno cambia la regola di lettura).
Nessun job lanciato al congelamento.** **Campagna `readout-canale-20261023` eseguita (deploy 6242bc3, autorizzata il 23/10) e CHIUSA il
25/10/2026: 30 job COMPLETED, 2,74 GPU-ora e 238,9 core-ore; confronti, riepilogo delle 21 sonde e consuntivo in
`results/readout_canale_20261023/` (README con la tabella dei verdetti); risposte in `docs/sintesi_pretraining_20261024.md`.**
Richiesta di Simone: «Parti dal readout per canale sui checkpoint esistenti (3.1), insieme alla baseline classica; il pilot degli obiettivi
dopo». Origine: `docs/diagnosi_pretraining_20261023.md`, causa 3 (il readout delle sonde media sui canali e sui latenti e cancella *quale*
elettrodo e' attivo). Codice: `src/wearusfm/harness/channel_readout.py`, `scripts/channel_readout.py`, launcher
`scripts/slurm/channel_readout_{extract,probe}.sbatch`, test `tests/cpu/test_channel_readout.py`.

## Domanda

Quanta informazione dei compiti P1/P2 c'e' nell'encoder congelato di A che il readout medio (`pool_p1p2`: 2d numeri per finestra) non fa
arrivare alla sonda lineare? E l'encoder, letto nel modo migliore, batte le feature classiche per canale? Lettura descrittiva per il
resto; nessuna adozione, nessun cambio delle sonde ufficiali P1/P2 da questo protocollo.

## Cosa si misura (nessun training, nessun parametro nuovo)

Dalla stessa passata in avanti senza maschera (identica a `encode`, provata nei test), per finestra e per canale, la **media nel tempo** di:

| strato | cosa | larghezza |
|---|---|---|
| `pool` | il readout di sempre (controllo d'uguaglianza con le cache e i rapporti storici) | 2d = 768 |
| `local` | uscita dell'encoder locale (target dell'opzione a di D11) | C x d (Myo 3.072; DB2/DB3 4.608; DB6 6.144) |
| `dec` | decoder a query del teacher sulla vista senza maschera, interrogato su ogni (canale, patch) valida: il target JEPA dell'opzione b esteso a tutte le posizioni, la rappresentazione profonda per canale (le teste delle ancore leggono invece il decoder dello studente sulla vista mascherata, alle sole posizioni nascoste) | C x d |

Canali scartati dal QC o senza patch a 0 (come la baseline classica). Niente strato «token» prima dell'encoder locale: il front-end e'
lineare nel segnale e la sua media nel tempo su un segnale a media nulla e' ~0 (resterebbe solo l'identita' di canale, costante).
Preprocessing, finestre, scala, ritagli e soggetti sono quelli della validazione interna della calibrazione (`calibration_probe`):
P1 = soggetti di pretraining di DB2, DB3, DB6 (finestre da 1 s, scala per sessione); P2 = soggetti di EPN-612 e UCI-EMG fuori dai test dei
tre split (finestre dei caricatori della P2, montaggio myo8). L'impronta degli ingressi (`input_sha256`) e' la stessa delle cache di S dei
bracci: le righe sono accoppiabili una a una (test).

## Sonda: la stessa, a larghezza pari

Stessi fold di soggetti (4, seme 20261015), stessa scelta di C sui soggetti di validazione, stessi pesi (P1: 8/13, 3/13, 2/13; P2: media
dei due dataset), stesso errore standard (`calibration_probe`), **stessa funzione di sonda** (`harness.probe.linear_probe`). Differenza,
dentro ogni fold e sul solo train: standardizzazione delle colonne, poi **PCA a k componenti** (randomized, seme della sonda), poi la
logistica di `linear_probe` **senza ristandardizzare** (`scale=False`: la penalita' L2 resta nello spazio delle colonne standardizzate).
Il parametro `scale` e' aggiunto a `linear_probe` il 23/10/2026 con default invariato; il file era vincolato byte per byte ai rapporti
della diagnostica dei rami (`tests/cpu/test_branch_pilot.py`): il vincolo si aggiorna per decisione di Simone («adesso le modifiche
principali sono quelle che fai tu»), e il percorso di default della sonda, quello del pilot e della diagnostica, non cambia (test
d'uguaglianza). Con k >= larghezza non si riduce nulla e la sonda riceve le colonne come nella calibrazione (uguaglianza provata nei test:
`pool` da' la stessa S, gli stessi C e le stesse predizioni di `calibration_probe`; un test controlla che scaler e PCA vedano solo le righe
di train del fold). **k effettivo = min(k, righe di train del fold - 1)** (`channel_readout.effective_k`): le feature standardizzate e
centrate di n righe hanno rango al piu' n - 1, e con k >= rango la PCA e' una base dello spazio dei dati di train, dove la soluzione L2
coincide con quella a larghezza intera. Su **UCI-EMG** (19 soggetti interni, ~220-240 finestre di train per fold) il tetto scatta sempre:
li' la «pari larghezza» non vincola nulla e il readout per canale e' di fatto a larghezza intera, dichiarato nel rapporto (`k_capped`,
`k_by_fold`); su EPN-612 (~35.000 finestre di train) e sui dataset della P1 no. **k = 2d = 768** per default: la larghezza del readout medio,
cosi' il confronto non premia la sola larghezza; `--k 0` = larghezza intera (in questa tranche solo sulla P2, 3.072 colonne: le sonde
di S della P2 a 30M durano 11-12 min, quindi ~4x resta sotto un'ora; sulla P1, 4.608-6.144 colonne per 6-8x di 37-39 min, supera le 4 h
della seriale: tranche 2, divisa per dataset). La PCA toglie direzioni a bassa varianza che potrebbero essere discriminanti: e' un limite
dichiarato del confronto a pari larghezza, non un pregio; la larghezza intera sulla P2 dice quanto costa.

## Confronti e lettura (da congelare col commit del lancio, prima di qualunque sonda)

Tutti coi due semi di A e accoppiati per finestra e per modello (`compare`: stessi `input_sha256` e `oof_rows_sha256` per dataset,
stesso checkpoint per seme; i rapporti della baseline classica, senza modello, valgono per tutti e due i semi), su S della P1 e della P2,
con la **regola dei bracci cosi' com'e' implementata** (decisioni.md 16/10/2026; `arms_compare.s_comparison`,
`downstream_ab.paired_component`: soggetti ricampionati insieme per i quattro rapporti, 1.000 ricampionamenti, seme 20261009, intervallo
dei percentili): «migliora» = delta medio dei due semi >= +0,01, limite inferiore dell'intervallo al 95% > 0 e delta > 0 in tutti e due i
semi; «peggiora» la speculare; altrimenti «regola non superata (stima, intervallo o accordo dei semi insufficienti): non esclude una
differenza, neanche oltre 1 punto». Il bootstrap congiunto tiene conto che i quattro rapporti condividono i soggetti (e, nel confronto 2,
il riferimento): niente errori standard sommati come indipendenti.

1. **Per canale contro readout medio** (la domanda): candidati `local` e `dec` a k = 768, riferimento i rapporti storici di S di A
   (`$WORK/wearusfm_runs/results/w1arm/W1ARM-A-30M-s{0,1}/s_p{1,2}.json`, readout medio: s0 P1 18,34 ± 0,67, P2 81,92 ± 1,84; s1 P1
   18,28 ± 0,72, P2 80,75 ± 2,20, letti sul cluster il 23/10), per compito: quattro letture (2 strati x 2 compiti), senza correzione per
   confronti multipli (dichiarato). Risposta: «il readout medio nasconde informazione» se almeno uno dei due strati «migliora» in un compito
   senza «peggiorare» nell'altro. La larghezza intera sulla P2 (`--k 0`) e' un descrittivo accanto: dice quanto la PCA a 768 costa, non
   entra nella risposta.
2. **Encoder contro baseline classica:** un solo candidato, scelto **prima** di aprire i rapporti della baseline: fra `local` e `dec` a
   k = 768 quello con la S della P1 media dei due semi piu' alta, se supera la S della P1 media del `pool` storico (18,31); altrimenti il
   `pool` storico; a parita' il `pool`. Contro `interna_p1.json` / `interna_p2.json` della baseline
   (`$SCRATCH/wearusfm_runs/baseline_td_20261023/p{1,2}/`; stesso rapporto per i due semi), stessa regola. Risposta: «l'encoder batte le
   feature classiche» solo se «migliora» in tutti e due i compiti. **Distorsione dichiarata:** la P1 serve sia a scegliere il candidato
   sia a confrontarlo, quindi la sua lettura e' ottimistica (al piu' per la scelta fra due strati); la P2 non entra nella scelta.
3. **Descrittivi:** `local` contro `dec` (dove sta l'informazione), varianza spiegata della PCA per fold, rango effettivo (su 8.192 righe),
   S per dataset (DB3 a parte), C scelti.
4. **Controllo d'uguaglianza:** `pool` di A s0 sulla P1 contro la S storica di s0 (0,1834, `s_p1.json`): `channel_readout.py --stage
   equality` (stesse finestre e righe, stesso checkpoint), **|delta S| <= 1e-3** e conteggio delle predizioni diverse (bf16 su GPU in tutti e
   due i casi: qualche ulp puo' spostarne alcune, si dichiara). Oltre 1e-3 il comando esce con 1 e nessuna lettura dei punti 1-3 finche'
   non si capisce la differenza.

Non si ritocca la lettura dopo i numeri; la seconda tranche (sotto) si decide prima, non in base ai risultati della prima. Questo confronto 2
sostituisce il punto (4) della voce della baseline classica in decisioni.md («lettura solo affiancata; un confronto accoppiato, se servira',
con `load_oof` e `paired_difference`»): il confronto accoppiato si fa con `s_comparison` e la regola dei bracci, e resta descrittivo
rispetto alla baseline (nessuna soglia di adozione).

**Opzione proposta a Simone (ablation, da decidere prima di lanciare): ancore-only s0/s1 nella stessa campagna.** Il verdetto «ancore-only =
A» dei bracci e' stato preso col readout medio; se col readout per canale A supera ancore-only (stessa regola, riferimento ancore-only,
candidato A), il JEPA aggiunge informazione che la media non mostra e la diagnosi cambia; se restano uguali, si rafforza. Costo: 4 estrazioni
GPU in piu' (P1 e P2 per i due semi: tetto 5,0 GPU-ora, attese ~2,3) e 8 sonde CPU a k 768 (s0/s1 x P1/P2 x {local, dec}: tetto 256 core-ore,
attese ~40). Lettura: la stessa regola dei bracci, descrittiva; nessuna adozione.

## Checkpoint e tranche

- **Tranche 1 (questa):** A s0 e A s1 post-decay (teacher), gli stessi della valutazione dei bracci
  (`$SCRATCH/wearusfm_runs/runs/w1arm/W1ARM-A-30M-s{0,1}/checkpoint_postdecay.pt`, SHA-256 nell'esito dei bracci).
- **Tranche 2 (da decidere con Simone prima di leggere la 1):** rete casuale s0 con l'architettura di A (`scripts/random_checkpoint.py`,
  gia' previsto dalla baseline; non e' il passo 0 di A) con gli stessi strati, e larghezza intera sulla P1 divisa per dataset.

## Job e costi (da autorizzare)

Misure di riferimento (sacct dei bracci a 30M, 1 GPU, 8 core, letti il 23/10): estrazione di S della P1 11 min, della P2 24 min; sonda di S
della P1 37-39 min, della P2 11-12 min. Qui l'estrazione aggiunge il decoder su ogni (canale, patch) e due strati per canale (stima: x2);
la sonda a k = 768 aggiunge la PCA (~1-3 min per fold e dataset) e la lettura di cache piu' larghe.

| Passo | Job | Risorse | Tetto | Atteso |
|---|---|---|---|---|
| Checkout di campagna `readout-canale-20261023` | 1 CPU (`campaign_checkout.sbatch`) | secondi | — | — |
| Estrazione P1 (A s0, A s1) | 2 GPU (`channel_readout_extract.sbatch`) | 1 A100, 8 core, 100 GB, 1 h | 2,0 GPU-ora | ~0,7 (~20 min l'una) |
| Estrazione P2 (A s0, A s1) | 2 GPU, `--time=01:30:00` | 1 A100 | 3,0 GPU-ora | ~1,6 (~45 min l'una) |
| Sonde a k = 768: s0 P1 {pool, local, dec}, s1 P1 {local, dec}, s0 e s1 P2 {local, dec} | 9 CPU seriali (`channel_readout_probe.sbatch`) | 8 core, 30 GB, 4 h | 288 core-ore, 0 GPU-ora | P1 ~50 min, P2 ~15 min l'una: ~40 core-ore |
| Sonde a larghezza intera sulla P2 (`--k 0`): s0 e s1 {local, dec} | 4 CPU seriali | idem | 128 core-ore | ~1 h l'una: ~32 core-ore |
| Confronti | sul Mac dai rapporti e dalle predizioni fuori fold copiati (`--oof-map`, SHA-256 verificato; secondi) | — | — | — |
| **Totale** | 1 + 4 + 13 job | | **5,0 GPU-ora (= 40 ore locali di GPU) + 416 core-ore (= 416 ore locali CPU; 52 ore di orologio al massimo)** | **~2,3 GPU-ora (~18 ore locali) + ~70 core-ore (= ~70 ore locali; ~9 ore di orologio)** |
| Opzione ancore-only s0/s1 | 4 + 8 job | idem | 5,0 GPU-ora + 256 core-ore | ~2,3 GPU-ora + ~40 core-ore |

Disco: ~3,1 GB di cache per checkpoint su `$SCRATCH/wearusfm_runs/readout_canale_20261023/features/` (float16: P1 ~2,5 GB, P2 ~0,6 GB),
rapporti in `$WORK/wearusfm_runs/results/readout_canale_20261023/`. Calendario: estrazioni in poche ore di coda; sonde a 2 alla volta sulla
seriale dietro alle baseline classiche (in corso), ~1 giorno. **Voce di budget GPU (proposta):** la valutazione dei bracci (tetto 58 GPU-ora
approvato da Simone il 22/10/2026, decisioni.md). Consuntivo non ancora registrato; stima dal `sacct` letto il 23/10 (durate dei job di
estrazione e sonda dei quattro run sani, 1 GPU ciascuno, piu' guardie e confronto): ~12 GPU-ora usate, **residuo ~46**, da cui queste 5,0 di
tetto; in alternativa l'infrastruttura (12,78 residue su 25). Ore CPU dell'account IscrB_WearUsFM, nessun budget GPU.

## Sequenza di lancio (dopo review del `critico`, decisioni.md e via di Simone)

1. Commit del codice e del protocollo; `git push leonardo deploy/readout-canale-20261023` (nessun checkout attivo lo usa; `main` non si tocca).
2. Checkout di campagna da CPU, poi le 4 estrazioni GPU, poi le 9 sonde (comandi in `results/infra/readout_canale_20261023/comandi.txt`,
   tutti con `scripts/submit_job.py`: prenotazione, ledger, nessun rilancio alla cieca).
3. Rapporti e predizioni fuori fold (storici di A, della baseline e nuovi) copiati sul Mac in `~/wearusfm_local/readout_canale_20261023/`
   (fuori da Git: sono file di dati) e verificati per SHA-256; `compare` sul Mac con `--oof-map <prefisso del cluster>=<cartella locale>`
   (il percorso cambia, lo SHA-256 del file resta quello del rapporto); esiti JSON in `results/readout_canale_20261023/` e voce nel registro.

## Limiti dichiarati

- k = 768 e PCA sul train sono una scelta di comparabilita', non la lettura migliore possibile dell'encoder; la larghezza intera sulla P1
  arriva nella tranche 2 se autorizzata.
- La media nel tempo resta: la dinamica dentro la finestra non si legge (ne' qui ne' nel readout medio).
- Due semi di A: incertezza condizionata ai due modelli, come nei bracci; la baseline classica e' una sola e fa da riferimento a tutti e
  due i semi.
- Quattro letture nel confronto 1 e la scelta del candidato sulla P1 nel confronto 2, senza correzione: dichiarate sopra.
- Nessun numero di test: tutto sulla validazione interna; le sonde ufficiali P1/P2 restano quelle di D12.

## Emendamenti dopo il congelamento (63ac0dd), dalla seconda review del `critico`, scritti dopo l'apertura della baseline classica

1. **k su UCI-EMG:** 63ac0dd rifiutava k > righe di train e le quattro sonde P2 a k 768 si sarebbero fermate su UCI-EMG dopo EPN-612; ora k
   effettivo = min(k, righe - 1), dichiarato (`k_capped`). Su UCI il readout per canale e' di fatto a larghezza intera. La regola di lettura
   non cambia; cambia cio' che «pari larghezza» significa su quel dataset, e si dice.
2. **Accoppiamento fra i due semi:** `compare` ora rifiuta candidati dei due semi di tipo diverso (strato, k, larghezza, protocollo, ricetta)
   e riferimenti uno con modello e uno senza; prima la normalizzazione per `s_comparison` poteva nasconderlo.
3. **Controllo d'uguaglianza:** una sola soglia (1e-3 su S) e un comando versionato (`--stage equality`).
4. **`dec`** descritto correttamente (target dell'opzione b del teacher esteso a tutte le posizioni, non cio' che leggono le teste delle ancore).
5. **Ore locali:** le core-ore della coda seriale SONO le ore locali del budget CPU (piano operativo §5); le «52 ore» erano ore di orologio.
6. Il punto (4) della voce della baseline classica e' sostituito dal confronto 2 (sopra).
7. **Opzione ancore-only** aggiunta come proposta, non come parte della tranche 1.
8. **(24/10/2026) Confronto dell'opzione ancore-only: `--stage compare --models`.** Il controllo d'identita' dei modelli della fase `compare`
   (scritto per i confronti 1 e 2: stesso checkpoint per seme, o riferimento senza modello) rifiutava il confronto fra due modelli diversi a
   readout uguale; l'opzione `--models` lo sostituisce con i controlli propri di quel confronto (quattro rapporti con modello, due checkpoint
   diversi per lato e fra i lati a pari seme, stesso strato/k/larghezza e stesso insieme di feature nei due lati). Regola, bootstrap
   (`arms_compare.s_comparison`, seme 20261009) e lettura descrittiva invariati; aggiunto dopo l'apertura dei rapporti local di ancore-only
   (s0 il 23/10, s1 il 24/10) e prima di eseguire il confronto: si dichiara.
Nessuno di questi emendamenti cambia soglie, regola o candidati del confronto; i rapporti della baseline (P1 e P2) sono stati aperti dopo
63ac0dd e prima di questi emendamenti: si dichiara.
