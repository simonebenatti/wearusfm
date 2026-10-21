# Finestra 1: protocollo dei bracci dell'obiettivo (punto 5) — BOZZA da firmare

**Stato: FIRMATO da Simone il 16/10/2026** («firmo i bracci come proposti»): i cinque punti di «Da firmare» con le proposte di questo documento e i tre punti di protocollo del `critico` in coda, come riportati in `docs/decisioni.md`. Verifica di GPT dopo, non bloccante. Simone ha scelto
«Protocollo dei bracci (punto 5)» mentre gira il collaudo `w1_selezione_v1`. Dal 16/10 protocolli, bracci, loss e harness sono di Claude
(CLAUDE.md, «Ruoli dal 16/10/2026»). Fonti: accordo del 15/10 (`docs/accordo_operativo_finestra1_20261015.md`, «Obiettivo» e prompt GPT
punti 3-4), bozza della lista dei job di GPT del 12/10 (`docs/finestra1_job_plan.md` nel worktree di GPT, non committata), v10 §10,
D11/D12/D14, confronti 2A/2C. Le parti marcate *proposta* sono scelte da firmare; nulla e' lanciato da questo documento.

## Cosa confrontare (dall'accordo, in ordine di priorita')

| Braccio | Definizione | Stato del codice |
|---|---|---|
| **A** (riferimento) | JEPA target b + ancore fisiche attuali e multi-scala (peso 0,2), keep e rami spenti, RVQ spento | pronto |
| **JEPA-only** | A con peso delle ancore 0 | pronto dal 16/10 (`--anchor-weight 0`, b1238a1) |
| **Ancore-only** | A senza il termine JEPA (solo le ancore attuali) | pronto dal 16/10 (`--jepa-weight 0`, `JEPAConfig.jepa_weight`, b1238a1) |
| **Keep** (confronto separato) | A con la keep dello Step 1 a 0,05 | pronto (`--keep-calibration`, `--keep-weight`), ma vedi «Keep» sotto |

- *Ancore-only* sono le **ancore attuali** (log-RMS, forma spettrale, inviluppo, multi-scala), non il braccio piu' ampio di v10 dei «soli
  target espliciti» spaziali e multiscala: quello resta fuori da questo lotto, come il waveform MSE (accordo; bozza GPT).
- Fuori dal default e da questi confronti: PSD/wavelet (2A/2C negativi), RVQ (run5 chiusa).
- Il rapporto delle norme dei gradienti JEPA/ancore (~9) **non** cambia i pesi (accordo): i bracci si confrontano con i pesi del default.

## Disegno (proposta)

- **Taglia e durata:** 30M (preset `sanity`), D = 4 epoche del manifest derivato al 100% = 98.094 passi a batch 32, WSD D14 (warmup
  1.962, decay sul 20% finale), checkpoint post-decay valutato. LR: quello di 30M uscito da C1-C2 (**da decidere quale**, se C2 conferma due
  LR: *proposta* il finalista con S migliore al seme 1, parita' al piu' basso, come la regola firmata).
- **Semi:** 2 per braccio (0, 1), quelli gia' previsti per le ablation (accordo: i pilot brevi non li sostituiscono). Stesso seme = stessa
  inizializzazione e stesso flusso dei dati fra i bracci (stessa architettura: le teste delle ancore e il predittore esistono in tutti i
  bracci, cambiano solo i pesi della loss), quindi confronti accoppiati per seme.
- **Esposizione:** uguale per tutti (stessi passi, stesso manifest e sampler); nessun braccio piu' lungo per compensare.
- **Valutazione:** P1 e P2 ufficiali (D12) del teacher post-decay, encoder congelato, coi soggetti di test: e' l'esito scientifico della
  campagna, non una scelta di iperparametri. Piu' le sonde di non-collasso gia' in uso (decisione 16) e S della validation interna come
  descrittivo.
- **Regola di lettura (*proposta*, da congelare prima dei risultati):** un braccio «batte» A se migliora P1 **e** P2 di almeno +1 punto
  con la media sui due semi e l'errore standard combinato (`harness.probe.combine_seeds`), come la soglia dei confronti 2A/2C; altrimenti A
  resta il default. Se JEPA-only o ancore-only eguagliano A entro la soglia, si registra che quel termine non aggiunge valore misurabile a
  30M, senza cambiare il default da solo.

## Keep: due opzioni da scegliere

La keep (B) richiede la testa di readout dello Step 1, che consuma numeri casuali all'inizializzazione: per un confronto accoppiato anche il
riferimento deve avere la testa a peso 0 (`with_spectral_keep`: «Both A (weight=0) and B instantiate the SAME head»).
1. **Confronto separato** (accordo): A' (testa a 0) e B (0,05), 2 semi: 4 run in piu'. Gli altri bracci restano senza testa.
2. **Testa a 0 in tutti i bracci:** un solo riferimento per tutto, B aggiunge 2 run; ma tutti i bracci dipendono dalla calibrazione della keep.
**Da verificare in entrambi i casi:** la calibrazione della keep oggi e' quella dello Step 1 su emg2qwerty (`4640dfcf...`); per il corpus
intero va ricalcolata o motivata prima del lancio.

## Implementazione necessaria (Claude)

1. `--anchor-weight` in `train_jepa.py` (oggi 0,2 fisso dal preset) e controlli: >= 0, finito.
2. Peso della JEPA in `JEPAConfig` (default 1, comportamento attuale invariato con test d'uguaglianza) e `total = jepa_weight x jepa +
   anchor_weight x ancore`, anche nelle riduzioni distribuite (`jepa_loss_terms`, `loss_counts`): con peso 0 il predittore non riceve
   gradiente, quindi in DDP serve `find_unused_parameters` o l'esclusione del predittore dall'aggiornamento; il teacher EMA resta definito.
3. Opzione `--jepa-weight` nel trainer; la ripresa rifiuta un checkpoint con pesi diversi (gia' cosi' per la configurazione).
4. Pianificatore dei bracci (stessi controlli di `wsd_chain.py`) e piano di valutazione P1/P2 per la catena; test CPU dei quattro bracci
   (pesi effettivi nella loss, gradienti nulli dove devono, stesso stato iniziale fra bracci a pari seme).
Prima del lancio: review del subagent `critico` e un collaudo breve su dbg dei due bracci nuovi (pochi minuti di GPU).

## Costi (stima dalle misure: 30M 0,70 s/passo a 4 GPU col prefetch)

| Voce | Run | GPU-ora attese / tetto |
|---|---|---|
| Un run di 4 epoche (98.094 passi, ~19 h; un job da ~23 h) | 1 | ~77 / ~92 |
| A, JEPA-only, ancore-only, 2 semi | 6 | ~464 / ~552 |
| Keep, opzione 1 (A' e B, 2 semi) | 4 | ~309 / ~368 |
| Keep, opzione 2 (solo B, 2 semi) | 2 | ~155 / ~184 |
| Valutazione P1/P2 (~1,5 GPU-ora per checkpoint, come 2C) | 8-10 checkpoint | ~12-15 |
| **Totale** | 8-10 run | **~630-790 attese, tetto ~750-935** |

Calendario: con 8 nodi i 6 run dell'obiettivo entrano in un'ondata (~23 h), la keep in una seconda; valutazione ~2-3 h dopo ciascuna.
Questi run sono una parte della famiglia «Ablation a 30M» (~2.100 GPU-ora in `docs/finestra1_fattibilita.md`); gli altri bracci della bozza
GPT (target a/b, identita', K, locale, bottleneck, vicinato) restano da prioritizzare con Simone e il budget della finestra 1.

## Da firmare (Simone)

1. Bracci e definizione di ancore-only (le ancore attuali, non il braccio v10 piu' ampio).
2. 30M, 4 epoche, 2 semi; LR di 30M da C1-C2 (quale, se C2 ne conferma due).
3. Regola di lettura: +1 punto su P1 e P2, media dei due semi.
4. Keep: opzione 1 o 2, e la calibrazione della keep per il corpus intero.
5. Via all'implementazione dei punti 1-4 sopra (lavoro locale, nessun job).

## Stato dell'implementazione (16/10/2026, Simone: «poi lavoriamo su» JEPA puro e ancore-only)

Fatti i punti 1-3 di «Implementazione necessaria»:
- `JEPAConfig.jepa_weight` (default 1) in `training/jepa.py`: totale = `jepa_weight` x JEPA + `anchor_weight` x ancore, anche nel contratto
  delle loss distribuite (`terms.weight["jepa"]`); pesi finiti e non negativi, mai tutti a zero; JEPA a peso 0 senza target delle ancore
  rifiutata. Col peso 1 la JEPA entra com'era: **training breve su CPU identico bit per bit** al codice di prima (studente `a9440678...`,
  teacher `7d88427b...`, anche coi pesi espliciti 1 e 0,2). I checkpoint di prima si riprendono (peso 1 implicito); un peso cambiato no.
- `--anchor-weight` e `--jepa-weight` in `scripts/train_jepa.py`; ammessi negli argomenti liberi delle catene WSD.
- **Correzione al punto 2 sopra:** in DDP non serve `find_unused_parameters`. Un termine a peso 0 resta nel grafo (`backward_loss` lo
  moltiplica, non lo salta): ogni parametro dello studente riceve un gradiente, zero dove deve, in entrambi i bracci (test, e training DDP a
  2 processi gloo con accumulo per entrambi i bracci). Il decoder dello studente, in ancore-only, e' allenato dalle ancore (le teste leggono
  le sue uscite); il teacher EMA si aggiorna come sempre ed e' l'encoder che le sonde valutano, come negli altri bracci.
- Cosa leggere con cautela: nel JEPA puro le teste delle ancore restano non allenate, quindi le loss delle ancore registrate non dicono
  nulla; in ancore-only la JEPA registrata misura un predittore che non viene allenato per quello.
- Test: `tests/cpu/test_objective_arms.py` (19). Resta il punto 4 (pianificatore dei bracci e piano di valutazione), poi `critico` e collaudo
  breve su dbg.

## Review del `critico` sull'implementazione (16/10/2026): APPROVATO, 4 rilievi minori

Su b1238a1 (criteri: default identico, totale e contratto distribuito, pesi, DDP, ripresa, pianificatori, correttezza dei bracci, test,
coerenza con decisioni e CLAUDE.md): tutti soddisfatti; ha riprodotto le impronte del default anche col `jepa.py` di b8a2f35 caricato in
memoria, e col percorso del contratto a `--accum-steps 3`. Rilievi:
1. Diagnostica dei gradienti per componente (`--keep-grad-every`, solo con keep o rami accesi) con la JEPA non pesata: **corretto**, ora
   pesata come nel totale (col peso 1 la stessa espressione di prima). Il caso con peso diverso da 1 non ha un test (richiede la
   calibrazione della keep): riletto, non provato.
2. Test d'uguaglianza fuori dal repo: **corretto**, `test_default_objective_is_bit_identical_to_the_code_before_the_jepa_weight` confronta
   col `jepa.py` di b8a2f35 letto da git perdite, termini del contratto e gradienti (torch.equal), target a e b, per finestra e per token.
3. Contratto delle loss e tabella sopra non aggiornati: **corretti** (`docs/contratto_loss_distribuite.md`, colonna «Stato del codice»).
4. **Da chiudere nel protocollo prima della firma** (non codice):
   - (a) un braccio fermato dall'allarme di collasso (decisione 16) non ha il post-decay: la regola di lettura deve dire come si conta
     (per esempio «non ammissibile = perde», o un esito a parte), soprattutto per il JEPA puro, il candidato piu' probabile al collasso;
   - (b) in ancore-only il rapporto di collasso da query misura un decoder allenato dalle ancore, non dai latenti del teacher: va scritto
     se l'allarme firmato a 0,05 vale anche li';
   - (c) i bracci usano il LR calibrato su A: un «non batte A» o un «eguaglia A» si legge «al LR di A».

## Firma (16/10/2026)

Simone: «firmo i bracci come proposti». Firmati, nella forma proposta in chat lo stesso giorno:
1. Bracci: A, JEPA-only (peso delle ancore 0), ancore-only (solo le ancore attuali, non il braccio v10 dei target espliciti), keep come
   confronto separato.
2. 30M, D = 4 epoche, 2 semi (0, 1); LR di 30M da C1-C2: se C2 conferma due LR, quello con S migliore al seme 1, a parita' il piu' basso.
3. Lettura: un braccio batte A se migliora P1 **e** P2 di almeno +1 punto con la media dei due semi (errore standard combinato).
4. Keep: opzione 1 (A' con la testa a peso 0 e B a 0,05, 2 semi); la calibrazione della keep si ricalcola sul corpus intero prima del lancio.
5. Via all'implementazione del pianificatore dei bracci e del piano di valutazione P1/P2 (lavoro locale).
6. (critico a) Un braccio fermato dall'allarme di collasso si registra come «collassato» e conta come «non batte A».
7. (critico b) L'allarme di collasso firmato (0,05, decisione 16) vale per tutti i bracci, ancore-only compreso, senza eccezioni.
8. (critico c) «Non batte A» o «eguaglia A» si legge «al LR di A»: nessuna calibrazione del LR per braccio in questa finestra.

## Decisioni del 16/10/2026 sulla critica del punto 5 (Simone: «iniziamo a risolvere 1 e 2, apporvo le soluzioni proposte, se riesci firmo e implementa eventuali corrzioni dove non gi afatte»)

Sostituiscono il punto 3 della firma («+1 punto sulla media dei due semi») e la frase «altrimenti A resta il default» del Disegno.
1. **Regola di lettura sui test ufficiali (come il 2A/2C):** un braccio «batte A» su P1/P2 se, nel confronto accoppiato per seme
   (`harness.downstream_ab.compare_reports`: riferimento A, braccio al posto di B, semi 0 e 1), per P1 **e** per P2: delta medio >= +0,01
   (+1 punto), limite inferiore dell'intervallo al 95% > 0, delta > 0 in tutti e due i semi. «Non dannoso» resta delta >= -0,01. E' la
   lettura scientifica del confronto; non sceglie il default.
2. **Il default si sceglie senza i soggetti di test:** sulla validation interna S (le sonde a fold dei soli soggetti delle partizioni train,
   `scripts/calibration_probe.py` per la P1; l'analoga per la P2 sui soggetti fuori dai test dei tre split 7:1:2), con la stessa regola del
   punto 1 applicata a S (delta accoppiato sui soggetti, per seme): un braccio sostituisce A come default solo se batte A su S della P1 **e**
   su S della P2. P1/P2 ufficiali sono la conferma e il risultato scientifico; se S e test ufficiali non concordano, il default segue S e il
   disaccordo si registra. La S della P2 la conferma Simone: «per p2 bene òa proposta tua» (16/10). Implementata in `scripts/calibration_probe.py
   --task p2`: soggetti fuori dai test dei tre split 7:1:2 (semi 0, 1, 2), finestre degli altri scartate prima dell'estrazione,
   stessa estrazione e sonda della P2, 4 fold di soggetti, media dei due dataset a pesi uguali.
Non cambiano: bracci, disegno (30M, 4 epoche, 2 semi), keep separata, allarme di collasso, letture «al LR di A».
Implementazione delle due regole: `scripts/arms_compare.py` (test `tests/cpu/test_arms_compare.py`). Un braccio collassato non batte il
riferimento; un run fermato per un altro motivo non e' valutabile (si decide a mano); se piu' bracci battono A su S, la scelta e' di Simone.

## Pianificatore e piano di valutazione (16/10/2026; critica del punto 5, punti 3-6)

`scripts/arms_plan.py` (test `tests/cpu/test_arms_plan.py`): piani per `wsd_chain.py` (gruppo «objective»: A, JEPA-only, ancore-only; gruppo
«keep»: A' e B, solo con la calibrazione della keep sul corpus intero) e piano di valutazione per `evaluation_chain.py`: per ogni run guardia
-> P1 e P2 ufficiali (`probe_p1*.sbatch`, `probe_p2*.sbatch`, ora con CODE_DIR) e S della P1 e della P2 (`calibration_*.sbatch`), poi
`arms_compare.sbatch` col confronto firmato (spec `confronto_<gruppo>.json`). Test dello stato iniziale: a pari seme i tre bracci hanno la
stessa architettura e lo stesso stato iniziale, con le configurazioni costruite da `train_jepa.py` (rilievo 5).

**Preventivo** (LR di esempio 3e-4; un job da 23 h per run, 98.094 passi a 0,70 s):

| Gruppo | Run | Training GPU-ora attese / tetto | Valutazione, sonde su CPU | Valutazione, sonde su GPU |
|---|---|---|---|---|
| Obiettivo (A, JEPA-only, ancore-only, 2 semi) | 6 | 463,8 / 552,0 | 10,5 / 27,0 GPU-ora + 192 core-ora, **~24 h di sonde in fila** | 34,5 / 87,0 GPU-ora, in parallelo |
| Keep (A', B, 2 semi) | 4 | 309,2 / 368,0 | 7,0 / 18,0 + 128 core-ora, ~16 h in fila | 23,0 / 58,0 |
| **Totale** | 10 | **773,0 / 920,0** | 17,5 / 45,0 | 57,5 / 145,0 |

Sulla seriale le sonde girano una alla volta (`QOSMaxCpuPerUserLimit`, collaudo `w1_selezione_v1`): ~4 h di sonde per run, quindi ~24 h per i
sei run dell'obiettivo dopo i training; sulle GPU costano ~24 GPU-ora in piu' e girano in parallelo (rilievo 4).
**Prima del lancio dei bracci:** LR di 30M da C1-C2 (la regola firmata), calibrazione della keep sul corpus intero (per il gruppo keep;
rilievo 6), review del `critico` su pianificatore, confronto e S della P2, collaudo breve su dbg dei due bracci nuovi, preventivo e via di Simone.
