# Finestra 1: protocollo dei bracci dell'obiettivo (punto 5) — BOZZA da firmare

**Stato: bozza di Claude del 16/10/2026, da firmare da Simone prima del lancio; verifica di GPT dopo, non bloccante.** Simone ha scelto
«Protocollo dei bracci (punto 5)» mentre gira il collaudo `w1_selezione_v1`. Dal 16/10 protocolli, bracci, loss e harness sono di Claude
(CLAUDE.md, «Ruoli dal 16/10/2026»). Fonti: accordo del 15/10 (`docs/accordo_operativo_finestra1_20261015.md`, «Obiettivo» e prompt GPT
punti 3-4), bozza della lista dei job di GPT del 12/10 (`docs/finestra1_job_plan.md` nel worktree di GPT, non committata), v10 §10,
D11/D12/D14, confronti 2A/2C. Le parti marcate *proposta* sono scelte da firmare; nulla e' lanciato da questo documento.

## Cosa confrontare (dall'accordo, in ordine di priorita')

| Braccio | Definizione | Stato del codice |
|---|---|---|
| **A** (riferimento) | JEPA target b + ancore fisiche attuali e multi-scala (peso 0,2), keep e rami spenti, RVQ spento | pronto |
| **JEPA-only** | A con peso delle ancore 0 | il modello lo supporta (`JEPAConfig.anchor_weight`); manca l'opzione `--anchor-weight` in `train_jepa.py` |
| **Ancore-only** | A senza il termine JEPA (solo le ancore attuali) | **da implementare**: oggi `total = jepa + anchor_weight x ancore`, manca un peso della JEPA |
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
