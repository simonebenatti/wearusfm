# Risultati già leggibili: 30M lungo e 300M prima del decay

26/11/2026. Risposta alla domanda di Simone: «della 30M e del primo pezzo
della 300 possiamo vedere risultati scientifici? o serve altro?».

**Il 30M ha già risultati di valutazione a 39.237 e 49.046 passi. Il 300M a
39.237 ha diagnostica di apprendimento leggibile, ma non ancora la valutazione
downstream di quel checkpoint.** Le tre estrazioni 30M appena concluse sono
ingressi per i nuovi lettori uniformi, non nuovi score.

## 30M: risultati scientifici già pubblicati

Fonte: `docs/reviews/consolidation_long_results_20261107.md` e
`results/infra/consolidation_long_20261105/gpt_review_20261107/scientific_readout.json`,
nel worktree `wearusfm-consolidation`, ramo
`codex/pose-temponet-proposal-20261117`, commit letto
`0e56c8b1c67f8257e800f6cd51bafacac28971b4`. Conclusioni della valutazione
firmate da Simone il 7/11; non sono nuovi fit o nuove conclusioni promosse qui.

| Misura | 19.618 | 39.237 | 49.046 |
|---|---:|---:|---:|
| Posa dinamica, E_dyn ridge post ↓ | 0,838933 | 0,809722 | 0,785902 |
| Posa dinamica, E_dyn testa teacher ↓ | 0,832954 | 0,805007 | 0,778629 |
| P1, pool storico | 17,19% | 17,52% | 17,53% |
| P2, pool storico | 80,52% | 80,75% | 81,87% |

49k rispetto a 19k:
- riduzione relativa di E_dyn ridge **6,32% [5,87; 6,83]**, testa
  **6,52% [5,92; 7,15]**;
- EPN612 passa da circa 82,58% a **84,51%**, delta
  **+1,93 pp [1,42; 2,44]**;
- P1 **+0,35 pp [-0,22; 0,91]**, P2 **+1,35 pp [-0,13; 2,71]**:
  il beneficio dei primari aggregati della classificazione resta irrisolto.

La posa migliora anche nella componente temporale centrata e nel controllo
allineato/disallineato; non soltanto nel livello medio della finestra.
La qualità assoluta resta limitata: correlazione centrata 0,463/0,471 e
varianza delle predizioni pari al 22–24% dei bersagli (ridge/testa).
Posa su 15 utenti esclusi dal training; classificazione e posa restano
popolazioni di sviluppo, un solo seme di pretraining. La posa è un compito
supervisionato durante il pretraining, quindi non dimostra da sola
trasferimento generale del FM.

Il checkpoint teacher a 49k è lo stesso usato dalle nuove estrazioni:
`bfa6141fc2ed6505d2f582a46f0c04f36fd35f867e8cff2b11555b241947a490`.
La verifica locale del critico ha confrontato la ricevuta storica
`LONG-G49046-pose-val-20261105.json` con il pin del nuovo controllo.

Le valutazioni storiche non sono invalidate dalle nuove estrazioni. Il piano
scaling `docs/scaling_49046_execution_20261119.md`, righe90–103, motiva la
ripetizione con schemi, provenienze e contesto numerico dei vecchi artefatti
non direttamente certificati dai validatori del nuovo confronto.
I vecchi risultati non vanno presentati come esito dei nuovi lettori uniformi.

## 300M: lettura descrittiva del training esistente

Run `T100-G-300M-39237`, job 64837401, codice
`87da975bc0fd4824684e166659bb996e118ff107`.
Checkpoint completo a 39.237:
`c9685981cfeba9ccc4860a8aa86a66b2d9de9f1dc8a69671240e0bfcd41accc2`.

Letti in sola lettura i file già scritti su Leonardo, senza job o forward nuovi.
SHA256 completi confrontati con `run_inventory` della ricevuta di training:
- `fixed_eval.jsonl`:
  `de5614094f944a59a7f69f126ecc1f97d07bdda3f80d6c87a06c91cbc5ef6df3`;
- `pose_panel.jsonl`:
  `f2c71907fe2527132df8255a86a8a3c4dc586a595b6df40d180cb21dd5504f68`.

Le 39 letture di ogni file coprono **20.000–39.000**, ogni 500 passi.
Il checkpoint è a 39.237; l'ultima misura a 39.000 non è una
valutazione esatta a 39.237. Nessuna selezione del punto migliore.

Il pannello valuta lo **student**, in eval e senza gradiente, su 256 finestre
fisse di 32 utenti del training. Lo stesso `panel_sha256` e tutte le 256 finestre
valide sono presenti nelle 39 letture. Definizione in
`src/wearusfm/training/run.py::_pose_panel_errors` del codice congelato.

| Pannello posa, student/train | 20.000 | 39.000 | Variazione descrittiva |
|---|---:|---:|---:|
| MSE standardizzata della posa ↓ | 0,941528 | 0,840196 | -10,76% |
| MAE in radianti ↓ | 0,291992 | 0,268223 | diminuisce |
| Varianza temporale delle predizioni | 0,057270 | 0,052116 | -9,00% |

La loss totale sul piccolo insieme fisso passa da 1,799483 a 1,637803.
La riduzione dell'errore sostiene che l'ottimizzazione continua a progredire.
Non dimostra generalizzazione, superiorità sul 30M, né migliore ricostruzione
della dinamica: la MSE mescola livello e movimento, e la varianza predetta
da sola non misura accuratezza temporale. Il suo calo non prova un
peggioramento del movimento; semplicemente non sostiene l'ipotesi di
traiettorie diventate più ampie. Non sono disponibili qui E_dyn, E2 o
correlazione sui 15 utenti esclusi per questo checkpoint 300M.

**0,840 del pannello 300M non va confrontato con 0,786 della ridge 30M:**
student/train/MSE assoluta contro teacher/utenti esclusi/errore dinamico
normalizzato; metriche, supporti e lettori diversi.

Estratto di tutti i 39 punti scalari, hash e calcoli:
`results/infra/scaling_49046_20261118/science_readout_20261126/diagnostics_300m.json`.
I valori per finestra non sono copiati nell'estratto; nessun IC nuovo è
calcolato o implicito per queste differenze sul train.

## Che cosa manca e quale sequenza conviene

1. Possiamo discutere subito la valutazione storica30M e questi segnali del300M,
   senza nuovo calcolo di training.
2. Per score downstream del300M a39.237 occorrono inferenza sul checkpoint e
   lettori addestrati sulle feature congelate, poi valutazione sugli utenti
   esclusi; non occorre riaddestrare il FM. Un confronto alla stessa dose può
   usare il30M39.237, con un protocollo di lettura compatibile e verificato.
   È una valutazione intermedia aggiuntiva: non era inclusa nella catena49k.
3. La priorità raccomandata resta terminare il300M a49.046 e usare la lettura
   uniforme già prevista. Le nuove sette cache30M sono pronte; per ottenere
   gli score del nuovo protocollo mancano i fit CPU. La catena corrente
   prepara quei lettori insieme alle cache terminali300M.
4. Se il seguito300M resta fermo a lungo, si può valutare un anticipo dei fit
   CPU30M o una misura mirata del300M39k: serve un piano concreto separato,
   con costi e riuso espliciti, non l'affermazione che manchi altro pretraining.

Nessun nuovo job, modifica scientifica, preparazione di rilancio o modifica
ai checkout congelati in questa lettura. Claude resta operatore della catena 49k.

## Verifica della lettura

Il critico ha ricalcolato i contrasti descrittivi dal JSON archiviato e
confrontato i numeri del 30M con le fonti locali: **APPROVATO, nessun blocco**.
Verificate le 39 letture regolari, l'identità del pannello e la distinzione
fra metriche, popolazioni e passi. La sua verifica non comprende una seconda
lettura SSH dei payload: gli hash remoti sono stati verificati da GPT durante
la raccolta descrittiva riportata sopra.
