# Step 1 — chiusura P1/P2, 12 ottobre 2026

> Archivio del 21/10/2026: nota GPT del 12/10 recuperata fra i file non tracciati.
> La baseline casuale citata in fondo e' stata successivamente completata (tabella,
> punto 2). Il file `results/passo6/step1_downstream_20261012.json` citato nel testo
> non risulta nella storia Git controllata: il riferimento resta da recuperare o
> ricondurre ai report sorgente. Questa nota conserva il testo storico e non
> costituisce una nuova verifica dei risultati o della provenienza dichiarata.

Tutti i 16 job del protocollo `step1_downstream_4000_v1` sono **COMPLETED, exit 0:0**. Nessun nuovo job,
retry o training e' stato eseguito per questa chiusura. I report sono stati recuperati tramite leonardo-ops,
verificati contro gli SHA remoti e ricalcolati localmente con lo stesso estimatore.

| Metrica | A, keep 0 | B, keep 0,05 | B − A, punti percentuali | IC95 del delta, pp |
|---|---:|---:|---:|---:|
| P1 | 17,8774% | 18,2599% | +0,3825 | [−0,0844; +1,0025] |
| P2 | 79,7829% | 80,6060% | +0,8231 | [−2,3205; +3,9667] |

Le due guardie pratiche prespecificate (delta medio >= −2 pp) passano. **Non** e' una dimostrazione statistica
di non-inferiorita' o una decisione di adottare B. Gli intervalli aggregati attraversano zero. P2 contiene
split sovrapposti: SE RMS, non diviso per sqrt(3). Incertezza condizionata ai due modelli per condizione;
i seed di training non sono resamplati come se fossero soggetti.

Per dataset, delta: DB2 +0,4593 pp; DB3 −0,2184 pp; DB6 +0,9767 pp; EPN-612 +1,2051 pp;
UCI-EMG +0,4411 pp. L'intervallo positivo di EPN-612 e' un risultato di componente, non sostituisce
gli aggregati congelati. DB3 resta vicino al caso. Nessun tuning o esclusione post-hoc.

## Provenienza e costi

- Codice training: `69fa2da8757b322ea74875b4e9d104a89bbf6d5d`; teacher finali a 4.000 update, emg2qwerty, seed 0/1.
- Codice valutazione: `8dad7e82713ce4ed2741acc9c9d4fd9ac656d2f7`.
- Split SHA256: `a18e7565157d5dbba8870ffa67c2342ff90de479da480dccb456922b6f6f4ed7`.
- Comparison originale: `7c93c1b700b7462ce46eeb27a3de65abd38d95caef38b6dc03c76520957527c1`.
- Predecessore spettrale: `e2ca2d71118755f9389cacf5747fd845b377659d78e1e54004d9946e9c280e44`.
- Estrazioni `59771819–59771826`: **3,7194 GPUh**, 29,7556 ore locali GPU, sotto cap 6 GPUh.
- Sonde `59771828, 59771842, 59771846–59771849, 59771851, 59771853`: **48,54 core-ora CPU**, sotto cap 128.
  CPU allocata includendo le estrazioni: 78,30 core-ora; non aggiungere le ore CPU delle estrazioni
  una seconda volta alla conversione contabile GPU (8 ore locali/GPUh).

Hash dei checkpoint e degli otto report nel risultato versionato `results/passo6/step1_downstream_20261012.json`.
Il JSON e' riformattato: l'hash sopra identifica il file originale sul cluster, non i byte della copia riformattata.
Ricalcolo locale: stessi otto hash report; numeri concordi entro 1e-12 (unica differenza di arrotondamento:
SE P1 ~4,34e-19). Hash input effettivi, etichette/soggetti e larghezza 768 concordi fra A/B.

Warning di convergenza delle sonde: P1 1/1/2/2, P2 9/9/8/9 per modello, piu' warning di classi P1.
Gia' previsti dal protocollo, non cancellati con retry o modifiche al solver. Nessun traceback/OOM/nonfinite.

## Cosa segue

Manca lo **stesso encoder casuale**, inizializzato coi medesimi seed e valutato con le stesse sonde.
Il successo spettrale di B non basta: B e' supervisionato sulle coordinate spettrali della stessa misura.
Proposta conservativa per finestra 1: mantenere A come riferimento di lavoro e B come candidato, finche'
baseline e firma scientifica non chiudono la scelta. Nessuna modifica automatica ai default.
