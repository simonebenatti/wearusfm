# Manifest annidati per soggetto (D16) e passi per epoca dal corpus (Claude, 15/10/2026)

Punto 6 della numerazione unica; prompt operativo di GPT del 15/10 (punto 4: «manifest annidati e provenienza riproducibile ... con verifica
degli ingressi e nessun uso dei test per scegliere configurazioni»). Simone, 15/10: «se non rompe niente tra voi 2 puoi iniziare coi
manifest annidati».

## Cosa esisteva gia'

- **I sottoinsiemi** sono in `splits/v1/splits_v1.json` (firmato il 03/10, decisione 6 di `docs/fogli_firma_d9_d10.md`): per ogni dataset una
  permutazione dei soggetti di pretraining (seme del dataset + 1) e, per ogni frazione 0,125 / 0,25 / 0,5, il prefisso di `ceil(f x n)`
  soggetti (almeno 1). Annidati per costruzione e stratificati per dataset (la topologia e' del dataset); i soggetti di test ne sono fuori.
- **Il manifest** (`manifest-v1`, revisione v1.1) porta in ogni riga di pretraining le frazioni a cui appartiene (`nested`); il loader le
  filtra (`ManifestIndex.load(nested=...)`) e rinormalizza i pesi del manifest completo sulle righe rimaste.
- **Mancava** il collegamento al training: nessuna opzione per allenare su una frazione.

## Cosa ho aggiunto

- `RunConfig.nested` ('0.125' / '0.25' / '0.5' / None) e `--nested` in `scripts/train_jepa.py`; `load_index(..., nested)`. La frazione fa
  parte della ricetta: una ripresa con un'altra frazione e' rifiutata. Nei piani delle catene si passa negli `args` del run.
- `scripts/nested_manifest_report.py`: dal manifest congelato e dalle durate delle finestre, per frazione: soggetti, sessioni e ore per
  unita' e per classe; quote effettive col peso del loader accanto a quelle del manifest completo; passaggi per unita' a E epoche; passi per
  epoca dal corpus; verifiche (annidamento 12,5 ⊂ 25 ⊂ 50 ⊂ 100, nessuna riga di test o benchmark in un sottoinsieme, ogni soggetto con
  tutte le sue sessioni o con nessuna) e provenienza (SHA di manifest, split e durate). Esce con 1 se una verifica fallisce.
- Test: `tests/cpu/test_nested_manifests.py`.

## Esito sul manifest congelato (`results/infra/nested_manifests_v1_1_20261015.json`)

Ingressi: `manifest_v1_1.json.gz` (SHA-256 del file 853f35ac..., hash interno b3b4a3cb..., split a18e7565...),
`window_seconds_59425316.json` (e660e324...). **Nessuna verifica fallita.**

| Frazione | Soggetti | Sessioni | Ore (quota delle ore) | Classi A / B / C | Spostamento massimo della quota di un'unita' | Passaggi max a E = 4 | Passi per epoca, batch 32 |
|---|---|---|---|---|---|---|---|
| 12,5% | 78 | 2.519 | 113,6 (13,8%) | 20,4 / 74,1 / 5,5 | 1,84 punti | 7,82 | 3.412 |
| 25% | 146 | 4.947 | 215,0 (26,1%) | 20,4 / 74,8 / 4,8 | 0,53 punti | 8,00 | 6.403 |
| 50% | 284 | 9.493 | 419,3 (51,0%) | 19,8 / 75,3 / 4,9 | 0,71 punti | 8,03 | 12.478 |
| 100% | 565 | 19.034 | 822,9 | 20,0 / 75,0 / 5,0 | — | 8,00 | **24.528** |

- **Passi per epoca dal corpus: 24.528 col batch 32** (finestra media 3,77 s), non i ~23.150 delle stime fatte con finestre da 4 s: **D = 4
  epoche = 98.112 passi (+6%)**. I costi della finestra 1 salgono di conseguenza (~6% sui run a D); i piani delle catene vanno fatti con questi
  numeri. Le frazioni reali sono 13,8 / 26,1 / 51,0% delle ore (il `ceil` per dataset arrotonda per eccesso nei dataset piccoli).
- **Quote nei sottoinsiemi con la regola attuale** (pesi del manifest completo rinormalizzati): le classi restano entro mezzo punto da
  20 / 75 / 5, l'unita' che si sposta di piu' cambia di 1,84 punti (al 12,5%), e i passaggi per unita' restano quelli del manifest completo
  (~8 a E = 4, il tetto firmato): ogni unita' e' esposta come nel 100%, per ora di dati.

## Da decidere (non mio: perimetro scientifico di GPT, firma di Simone)

La regola dei pesi nei sottoinsiemi. Alternative: (a) **quella attuale** (esposizione per unita' uguale al 100%, quote di classe entro mezzo
punto); (b) quote di unita' identiche al 100% (cambiano i passaggi per unita' quando un dataset piccolo conserva piu' ore della frazione);
(c) le regole di D9 rifatte sulle ore del sottoinsieme (pesi ∝ ore^0,5 nella classe, tetto di 8 passaggi). Finche' non si decide, il trainer
usa la (a) ed e' scritto nella configurazione (`RunConfig.nested`).

## Dopo la review di GPT (15/10/2026): manifest annidati come file, pesi rifatti con D9

GPT: i pesi ereditati e rinormalizzati non conservano quote ed esposizione (classe B 74,13% al 12,5%, passaggi fino a 8,031), e nel mio
primo conto del 25% i passaggi erano quelli di `allocate` prima della normalizzazione del sampler (8,00203 dopo). La tabella e la regola
«attuale» qui sopra sono superate. Concordato: un file per frazione, stessi soggetti e sessioni, pesi D9 ricalcolati, residuo esplicito,
controlli sulla distribuzione campionata, provenienza registrata.

**Codice.** `pretraining_manifest.assign_weights_nested` (D9 sulle ore del sottoinsieme; la quota che una classe non puo' assegnare perche'
tutte le sue unita' sono al tetto va alle unita' sotto il tetto delle altre classi, in proporzione alla loro quota, senza superare il tetto;
nessuna unita' al tetto cresce; senza residuo identico a `assign_weights`), `sampled_distribution` (la distribuzione del loader),
`nested_manifest` (file annidato: sessioni di pretraining del sottoinsieme, test e benchmark invariati, parametri con frazione, regola,
residuo e origine). `scripts/build_nested_manifests.py` (autoverifiche prima di scrivere: pesi del 100% riprodotti bit per bit, sottoinsiemi
annidati, passaggi entro il tetto; file deterministici; nessuna sovrascrittura di un file diverso). Loader: con `--nested` il manifest deve
dichiarare la stessa frazione, senza non deve essere annidato (il filtro sui pesi ereditati non esiste piu'). `scripts/nested_manifest_report.py`
ora legge i file veri. Job per il cluster: `scripts/slurm/build_nested_manifests.sbatch` (CPU, minuti). Test: `tests/cpu/test_nested_manifests.py`.

**Esito della costruzione locale** dal manifest v1.1 copiato (`results/infra/nested_manifests_v1_1_build_local_20261015.json`,
`..._report_local_20261015.json`): autoverifiche superate, nessun problema nel rapporto.

| Frazione | Soggetti | Sessioni | Ore | Classi A / B / C (campionate) | Residuo | Passaggi max | Passi per epoca, batch 32 |
|---|---|---|---|---|---|---|---|
| 12,5% | 78 | 2.519 | 113,6 | 20,00 / 75,00 / 5,00 | — | 8,000000 | 3.406 |
| 25% | 146 | 4.947 | 215,0 | 20,00 / 75,02 / 4,97 | C 0,0254% a 7 unita' di A e B | 8,000000 | 6.407 |
| 50% | 284 | 9.493 | 419,3 | 20,00 / 75,00 / 5,00 | — | 8,000000 | 12.482 |
| 100% | 565 | 19.034 | 822,9 | 20,00 / 75,00 / 5,00 | — | 8,000000 | 24.528 |

**Trovato in passando:** il manifest v1.1 (correzione di CapgMyo del 07/10) ha conservato il campo `hash` di v1 (b3b4a3cb...) invece di
ricalcolarlo; quello del suo contenuto e' 52549ef8.... Innocuo per training e firme (usano lo SHA del file), ma il campo e' stantio; non tocco
il manifest congelato, i file annidati registrano entrambi. **Prossimi passi:** review di GPT del codice; poi costruzione sul cluster da un
checkout di campagna (job CPU da autorizzare) e confronto degli hash di contenuto con quelli locali (i byte compressi possono cambiare con la
versione di zlib, il contenuto no).

## Allineamento all'accordo del 15/10/2026 (`docs/accordo_operativo_finestra1_20261015.md`, main e02ac74)

La sezione precedente (residuo ad A e B, ore nominali, conferma bit per bit del 100%) e' superata. **Regola** (`pretraining_manifest.DERIVED_RULE`):
base temporale sulle ore campionabili secondo window_s congelato (`window_seconds_59425316.json`, calcolato con `signed_config`, la stessa
configurazione delle finestre del training della finestra 1: verificato dal costruttore); righe senza finestre (null) a peso 0, ore nominali,
campionabili ed escluse distinte; D9 su quelle ore; se C non raggiunge il 5% perche' tutte le sue unita' sono al tetto, A resta al 20% e il
solo residuo di C va a B (stessi pesi, stesso tetto); A o B senza capacita': errore esplicito. Verifiche dopo le trasformazioni del sampler
(normalizzazione, probabilita' peso / durata, quote nel tempo; coerenti col codice del loader). Endpoint a E epoche con un solo floor sul
totale; dose effettiva riportata. Derivato anche al 100% (`--nested 1`); v1.1 intatto.

**Rapporti finali dal commit pulito `56cea75`** (`results/infra/derived_manifests_v1_1_build_20261015.json`, `..._report_20261015.json`;
ingressi: v1.1 SHA del file 853f35ac..., window_s e660e324...). Tutte le verifiche superate.

| Manifest | Soggetti | Sessioni | Ore campionabili | Classi A / B / C dopo il sampler | Passaggi max | Passi a 4 epoche, batch 32 (floor) | Hash del contenuto |
|---|---|---|---|---|---|---|---|
| v1.1 (genitore) | 565 | 19.034 | 822,39 | 20,0080 / 74,9900 / 5,0020 | 7,9987 | 98.094 | b3b4a3cb... dichiarato, 52549ef8... ricalcolato |
| derivato 12,5% | 78 | 2.519 | 113,62 | 20,0000 / 75,0000 / 5,0000 | 8,0000 | 13.625 | 80a214fd... |
| derivato 25% | 146 | 4.947 | 214,99 | 20,0000 / 75,0254 / 4,9746 | 8,0000 | 25.628 | 13d3dce8... |
| derivato 50% | 284 | 9.493 | 419,33 | 20,0000 / 75,0000 / 5,0000 | 8,0000 | 49.926 | 52ec61f0... |
| derivato 100% | 565 | 19.034 | 822,39 | 20,0000 / 75,0000 / 5,0000 | 8,0000 | 98.094 | 167923c3... |

**Differenze del derivato al 100% dal v1.1** (spiegate dai numeri nel rapporto): le 62 sessioni di emg2pose con window_s null (0,47 h) hanno
nel v1.1 peso 0,000399; il sampler da' loro probabilita' 0 e ridistribuisce quella massa con un fattore comune, quindi il v1.1 campiona A
20,008% / B 74,990% / C 5,002%. Il derivato le porta a peso 0 e rifa D9 sulle ore campionabili: 20 / 75 / 5 esatti; gli altri pesi cambiano
al massimo di 1,3e-5. Endpoint uguale (98.094 passi a 4 epoche col batch 32: dose effettiva 3,99997 epoche). Il residuo del 25% (0,0254% da C
a B) coincide con la stima dell'accordo (A 20%, B 75,02535%, C 4,97465%).

**Prossimo:** review di GPT; poi costruzione sul cluster da un checkout di campagna (preventivo sotto) e confronto degli hash di contenuto con
questi (i byte compressi possono cambiare con zlib, il contenuto no).

**Preventivo della costruzione sul cluster (da autorizzare dopo la review).** Riferimento di deploy allo SHA del commit rivisto, checkout di
campagna su `lrd_all_serial` (1 CPU, 10 min, ~0,2 core-ora), poi `scripts/slurm/build_nested_manifests.sbatch` su `lrd_all_serial` (1 CPU,
4 GB, tetto 30 min = 0,5 core-ora; stima 2-5 min) con ingressi `$WORK/wearusfm_runs/results/passo6/capgmyo_grid_fix/manifest_v1_1.json.gz` e
`.../passo6/window_seconds_59425316.json`, uscita `$WORK/wearusfm_runs/results/passo6/derived_v1_1/` (fuori dal codice). **0 GPU-ora; tetto
complessivo ~0,7 core-ora.**
