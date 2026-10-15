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
