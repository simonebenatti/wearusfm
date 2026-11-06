# Seconda review GPT: aggregati raccolti e rilascio del confronto scaling

**APPROVATO, nessun blocco.** Claude puo' procedere con il riferimento di deploy
del solo checkout metadata, il relativo job e poi il confronto, nell'ordine
pubblicato e dopo i cancelli freschi previsti. Non serve un altro via di Simone.
Stop al primo rifiuto, errore o timeout; nessun retry cieco o nuovo fit.

Consegna rivista: `cf6983ca3a52e7037a440333a1776bdfe61021d4`, ramo
`claude/scaling-300m-exec-20261106`. Inclusi invii `3f08677`, raccolta/manifest/piano
`2b7b4551c6059d5456e6e12d171ef687eb0818f4`, comandi/dry-run `b12a86e`, prova
pre-deploy `cf6983c`; controller e selftest invariati da `4b567d5`.
Ancore: Review 10 `d3976c3`, sezioni 3 e 5; approvazione del cancello `098d2b0`.

## Aggregati accettati

- Sei ricevute corrispondono agli argv gia' rivisti e al dispatcher `c2bd7979...`,
  con submitter/ledger comuni. Job: **63666045, 63666072, 63666075, 63666076,
  63666078, 63666079**. Nessun duplicato o rilancio nel materiale consegnato.
- Raw `sacct -P`: sei COMPLETED 0:0, 2 CPU, 30G, limite 600 s;
  durate 41/22/34/24/28/23 s. Costo misurato **0,095555556 core-ore**.
- Le 18 copie raccolte di report/audit/log hanno hash e dimensioni coerenti con
  lo snapshot pubblicato `9ca3bc9d...`. I sei report sono completi, stessa dose,
  piano, codice e firma numerica `1ac8481e...`.
- Quattordici pin degli shard, 60 combinazioni cella/dataset/fold e 420 record di
  fit coerenti; nessun fit irrisolto. Impronte OOF, finestre e utenti coincidono
  con le ancore di Review 10 nelle tre celle, compreso fit 14.
- Rieseguiti offline i **26 controlli** del controller sul materiale reale raccolto:
  tutti passati. Audit indipendente aggiuntivo: **189/189 controlli passati**.
  Gli OOF remoti non sono stati riscaricati da GPT: i loro hash sono attestati
  dalla raccolta pubblicata e saranno rivalidati dal comparatore congelato.

## Manifest e comandi

Manifest `comparison_inputs.json`: **3.062 byte**, SHA256
`162da7fb558035679b2d5e7112c81f5039782d50ebae68d5164eb003e55171ea`,
hash del contenuto `bfa36e7f9af32dace3c976130d352615c96abd91ae64f7c8f19b8ea88ffcd14f`.
Sei pin distinti degli aggregati; piano sonde `d2cc7da1...`, posa `a500cc3c...`,
ridge `8f70ff3b...`; chiavi path/sha256/bytes, percorsi canonici attestati e sei
hash delle sorgenti uguali a `aef665b` e alla Review 10.
Rieseguiti `validate_inputs` e `make_inputs` del comparatore congelato: PASS.

Piano SHA256 `1b445cfb9701bfb9fe907800a1192ff5c0e1b124c59bd25f46a20b144ea81765`.
Due righe d'invio SHA256 `b146f42f3588ae56793f390b39dd902bfb8b884e35c4ee6593acba8b6a1e66d7`.
Manifest e piano sono presenti con gli stessi byte nel commit del checkout.

| Job | Risorse e limite | Identita' |
|---|---|---|
| `SCALING-inputs-checkout-20261106` | 2 CPU, 4G, 5 min | Metadata `2b7b4551c6059d5456e6e12d171ef687eb0818f4` |
| `SCALING-compare-20261106` | 8 CPU, 30G, 30 min | Numerica frozen `aef665ba7ea66fc80291846487907bb3ef9107df` |

Verificati argv esatti, `/dev/stdin`, hash dei due launcher, nessuna dipendenza,
output nuovo e distinzione fra commit metadata e runtime scientifico. Il checkout
pubblica il manifest in `scaling_300m_comparison_inputs_20261106_v1`; non sostituisce
il checkout da cui viene importato il codice numerico.

## Controller, prove e ordine operativo

Controller SHA256 `0e3edb29eb845b89e8ccd72cc3ba0f477ef1d8d2f5d466d12ee16a0499806df7`.
Rieseguito il selftest: **2/2 passati in 11,60 s**, catena sintetica con cartelle
temporanee, scheduler/SSH/ledger finti e confronto congelato. Una review
indipendente ha aggiunto **35/35 prove locali sui gate**, con snapshot pubblicato
e stati sintetici, letture remote e scritture sostituite da stub.

Ordine approvato, eseguito esclusivamente da Claude:

1. `pre-deploy` fresco: pin, ledger, coda riuscita/posto libero, output e checkout
   assenti, origine pubblicata, riferimento deploy nuovo e non estratto in un
   worktree, nessun hook attivo nel contesto verificato.
2. Pubblicazione di **2b7b4551c6059d5456e6e12d171ef687eb0818f4** sul solo riferimento
   `deploy/scaling_300m_comparison_inputs_20261106_v1`; poi `gate-checkout` fresco
   e invio del checkout tramite la prima riga pubblicata.
3. Attesa di COMPLETED 0:0 e `gate-compare`: checkout con HEAD esatto e albero
   pulito, manifest nel percorso/hash/dimensione attesi, input ancora pinnati,
   output assente, ledger nuovo e posto libero. Guardia reale del dispatcher:
   caso valido e tre casi negativi secondo il piano.
4. Invio della seconda riga, confronto. La regola «Due fit per volta» della linea 2
   cade solo quando **SCALING-compare-20261106** risulta sottomesso nel ledger;
   il checkout metadata non la fa cadere.

Le prove locali non attestano l'esecuzione reale di `gate-checkout`, `gate-compare`
o il controllo del runtime a otto thread dentro il job. Restano requisiti della
sequenza, non nuovi consensi. Lo snapshot pre-deploy delle 13:22:49 e' storico e
non sostituisce la lettura fresca. Nessun afterok sostituisce l'accettazione.

## Budget e perimetro

Consumato: **125,089444445 core-ore**. Checkout massimo con margine 0,266666667;
confronto massimo con margine 4,4. **Caso peggiore 129,756111111 su 310**,
ricalcolato dal raw sacct e dalla base accettata. Nessuna nuova GPU o modifica di
dose, loss, ricetta, semi o S8; continuazione a 19.618 non autorizzata.

Prove locali in `results/infra/scaling_300m_20261104/comparison_release_review_20261106/`.
Nessuno score interpretato, nessuna conclusione scientifica anticipata. Il risultato
del confronto e la sua analisi saranno oggetto della prossima review. GPT non ha
eseguito SSH, deploy, checkout, invii o modifiche al codice di Claude.
