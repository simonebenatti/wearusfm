# Review prima del calcolo della dinamica di B

28/10/2026, GPT. Protocollo [dinamica_b_20261028.md](../dinamica_b_20261028.md), base `21c3833`.
Analisi autorizzata da Simone; nessun nuovo training, estrazione GPU o selezione di modello.

## Critico scientifico

Review indipendente di `/root/channel_pooling_review`, con ruolo di critico prima del lancio: **FAVOREVOLE**.

Confermate la scomposizione MSE e l'identita' additiva del guadagno medio/centrato. Recepiti prima del calcolo:
unita' standardizzate; supporto comune n>=2; pesi per elementi; bootstrap appaiato per utente; nessuna media impropria
dei rapporti; validita' distinta di correlazione e rapporto MSE quando la varianza predittiva e' nulla.

Il protocollo chiarisce che centrare entrambe le traiettorie equivale a correggere perfettamente il bias per finestra/giunto.
Un vantaggio centrato misura le fluttuazioni dopo tale correzione, non dimostra che il modello operativo batta la media
vera del bersaglio. MSE totale, errore delle medie e MSE centrata restano affiancati; nessun nuovo criterio di promozione.

## Review tecnica

`/root/leonardo_ops_b` ha controllato source, builder e provenienze: flusso nel compute node, nessun edit del checkout,
input hashati, source temporaneo verificato, output esatto con scrittura esclusiva, selezione dei campi NPZ senza latenti.
Le impronte dei riferimenti nel builder coincidono con i file raccolti nella tranche E1/E2. Risorse 4 core / 16GB / 10min
coerenti con gli array necessari e il calcolo a blocchi. Il root ha riletto le formule, i pesi, i supporti e il bootstrap.

Il completamento dei test sintetici, il controllo del payload generato e degli argv finali viene registrato nel piano prima
della sottomissione. Il checkout `24139ab` e' il riferimento dell'ambiente; lo script di analisi nuovo ha SHA256 distinto,
incluso nel payload e nel risultato. La provenienza non attribuisce il nuovo codice al checkout storico.


## Chiusura prima della sottomissione

**18 test sintetici passati**, rieseguiti dal root in 0,19s. Sintassi Bash valida; il payload ricostruito dal builder coincide
byte per byte con quello versionato. Gli argv del piano coincidono con il file dei comandi; limiti CPU/memoria/tempo e
assenza di GPU verificati. Review tecnica finale favorevole: nessun difetto bloccante.

Source SHA256 `5e5ff2c8b27d5692a064534f793924ddf5463c7bab13ccf75356c0950d9dd5b8`;
payload SHA256 `288d76a289435550983055e3e35a5084e57c7ca0053c5bf12efb569c36a7ac2f`, 24.712 byte.
Piano `results/infra/pilot_posa/piano_b_dynamics_20261028.json`. Pubblicazione e verifica del commit remoto precedono il lancio.
