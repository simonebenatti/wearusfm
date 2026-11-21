# Audit del supporto contrastivo RMS100/fast125

L’audit CPU della linea 3 ha superato il criterio firmato. Il rapporto del job `64937087` contiene i 1.024 batch attesi. Il job è terminato con `COMPLETED 0:0` in 4.022 s, con 8 CPU e un limite di 7.200 s. Con il checkout `64936161` il consumo è 8,99 core-ore e zero GPU-ore. La decompressione dell’archivio restituisce i 20.275.571 byte originali, SHA-256 `eb7cbdf49ffdd00148647e82c5f51768edbd17b5ef515d960a1f6edec35754d3`. L’accounting verificato lega questi byte alla singola allocazione. [Evidenze e contabilità](../../results/infra/contrastive_targets_20261118/audit_execution_20261120/)

## Qual è la domanda risolta

Per InfoNCE contiamo le finestre che hanno abbastanza negativi ammessi dalle esclusioni. Per Rank-N-Contrast (RnC) contiamo i batch con esempi di posa che permettono di ordinare le coppie. Queste due percentuali hanno denominatori diversi. Gli update in tabella partono da 1, il campo `step` nell’audit parte da 0. Il pannello e gli update 545–576 sono sottoinsiemi del tratto attivo.

| Porzione del flusso | Finestre InfoNCE ammissibili | Supporto RnC |
|---|---:|---|
| Pannello degli update 257–512 | 8.179 / 8.192 (99,84%) | 256 / 256 batch |
| Tratto attivo osservato, update 257–1.024 | 24.527 / 24.576 (99,80%) | 768 / 768 batch |
| Update 128 e 256, scelti per la parità | 32 / 32 a ciascun update | 57 e 64 ancore, rispettivamente 3.192 e 4.032 coppie |
| Update 545–576, scelti per la ripresa | 1.023 / 1.024 | 32 / 32 batch, almeno 47 ancore per batch |

CapgMyo spiega tutte le 13 esclusioni del pannello e 45 delle 49 nel tratto attivo. È il dataset raro già esentato dal requisito InfoNCE per singolo dataset. Le altre quattro sono di NinaPro DB3, con 233 finestre ammissibili su 237 (98,31%). La copertura globale include entrambe le popolazioni. Nel tratto della ripresa l’unica finestra esclusa per InfoNCE viene da CapgMyo all’update 560.

Il confronto record per record con l’audit storico RMS25/fast50 restituisce gli stessi dati per tutti i 1.024 batch, comprese le coppie e le esclusioni. Nel JSON cambiano solo le identità di configurazione e dei sorgenti. La nuova coorte offre quindi la stessa disponibilità di esempi su questo campionamento. L’audit usa identità delle finestre, maschere e supporto della posa. Per sapere se i nuovi bersagli cambino i gradienti o il valore della rappresentazione occorrono misure sul modello. [Audit storico](../../results/infra/contrastive_comparison_20261116/audit_completed_20261117/)

## Decisione scientifica

Procedere alla calibrazione dopo il controllo locale P-5 dell’orchestrazione. I 256 batch del pannello hanno supporto per entrambe le loss, inclusi i primi 32 (update 257–288). Per la selezione effettiva servono anche tre norme di gradiente finite e strettamente positive sullo stesso batch, relative all’obiettivo di base, a InfoNCE e a RnC. Il lettore prende i primi 32 che soddisfano tutti i criteri. La calibrazione usa la ricetta C0, il batch misto di 32 e 512 passi senza aggiornamenti dell’ottimizzatore.

Il lettore determina separatamente i due pesi dalla mediana dei rapporti fra norma di base e norma contrastiva sui 32 batch comuni, moltiplicata per il riferimento storico firmato e per 0,1. Entrambi i pesi devono rientrare in `[1e-4, 1]`. Una copertura insufficiente, meno di 32 batch utili o un peso fuori intervallo fermano la promozione. Registrare anche tempi e picco di memoria GPU del profilo bf16 con SDPA math e TF32 spento al batch reale. Solo dopo la lettura della ricevuta positiva si può preparare la fase CUDA prevista dal preflight. [Piano scientifico e piano delle risorse](../../results/infra/contrastive_targets_20261118/)

## Verifica per la consegna a 3-launcher

Il 20/11 sono passati 107 test sui controlli di release, piano, parità e ripresa. Il runner ha inoltre validato sul Mac i quattro input con le stesse impronte della campagna, il rapporto CPU e la contabilità, traducendo solo i percorsi fisici. Le verifiche operative sul login spettano a 3-launcher.

Al commit di preparazione `a9647368` sono pubblicati i due script per il controllo P-5 sul percorso reale di `contrastive_run.main`. Il sistema locale ha rifiutato l’apertura del canale di comunicazione fra processi su loopback. Al momento di questa annotazione mancano le ricevute dei quattro processi per C0 e per la ripresa. Il pacchetto di calibrazione attende questa prova. Nessuna fase GPU è stata rilasciata. [Prova locale P-5](../../results/infra/contrastive_targets_20261118/gloo_p5_20261120/)

### 21/11/2026 — P-5 chiuso, calibrazione rilasciata

Sono disponibili le ricevute del vero ingresso del trainer, con quattro processi CPU/Gloo, un modello ridotto e un batch sintetico di 32. Per C0 i sei casi a peso zero hanno prodotto un unico gradiente per punto in tutti i rank ai passi 2 e 3. Lo stato vivo è stato ripristinato dopo ciascuna misura. Verificati gli SHA dei 16 file decompressi. [Ricevuta C0](../../results/infra/contrastive_targets_20261118/gloo_p5_20261120/full_c0/receipt.json)

Per CN il continuo e due processi di ripresa indipendenti sono partiti dal medesimo confine al passo 2 e sono stati confrontati al passo 4. Batch, stato RNG CPU, contatori e ordine della coda concordano su ogni rank. I tre confronti superano il criterio dell’1%. Sui 220 tensori mobili il massimo scarto relativo del continuo rispetto a una ripresa è 0,000114%, fra le due riprese è zero. Verificati lo SHA e l’integrità dell’archivio delle tracce. La prova chiude P-5 sull’orchestrazione locale in FP32. Il comportamento a dose reale con bf16/CUDA sarà misurato sul cluster. [Ricevuta delle riprese](../../results/infra/contrastive_targets_20261118/gloo_p5_20261120/reload/reload_result.json)

Rilascio a 3-launcher del checkout e della sola calibrazione RMS100/fast125, con la review e il budget firmati. Nessun nuovo passaggio scientifico o ritocco dei pesi è richiesto prima di questa misura. Per il seguito si legge il rapporto autenticato della calibrazione e si decide sulla successiva cella CUDA.
