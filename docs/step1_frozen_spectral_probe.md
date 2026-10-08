# Step 1 — prima sonda spettrale frozen

Protocollo proposto l'8 ottobre 2026. Questa pagina prepara la prima misura
scientifica dopo i collaudi funzionali PSD/Step 1; non autorizza da sola job o
spesa HPC.

## Domanda

Quanto sono linearmente leggibili, nel teacher EMA congelato, la forma e
l'energia dello spettro medio multiscala definite da
`step1_psd_power_shape_energy_v1`?

Questa e' una diagnostica del checkpoint baseline. Non misura ancora l'effetto
della loss keep, non e' un confronto A/B e non e' generalizzazione a soggetti
mai osservati dal pretraining.

L'estrazione e' senza masking (`visible=None`), quindi non esercita il caso di
softmax del Perceiver con un istante interamente nascosto. L'eventuale patch
`safe_scores` resta una bonifica numerica P2 separata e non va mescolata a
questa baseline.

## Checkpoint e readout

- Checkpoint: teacher EMA finale di
  `$SCRATCH/wearusfm_runs/runs/sanity_0410/checkpoint.pt`, 10.000 passi.
- SHA256 verificato in sola lettura su Leonardo:
  `fa7b5a7ae8cd6c2c13ef1fe42fa79e07e370d2831fd2782f2a0a1d768fb7cc96`.
  Viene controllato sia dall'estrazione sia dalla sonda.
- Readout: `p1p2_qc_equal_channel_v1`, nelle tre viste `backbone`, `local` e
  concatenazione `p1p2`.
- Il checkpoint PSD `w1_psd_0710` da 625 passi non e' la baseline primaria:
  e' ancora nel warm-up e potra' essere misurato separatamente, senza mescolare
  i risultati.

## Dati e split

- Dataset omogeneo: `emg2qwerty`.
- Manifest v1.1, scale del job 59623700 e durate del job 59425316.
  Verificati presenti su Leonardo; SHA256 rispettivamente
  `853f35ac5252ef901be390385164c6dc5658f45270441eb74557c6be317ed6d3`,
  `76486d882b5deeca66f152c3541534e9f35efca35f944f6f8677b6478b877486`
  ed `e660e324a28a339c038ed07dbaa7d4c358f78f613cd32226eea3f522119a48b0`.
- Split soggetti gia' congelato dal job 59638890: 80 train, 10 validation,
  10 test, seed 0; SHA256
  `e064a1102bb4a7dd1b01a70caefa8fa2c58d8c9c751bafeafe6bd2892fff7a30`.
- 100 finestre native del loader per ogni soggetto e per ogni parte: 10.000
  finestre totali, crop da 1--4 s, stesso seed 0.
- Il target viene ricalcolato sugli stessi crop usati per il readout; non si
  riusano cache P1/P2 storiche.

## Sonda

- Encoder in `eval`, senza masking e senza gradienti.
- Ridge separata per ciascuna delle 32 coordinate.
- Standardizzazione delle feature e del target fit soltanto sul train.
- Alpha in `{0.1, 1, 10, 100}`, scelto soltanto sulla validation; test letto una
  volta dopo la selezione.
- Soggetti disgiunti fra train/validation/test, target mancanti esclusi e R2
  negativi conservati.
- Risultati principali: R2 macro separato per `fast_shape`, `fast_energy`,
  `slow_shape`, `slow_energy`, per ciascuna delle tre viste. Si riportano anche
  le 32 coordinate e i risultati per soggetto.
- Nessun bootstrap in questa prima misura: i valori per soggetto sono
  descrittivi. Un eventuale intervallo va aggiunto con ricampionamento dei
  soggetti, non delle finestre sovrapposte.

## Criteri di validita' fissati prima del risultato

La misura e' valida soltanto se:

1. checkpoint, split, manifest, subset e preprocessing hanno hash/provenienza
   completi e coerenti;
2. i tre split sono non vuoti e senza soggetti condivisi;
3. vengono estratte 10.000 righe finite e tutte le 32 coordinate risultano
   valutabili, salvo indisponibilita' fisica registrata esplicitamente;
4. il report contiene tutte le tre viste, le quattro famiglie e le 32
   coordinate, senza troncare gli R2 negativi;
5. nessun risultato di test viene usato per cambiare checkpoint, split, alpha,
   numero di finestre o readout.

Non viene fissata una soglia di successo del modello: questa prima sonda crea la
baseline. Un R2 positivo non dimostra miglioramento dell'embedding; il confronto
scientifico richiedera' i bracci A/B accoppiati e nuove sonde frozen.

## Risorse proposte

- Estrazione: una A100, 8 CPU, massimo 30 minuti su `boost_qos_dbg`, quindi
  massimo 0,5 GPU-ora (= 4 ore locali). Stima attesa inferiore, sulla base delle
  estrazioni P1/P2 gia' concluse.
- Ridge: 8 CPU, massimo 1 ora su `lrd_all_serial`, 0 GPU-ora e massimo 8
  core-ora CPU.
- Budget passo 6 prima del lancio: circa 12,1/100 GPU-ora usate e 87,9 residue;
  dopo il tetto massimo resterebbero circa 87,4 GPU-ora.

Il lancio richiede approvazione esplicita di questi tetti. Le uscite restano
fuori Git; nel repository entra poi soltanto il riepilogo verificato.

## Esito dell'8 ottobre 2026

Simone ha autorizzato i due job entro i tetti sopra. Estrazione `59703237` e
ridge `59703306` sono entrambe `COMPLETED 0:0`; la dipendenza `afterok` e' stata
rispettata. Costo: 0,4953 GPU-ora e circa 4,00 core-ora CPU.

Le 10.000 righe e tutte le 32 coordinate sono finite e valutabili; split
8000/1000/1000, soggetti 80/10/10 senza sovrapposizioni. R2 macro:

| vista | fast shape | fast energy | slow shape | slow energy |
|---|---:|---:|---:|---:|
| backbone | 0,8526 | 0,9160 | 0,7895 | 0,9158 |
| local | 0,7848 | 0,8905 | 0,7385 | 0,9011 |
| p1p2 | **0,8634** | 0,9142 | **0,7975** | 0,9154 |

Il concatenato P1/P2 e' il migliore sulle due famiglie di forma in questa
baseline. Non e' ancora un miglioramento causato dalla keep: servira' il
confronto A/B. Dettagli, provenance e hash in
`results/passo6/step1_frozen_probe_20261008.json`.
