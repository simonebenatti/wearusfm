# T100-LQ: lettura del benchmark di inferenza al passo 5.000

Job 63614514, COMPLETED 0:0, 110 secondi, 0,030556 GPU-ore e 0,244444 core-ore.
Ricevuta originale `results/infra/logsquare_inference_20261105/completed/receipt.json`,
SHA256 `7e6e665eb4fb321b81266d0f35806c61814958e758c236a8266baae945db4e9a`.
Checkout `cc814303e52d8e4067cf12fb1a291459a402c78c`.

## Verifica indipendente e verdetto

Il critico lq_benchmark_readiness ha ricalcolato senza usare le funzioni di riepilogo
128 misure in otto record, controllando ordine alternato, due warmup e sei misure
per braccio, supporto dei sei gruppi, pin, contesto e 110 sorgenti sul checkout e sui
file locali. Conteggi storici ricostruiti dalle provenienze locali. SHA originali di
ricevuta e log coincidono con la raccolta. Verdetto: APPROVATO per pose_5000 con tetto
3.000 s, dopo pubblicazione del nuovo piano e checkout congelato.

| Cella | rho massimo | rho ponderato | Stima proxy, s | Tetto, s |
| --- | ---: | ---: | ---: | ---: |
| pose_5000 | 1.002825 | 1.002825 | 1801.28 | 3000 |
| pose_19618 | 1.002825 | 1.002825 | 1718.85 | 3000 |
| p1_19618 | 1.012804 | 1.012574 | 891.11 | 2100 |
| p2_19618 | 1.013405 | 1.006351 | 1678.13 | 3000 |

Risorse di ogni cella: 1 GPU, 8 CPU, 100G. La somma dei tetti per le quattro estrazioni
è 3,083333 GPU-ore / 24,666667 core-ore, oltre ai checkout. Ora si può lanciare soltanto
pose_5000 (0,833333 GPU-ore / 6,666667 core-ore, più 0,333333 core-ore di checkout).
Il checkpoint a 19.618 non è ancora completo.

Il rapporto misura il solo forward, con FP32 per posa e BF16 con decoder per
classificazione. L'estensione alle durate complete storiche rimane un proxy.
Il vecchio riepilogo locale omette l'elapsed storico di pose_5000: per quella sola
stima puntuale il critico usa l'attestazione del benchmark. Il tetto di 3.000 s è
invece verificato indipendentemente dal walltime storico pinnato di 2.700 s.
Il gate runtime rilegge anche le ricevute storiche originali sul cluster.

## R1-b: norme del ramo log-quadratico

Tutte le 64 colonne di W_log hanno norma positiva, sia nello student sia nel teacher.
Nel teacher le norme medie sono 0,1383 per Fourier, 0,0655 per spline e 0,1069 per MLP.
Gli MLP, 16 colonne su 64, rappresentano il 24,42% della somma delle norme quadratiche.
Il ramo MLP non è rimasto spento. Le norme non misurano il contributo alle predizioni
né dimostrano un beneficio scientifico; epsilon rimane 0,001.

Nessun score di posa o classificazione è stato prodotto da questo benchmark.

Il critico ha poi verificato anche i due piani nuovi: APPROVATI per lo stato
APPROVED_FOR_EXTRACTION. Il controllo allocation_for sul cluster ha riletto tutti
i riferimenti storici originali ed è passato; evidenza in
`results/infra/logsquare_evaluation_20261105/prelaunch_gate.json`.
