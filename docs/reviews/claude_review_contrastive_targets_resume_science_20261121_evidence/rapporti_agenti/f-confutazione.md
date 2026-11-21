# Compito F (f-confutazione): avvocato del diavolo sulla tesi "rumore numerico, non difetto di ripristino"

Clone: `codex/contrastive-target-windows-20261118` @ `918245b6cee89fb71e159c0fe17f4a7c2e746fa7` (verificato con `git log -1`).
Rapporto: `data/preflight_readout.json` (3.081.874 byte). Script miei (sola lettura) in `agents/f-confutazione/`:
`stats.py`, `steps.py`, `ab.py`, `frac.py`, `growth.py`, `corr.py`, `counts.py`. Abbreviazioni: E = `results/infra/contrastive_targets_20261118`,
RJ = rapporto JSON, coppia = `continuous__reload_a` (identica a `continuous__reload_b` in tutti i campi letti).

## Verdetto in breve

La tesi regge nella sostanza, ma due dei suoi passaggi non sono dimostrati e uno e' quantitativamente sbagliato.

- **Regge (forte):** non e' un difetto di ripristino dello stato salvato. Lo stato a 545 coincide per hash su 4 rank e 2 bracci; tutti i
  conteggi discreti coincidono per 32 passi; il primo scarto a 545 sta nei gradienti ridotti da DDP mentre i gradienti diagnostici
  calcolati fuori da DDP coincidono bit per bit.
- **Regge (media):** i quattro tensori bocciati sono i quattro con l'errore piu' alto su 270 e sono gli unici che ricevono solo il
  gradiente InfoNCE; la loro "anomalia" exp_avg_sq/exp_avg e' spiegata per circa due terzi dalla storia corta del proiettore (rampa 257-512).
- **Non dimostrato:** "rumore comune ai due bracci". Nell'encoder CN ha errori di primo momento 1,63 volte quelli di CR (mediana sui
  266 tensori attivi), le chiavi della coda 1,33 volte; lo spostamento dei pesi invece solo 1,05 volte. Il livello non e' comune.
- **Non dimostrato:** "un livello di rumore" raggiunto in uno o due passi. In CN il primo scarto visibile e' al 547 (al 546 le norme
  coincidono esattamente) e la divergenza delle norme cresce di circa 7 volte nei 32 passi senza stabilizzarsi.
- **Non dimostrato (meccanismo):** il "diverso raggruppamento DDP" come causa. Al 546 la ripresa usa ancora i bucket non ricostruiti, ma
  in CN la norma del gradiente ridotto e quella del proiettore coincidono esattamente col continuo. E' un effetto del primo passo DDP
  dopo l'avvio, non dimostrabilmente della disposizione dei bucket.
- **Parzialmente vero:** "CR passa solo perche' non ha tensori di quel tipo". CR ha gli ID 266-269 nello stato, ma con errore 0 (gradiente
  nullo); in piu' ha margine (massimo 0,59%) perche' tutto il suo rumore e' piu' basso di quello di CN (massimo encoder CN 0,714%).

## 1. Numeri citati nella tesi (verificati sul JSON)

Comando: `python -I agents/f-confutazione/stats.py data/preflight_readout.json`. Fonte: `arms.<arm>.resume.pairs.continuous__reload_a.floating_tensor_relative_to_left`.

| | CN | CR |
|---|---|---|
| tensori `exp_avg` | 270, tutti != 0 | 270, 266 != 0 (266-269 = 0,0) |
| mediana exp_avg (non nulli) | 0,4613% | 0,2897% |
| massimo exp_avg | 1,3165% (ID 268) | 0,5913% (ID 47) |
| massimo exp_avg esclusi 266-269 | 0,7144% (ID 194) | 0,5913% |
| mediana exp_avg_sq | 0,0096% | 0,0059% |
| chiavi coda: non nulle / 5980 | 910, mediana 0,53%, max 0,92% | 940, mediana 0,40%, max 0,60% |

Verificato: 270 e 266, mediana 0,29% e massimo 0,59% di CR; i quattro fuori soglia (1,2216/1,0342/1,3165/1,1084%) sono i primi quattro
dei 270 di CN. Nota: le chiavi della coda di CN arrivano a 0,92%, quindi anche quel criterio era vicino alla soglia.

## 2. Stato al confine e conteggi: contro H1 (verificato)

- Tracce `E/{benchmark,reloads}_execution_20261121/{CN,CR}[_A]/trace-{0..3}.jsonl`: confrontando continuo e ripresa A, campo per campo
  (escluso `phase`), **nessuna differenza** in 32 passi x 4 rank x 2 bracci: hash dei batch, degli input, RNG CPU/CUDA prima e dopo,
  ordine/lunghezza della coda, copertura, `contrast_nce_weight`, e al 545 l'hash `boundary` di studente, teacher, `opt.state_dict()["state"]`
  (comprende `step`), `param_groups`, stato contrastivo (codice: `scripts/contrastive_target_resume.py:181-187`).
- `counts.py`: 5.965 confronti di conteggi (idonei, negativi, eta', chiavi sostituite, coppie RnC, finestre...) per braccio, **0 differenze**;
  `origin` identico a ogni passo. Nessuna biforcazione discreta (chiave ammessa o negativo diverso) che possa fare da amplificatore.
- Stati non coperti dall'hash: stato interno del Reducer DDP (documentato diverso), stato di NCCL/cuBLAS/allocatore. I buffer non persistenti
  (`src/wearusfm/model/frontend.py:120-125`) sono costanti ricostruite in `__init__`, non stato di training.

Conclusione: un difetto nel ripristino di cio' che si salva e' escluso. Resta aperta solo una dipendenza dalla storia del processo.

## 3. Dove nasce lo scarto (545-547)

`steps.py` e `ab.py` su `metrics.jsonl` (continuo righe 545-576, ripresa righe 1-32).

| passo | CN grad_norm (rel) | CN norma proiettore (rel) | CR grad_norm (rel) |
|---|---|---|---|
| 545 | -2,2e-12 (abs 9,54e-12) | -1,0e-13 | -3,4e-13 |
| 546 | **0 esatto** | **0 esatto** | -3,4e-6 |
| 547 | 2,2e-5 | 6,5e-5 | 1,3e-4 |
| 548 | -9,1e-5 | -1,1e-3 | -3,1e-5 |

Valori grezzi CN 546: `9.023137809933427` e `0.21809202574133565` identici in continuo, A e B.

- **A favore della tesi (dedotto, forte):** al 545 i gradienti diagnostici `contrast_global_grad_{base,nce,rnc}` e i coseni sono
  **identici bit per bit** fra continuo e ripresa (in CN e CR), mentre `grad_norm` e la norma del proiettore differiscono. I diagnostici
  sono calcolati con `torch.autograd.grad` e ridotti con un solo `all_reduce` piatto, fuori dai bucket DDP
  (`src/wearusfm/training/contrastive_objectives.py:415-436`, chiamata a `src/wearusfm/training/run.py:796-797`, prima di `loss.backward()`).
  Quindi forward e backward locali dell'encoder al primo passo coincidono; lo scarto compare nel gradiente ridotto da DDP. Limite: copre i
  parametri `SHARED_PREFIXES`, non il proiettore, e non osserva i gradienti locali del backward vero.
- **Contro il meccanismo "raggruppamento DDP" (verificato il dato, dedotta l'implicazione):** `trace-0.ddp.jsonl` della ripresa riporta
  `bucket_sizes "114202292, 68480"` sia al 545 sia al 546; i bucket ricostruiti come nel continuo compaiono dal 547
  (`E/readout_execution_20261121/ddp_trace_readout.json`: differenze ai soli update 545 e 546). Se la disposizione dei bucket bastasse a
  cambiare la riduzione, il 546 dovrebbe differire come il 545. In CN invece la norma totale (in float64, perche' il front-end e' in
  float64: `run.py:12`, valore non rappresentabile in fp32) e quella del proiettore (accumulata in float64, `run.py:1690-1693`) sono
  identiche. Una differenza di 1 ulp fp32 in un elemento tipico sposterebbe la seconda di circa 1e-15 relativo. Conclusione: lo scarto e'
  un effetto del **primo** passo DDP dopo l'avvio, non dimostrabilmente della disposizione dei bucket. Al 546 i coseni diagnostici CN
  differiscono di circa 5e-16 (per l'aggiornamento del 545), quindi l'identita' al 546 non e' totale.
- **"1 ulp" (dedotto, debole):** al 545 la norma del proiettore differisce di 1e-13 relativo, compatibile con pochi elementi diversi di
  1 ulp fp32. Il dato elementare non esiste negli artefatti.
- **"in uno o due passi" (non vero per CN):** CN diventa visibile al 547 (terzo update), CR al 546.
- **Specificita' InfoNCE a 545-546: nessuna.** `contrast_nce` e tutte le `contrast_nce_loss__*` coincidono bit per bit a 545 e 546 in CN;
  al 547 differisce un solo dataset su 8 (`ninapro_db10`, 1,8e-5), dal 548 tutti (13/13). In CR al 547 differiscono gia' 8/8.

## 4. Crescita nei 32 passi: il "livello" non c'e' (verificato)

RMS dello scarto relativo per blocchi di 8 passi (`growth.py`):

| campo | 545-552 | 553-560 | 561-568 | 569-576 |
|---|---|---|---|---|
| CN grad_norm | 1,5e-4 | 4,8e-4 | 6,5e-4 | 1,1e-3 |
| CR grad_norm | 1,6e-4 | 2,5e-4 | 7,9e-4 | 8,4e-4 |
| CN norma proiettore | 1,3e-3 | 2,1e-3 | 1,6e-3 | 1,5e-3 |
| CN jepa | 6,1e-5 | 1,3e-4 | 1,8e-4 | 4,3e-4 |
| CR jepa | 4,3e-5 | 7,9e-5 | 2,3e-4 | 3,4e-4 |
| CN contrast_nce | 5,3e-5 | 1,1e-4 | 1,2e-4 | 1,7e-4 |

L'intervallo 1e-4/1e-3 della tesi e' giusto come escursione, ma e' una crescita di 5-7 volte, non una saturazione. Solo la norma del
proiettore e' piatta fin dal primo blocco, circa 10 volte sopra la norma totale: questo sostiene il "rumore relativo piu' alto".
Implicazione (dedotta): l'esito dell'1% dipende dall'orizzonte di 32 passi; con un orizzonte piu' lungo anche CR potrebbe superarlo.

Rumore di log: A e B hanno checkpoint identici ma differiscono a ogni passo in 6-7 campi di loss (`jepa`, `total`, `band_shape`,
`log_*`, `ms_*`) di circa 1e-7 (`ab.py`). Quegli scarti fra continuo e ripresa al 545-546 non provano niente (accordo con GPT).

## 5. Struttura dello scarto

- **CN contro CR nell'encoder (verificato):** rapporto per tensore exp_avg CN/CR, mediana 1,627 (q10 0,984, q90 1,933; 266 tensori);
  chiavi 0,53/0,40 = 1,33; distanza dei pesi dal fork 0,6454/0,6164 = 1,05 (`relative_to_left_displacement_from_544.student`). La tesi
  ("comune") non spiega perche' InfoNCE alza di 1,6 volte l'errore dei momenti dell'encoder lasciando quasi invariato quello dei pesi.
  Ipotesi da verificare: exp_avg piu' piccoli in norma per cancellazione fra passi, quindi denominatore piu' piccolo. Servono le norme assolute.
- **Stessa sensibilita' per tensore (a favore, media):** correlazione dei log-errori CN/CR sui 266 tensori: Pearson 0,71, Spearman 0,57
  (exp_avg); 0,75/0,65 (exp_avg_sq). I tensori piu' colpiti in CN (194, 158, 212, 176, 140, 122, 104) hanno passo 18 (stesso ruolo in
  ogni blocco) e 194, 158, 176, 212, 140 sono fra i primi 30 anche in CR.
- **Pesi contro bias del proiettore:** W1/b1 = 1,181, W2/b2 = 1,188. Coerenti fra i due strati, nessun segno di un singolo tensore rotto.
- **exp_avg_sq/exp_avg:** proiettore 0,127-0,150; encoder CN mediana 0,020, q95 0,080, massimo 0,233. Il proiettore riceve gradiente solo
  dal 257 (`contrast_nce_weight` 0 fino al 256, poi rampa fino a 0,1185 al 512; `metrics.jsonl` CN). Con beta2 = 0,999 (AdamW senza betas:
  `run.py:1429`) e norma dopo clipping del proiettore come peso, gli ultimi 32 passi pesano il 29,05% di exp_avg_sq contro il 7,19%
  dell'encoder (`frac.py`): fattore 4,0 sui circa 6,8 osservati. Il residuo di circa 1,7 resta dentro la dispersione dell'encoder.
- **Clipping (verificato):** fattore < 1 in 576/576 passi; la norma dopo il taglio vale 1,0 a ogni passo; fattore 0,10-0,23 nei passi
  545-576. Una differenza relativa della norma totale (al massimo circa 2e-3) riscala tutti i gradienti: in exp_avg pesa al massimo
  circa 0,1%, troppo poco per lo 0,46% mediano o per l'1,2%. Il clipping non e' il canale principale (dedotto).
- **Coerenza 1% contro 1e-3:** l'errore di exp_avg (circa 97% del peso negli ultimi 32 passi con beta1 = 0,9) richiede differenze
  vettoriali per passo dell'ordine dell'1% sul gradiente del proiettore, cioe' circa 10 volte quelle della sua norma (1,3-2,1e-3).
  E' compatibile con perturbazioni in buona parte ortogonali, ma non e' verificabile: mancano i vettori. Le chiavi (uscite bf16 del
  proiettore del teacher normalizzate in fp32: `contrastive_objectives.py:246-250`, `model/contrastive.py:53-58`) hanno errori di
  0,4-0,9%, cioe' alcune unita' di arrotondamento bf16 (2^-8 = 0,39%): l'ordine di grandezza combacia.

## 6. Nodi e job (dedotto dai tempi)

Il nodo non e' registrato negli artefatti raccolti. `E/reloads_execution_20261121/execution_timeline.sacct.tsv` e `allocation.sacct.tsv`
mostrano job da 4 GPU su `node=1`. Si sovrappongono CN_B (65033756, 14:58:27-15:02:00) e CR_A (65033804, 14:59:45-15:03:10), e anche CR_A e
CR_B (65034107, dalle 15:02:51). Su nodi da 4 GPU, quindi, **CR_A e CR_B sono girati su nodi diversi** e producono comunque checkpoint
identici bit per bit. Per CN_A e CN_B (14:49-14:53 e 14:58-15:02) non si puo' dire. Il continuo e' un altro job. L'effetto nodo resta
possibile solo fra continuo e ripresa: lo esclude soltanto una prova sullo stesso nodo.

## 7. Cosa la tesi non spiega o lascia aperto

1. CN 1,6 volte CR anche nell'encoder (sezione 5), con pesi quasi alla pari.
2. L'origine al solo primo passo DDP, con il 546 identico a parita' di bucket (sezione 3). La riduzione diversa e' localizzata, il
   meccanismo no.
3. La crescita senza plateau (sezione 4): con un altro orizzonte l'esito del criterio puo' cambiare.
4. La scala assoluta: senza norme e coseni dai checkpoint non si sa se l'1,2% sia di un vettore grande o di uno quasi cancellato.

## 8. Controllo minimo e cosa mi convincerebbe

- **Lettura CPU sui checkpoint esistenti a 576 (costo GPU zero; da fare sul cluster, non sul Mac):** per gli ID 266-269 e per un campione
  dell'encoder (per esempio 194, 158, 47), ||m_c||, ||m_r||, ||m_c-m_r||, coseno, e la stessa cosa per exp_avg_sq e pesi. Atteso con H2:
  coseno >= 0,999 e ||m|| del proiettore non anomala rispetto al gradiente; coseno o norme del proiettore molto diversi dall'encoder
  riaprirebbero H1.
- **Se si fa la prova GPU proposta da GPT,** due falsificatori: (a) gradienti **locali** del proiettore gia' diversi al 545 (prima della
  riduzione) smentiscono la causa DDP; (b) la ripresa strumentata che al 546 differisce come al 545 conferma l'effetto della disposizione dei
  bucket, mentre l'identita' al 546 lo indebolisce ulteriormente. Controllo della strumentazione: lo stesso continuo strumentato deve
  riprodurre bit per bit `grad_norm` e `contrast_projector_grad_norm` del continuo storico ai passi 545-547 (valori della tabella in
  sezione 3), e la ripresa strumentata deve riprodurre i valori di CN_A/CN_B (identici fra loro).
- **Mi convincerebbe a tornare ai training lunghi:** gradienti locali identici bit per bit al 545 e primo scarto nel vettore ridotto, con lo
  scarto finale a 576 riprodotto (exp_avg 266-269 circa 1,0-1,3%, coseni >= 0,999). Questo direbbe che e' la riduzione, non lo stato.
  Se invece i gradienti locali differiscono gia' al 545, o la prova strumentata non riproduce lo scarto, la causa resta aperta e non tornerei
  ai training lunghi con la ripresa da checkpoint come garanzia di continuita'.
