# Compito A — Mappatura dei parametri e denominatori (linea 3, riprese CN/CR)

Clone `codex/contrastive-target-windows-20261118` a `918245b6cee89fb71e159c0fe17f4a7c2e746fa7` (verificato con `git log -1`).
Rapporto: `data/preflight_readout.json` (campi letti sotto). Script miei: cartella `agents/a-mappatura-denominatori/`
(`build_params.py`, `fp.py`, `fp2.py`). Nessun accesso a Leonardo, nessuna modifica al clone.
Legenda: **[V]** verificato (codice, comando o JSON); **[D]** dedotto.

## 1. Ordine dei parametri e ID 266–269

**Da dove viene l'ID.** Il lettore confronta `finals[left]["optimizer"]`, cioe' `opt.state_dict()` salvato
(`scripts/contrastive_target_resume.py:256-258`), percorrendolo con `_nested_equal`/`_floating_distances`
(`:226-239`, `scripts/contrastive_benchmark_readout.py:130-157`): il percorso `optimizer.state.<ID>.exp_avg` usa l'indice
intero che PyTorch assegna nello `state_dict` dell'ottimizzatore, cioe' la posizione del parametro nella lista passata ad AdamW. [V]

**Costruzione di AdamW.** `src/wearusfm/training/run.py:1387-1390`: `WearUsFM(cfg.model)` poi `attach_contrastive_head(student)`
(il proiettore viene registrato per ultimo come `model.contrastive_head`, `src/wearusfm/training/contrastive_objectives.py:114-121`);
`run.py:1429`: `torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)`:
**un solo gruppo**, nessuna esclusione di bias/norme dal weight decay, nessun iperparametro dedicato al proiettore. [V]
`ProjectionHead` = `Sequential(Linear(dim,hidden), GELU, Linear(hidden,128))`, `hidden=dim` (`src/wearusfm/model/contrastive.py:15-27`). [V]

**Ricostruzione su CPU** (`agents/a-mappatura-denominatori/build_params.py`, config `benchmark_execution_20261121/{CN,CR}/config.json`,
torch locale 2.14). Output identico per CN e CR:
```
n trainable 270  total elems 28559133  freeze False  lr 0.0001 wd 0.05 clip 1.0
260 anchor_heads.ms.ms_slow.weight (24, 384) 9216
261 anchor_heads.ms.ms_slow.bias (24,) 24
262 pose_head.norm.weight (384,) 384
263 pose_head.norm.bias (384,) 384
264 pose_head.out.weight (20, 384) 7680
265 pose_head.out.bias (20,) 20
266 contrastive_head.layers.0.weight (384, 384) 147456
267 contrastive_head.layers.0.bias (384,) 384
268 contrastive_head.layers.2.weight (128, 384) 49152
269 contrastive_head.layers.2.bias (128,) 128
groups 1: lr 1e-4, betas (0.9, 0.999), eps 1e-08, weight_decay 0.05, amsgrad False
head elems 197120   (0,69% dei 28.559.133 parametri)
```
**La mappatura di GPT e' confermata** [V]: 266/267 = peso/bias del primo lineare (384×384), 268/269 = peso/bias del secondo (128×384).
Riserva [D]: il nome si ricava dall'ordine di costruzione del codice del clone; presuppongo che i moduli del modello siano gli stessi
del sorgente congelato del run (l'impronta sotto conferma numero e iperparametri, non i nomi).

**Impronta `864a86de…9837`.** E' `state_sha256` di `opt.state_dict()["param_groups"]` con `lr` portato a `_lr_at(544)`, scritto
al confine come `boundary.optimizer_groups_at_next_lr` (`contrastive_target_resume.py:181-188`; atteso ricostruito in `:314-319`). [V]
Compare in **24/24** prime righe di traccia (2 bracci × 3 percorsi × 4 rank):
`bash -c 'for f in .../C{N,R}/trace-?.jsonl .../C{N,R}_{A,B}/trace-?.jsonl; do head -1 $f | grep -o ...; done | sort | uniq -c'`
→ `24 "optimizer_groups_at_next_lr": "864a86de…9837"`. [V]
Riproduzione (`fp.py`, `fp2.py`): con le chiavi di AdamW di **torch 2.14** (che aggiunge `decoupled_weight_decay`) ottengo
`1cfdcaef…4251`, **non** l'impronta; con le chiavi di torch 2.2 (versione del run: `numeric_profile.rank0.json` →
`"torch_version": "2.2.0a0+git6c8c5ad"`) e `params = list(range(270))`, `lr = 2.777777777777778e-05`, ottengo esattamente
`864a86decd9d15929e4bf5a0400908c0697ae08ded112e95b5db8140570e9837` (`MATCH n 270`; 265–274 provati, solo 270 combacia). [V]
Cosa dimostra: un solo gruppo, 270 parametri, iperparametri e lr del passo 545 identici nei 24 processi. **Non** contiene i nomi
(i `params` sono interi), quindi non prova da sola la mappatura ID→nome: quella viene dall'ordine di costruzione. [V/D]

## 2. Perche' in CR gli ID 266–269 hanno errore esattamente 0

- Il proiettore esiste anche in CR: `attach_contrastive_head` e' chiamato per qualunque `contrastive is not None` (`run.py:1388-1390`),
  e `ContrastiveRuntime.attach` calcola sempre `q_raw = student.contrastive_head(q_read)` (`contrastive_objectives.py:245`). [V]
- Il proiettore entra solo nella perdita InfoNCE; Rank-N-Contrast usa `zm = enc.z.float().mean(2)` dell'encoder
  (`contrastive_objectives.py:272, 288`). La perdita totale e' `sum(weight[k]*n*ws/den)` (`jepa.py:275`): in CR il peso NCE e' 0
  (metriche CR: `contrast_nce_weight 0.0` ai passi 544–576), quindi il gradiente del proiettore e' uno zero esatto. [V]
- Metriche: `contrast_projector_grad_norm` > 0 in **0/576** passi di CR e in 320 passi di CN (257–576, dopo `fill_steps=256`). [V]
- Gli stati 266–269 esistono in CR (compaiono nel dizionario `floating_tensor_relative_to_left` con 6.520 tensori come in CN): il
  gradiente e' un tensore zero, non `None` (DDP gira con `find_unused_parameters=False`, `run.py:436-438`, `rvq_codes None`,
  `keep_readout False`). Quindi `exp_avg = exp_avg_sq = 0` in tutti i percorsi e l'errore e' `0/max(0,1e-12) = 0`. [V per i fatti, D per la catena]
- Il peso del proiettore CR cambia solo per il weight decay disaccoppiato, identico nei tre percorsi: nel JSON
  `by_block_relative_to_left_displacement_from_fork.student.contrastive_head = 0.0` per CR. [V]
- **Conseguenza [D]:** il superamento di CR non dice nulla sul percorso del proiettore: per quei quattro tensori CR non ha segnale.
  Il confronto CN/CR sui tensori 266–269 non e' un controllo.

## 3. Denominatori del lettore

Formula unica, `scripts/contrastive_benchmark_readout.py:114-127`:
`err = sqrt(Σ‖x−y‖²) / max(sqrt(Σ‖s‖²), 1e-12)`, in float64, con `s = x` (senza riferimento) o `s = x − ref` (con riferimento);
`x` e' sempre il **primo membro** della coppia. [V]

(a) **Momenti di Adam e chiavi della coda**: `_nested_equal(left[part], right[part], part, RELATIVE_LIMIT, floating_errors)`
per `part in ("optimizer","contrastive_state")` (`contrastive_target_resume.py:256-258`) → `_tensor_distance({'x':a},{'x':b})`
(`readout.py:136`): **norma L2 del singolo tensore del primo membro**, pavimento 1e-12; violazione se `error > 0.01` (`readout.py:139`).
I `.step` sono esclusi dalla tolleranza ed esigono uguaglianza esatta (`readout.py:135, 143`). [V]

(b) **Studente e teacher**: `_tensor_distance(left_params, right_params, initial_params)` (`contrastive_target_resume.py:251-253`)
= norma della differenza / **norma dello spostamento del primo membro dall'inizializzazione**, solo parametri (i buffer esigono
uguaglianza esatta, `:249-250`), pavimento 1e-12. L'inizializzazione e' `saved['student']` del checkpoint di calibrazione a peso fisso
(`readout.py:276-282`: `step == 512`, ottimizzatore vuoto, `state_sha256(student) == state_sha256(teacher) == initial_state_sha256`).
Lo stesso `initial` serve anche da riferimento per il teacher (lecito: alla partenza teacher = studente). [V]

(c) **Lettura dal passo 544** (descrittiva, non entra nel `passed`): stessa formula con riferimento `fork[part]`, totale e per blocco
(prefisso del nome) (`contrastive_target_resume.py:274-286`). [V]

**Ordine delle coppie**: `itertools.combinations(("continuous","reload_a","reload_b"), 2)` (`:396`): il continuo e' il primo membro
in due coppie su tre; nella coppia A/B il primo membro e' `reload_a`. [V]

**Valori nel JSON** (`arms.<arm>.resume.pairs.continuous__reload_a`, identici in `continuous__reload_b`): [V]

| grandezza | JSON | GPT |
|---|---|---|
| CN 266/267/268/269 `exp_avg` | 0.01221615662445262 / 0.01034195290681912 / 0.013164698202657702 / 0.01108410806754013 | 1,222 / 1,034 / 1,316 / 1,108 % ✔ |
| CN 266–269 `exp_avg_sq` | 0.0016646 / 0.0015466 / 0.0016754 / 0.0015515 | "fra 0,155% e 0,168%" ✔ |
| CN studente/teacher da init | 0.0008974916605155586 / 8.213847318201e-05 | 0,08975% / 0,00821% ✔ |
| CR studente/teacher da init | 0.000808159672839074 / 7.268938902318168e-05 | 0,08082% / 0,00727% ✔ |
| studente da 544, CN / CR | 0.006454380772990552 / 0.006163708722546596 | 0,6454% / 0,6164% ✔ |
| teacher da 544, CN / CR | 0.0008190175 / 0.0007384370 | non citati |
| A/B, entrambi i bracci | tutto 0.0 | ✔ |

`boundary_mismatches = []`, `missing_active_support = []`, differenze di input/RNG/ordine = 0 in tutte le coppie, CN e CR. [V]

**Contesto che il JSON gia' contiene** (`floating_tensor_relative_to_left`, 270 `exp_avg`, 270 `exp_avg_sq`, 5.980 chiavi di coda): [V]

| | CN | CR |
|---|---|---|
| `exp_avg`, mediana sui 270 | 0,461% | 0,288% |
| `exp_avg`, 90° percentile | 0,588% | 0,403% |
| `exp_avg`, massimo fuori dal proiettore | 0,714% (ID 194) | 0,591% (ID 47) |
| `exp_avg`, proiettore 266–269 | 1,03–1,32% | 0 (nessun gradiente) |
| chiavi di coda, massimo | 0,919% (entry 5538) | 0,604% |
| studente da 544, blocco `contrastive_head` | 0,932% | 0,0 |
| studente da 544, `local` / `pool` | 1,028% / 1,047% | 0,886% / 0,870% |

Lettura [D]: lo scarto e' **graduato e diffuso**, non concentrato in un tensore "non ripristinato": tutti i 270 `exp_avg` di CN
differiscono (mediana 0,46%), le chiavi della coda arrivano a 0,92%, i blocchi `local` e `pool` dei **pesi** differiscono dal 544
piu' del proiettore stesso (1,03–1,05% contro 0,93%). Un ripristino mancato darebbe errori dell'ordine del 100% o zeri esatti su
tensori specifici, oppure una discrepanza gia' al confine (che i 24 hash escludono). Il proiettore e' la coda alta di una distribuzione
continua; in CN l'intero modello deriva ~1,6 volte piu' che in CR (mediana 0,46 contro 0,29%).

**Asimmetria dei denominatori [D, aritmetica su β di AdamW verificato].** Con β1 = 0,9, al passo 576 `exp_avg` e' per
`1 − 0,9³² = 96,6%` costituito da gradienti successivi al fork (orizzonte effettivo ~10 passi); `exp_avg_sq` (β2 = 0,999) solo per
`1 − 0,999³² = 3,1%`: il secondo momento e' dominato dalla storia comune, ecco perche' i suoi errori sono ~7–10 volte piu' piccoli.
I parametri invece sono giudicati contro 576 passi di spostamento (0,09%). Il confronto omogeneo con `exp_avg` e' lo spostamento dal 544
(0,65% sul totale, 0,93% sul blocco del proiettore): con quel metro `exp_avg` del proiettore (1,0–1,3%) e' coerente con la divergenza
generale di 32 passi. In piu', una media mobile di gradienti rumorosi ha norma ridotta dalla cancellazione dei segni, quindi la
norma del tensore stesso come denominatore amplifica lo scarto relativo rispetto a quello dei gradienti. Questo sostiene H3 come
spiegazione del *perche' proprio l'1%*, ma non esclude H1/H2 da solo.

## 4. Misure sui checkpoint esistenti (continuo 576, riprese 576, fork 544) e previsioni

Per ciascuno dei 4 tensori del proiettore, e come confronto per tutti i 270 (e per CR):

1. **Norme e coseno**: `‖m_c‖, ‖m_r‖, ‖Δm‖, cos(m_c, m_r)`, decomposizione di `Δm` in componente radiale `((‖m_c‖−‖m_r‖))` e tangenziale.
   Rumore: `‖m_r‖/‖m_c‖ = 1 ± ~10⁻³`, coseno ≥ 0,9999, `Δm` quasi tutto tangenziale. Sistematico: rapporto delle norme lontano da 1
   (per esempio ≈ β1 o ≈ 1/(1−β1^k) per un passo saltato o una bias correction errata), componente radiale dominante, `Δm` ∝ `m`.
2. **Parte post-fork del momento**: `m_576 − β1³²·m_544` come denominatore alternativo (e `v_576 − β2³²·v_544`).
   Rumore: con questo metro il proiettore si allinea agli altri tensori e a CR. Sistematico: il proiettore resta un outlier netto.
3. **Coerenza del momento**: `‖m‖ / ‖√v‖` (o mediana di `|m|/√v`) per tensore. Rumore/H3: il proiettore ha coerenza minore degli
   altri tensori, e lo scarto relativo cresce con 1/coerenza lungo tutti i 270. Sistematico: nessuna relazione con la coerenza.
4. **Direzione dell'update di Adam** `u = m̂/(√v̂ + ε)`: `‖Δu‖/‖u‖` e coseno fra `u_c` e `u_r`. E' la grandezza che conta per i pesi
   (il weight decay e' identico). Rumore: coseno ≥ 0,9999 e `‖Δu‖/‖u‖` vicino a `‖Δm‖/‖m‖` (v quasi identico); impatto pratico
   misurato da `lr·‖Δu‖` rispetto a `lr·‖u‖` (lr ≈ 2,8·10⁻⁵ in warmup). Sistematico: coseno basso o segno/scala anomali.
5. **Coerenza con i pesi**: `Δθ_576` del blocco `contrastive_head` (0,93% dello spostamento dal 544, gia' nel JSON) contro la somma
   degli update. Rumore: ordine di grandezza simile a `‖Δu‖/‖u‖` e nessun salto. Sistematico: pesi quasi uguali con momenti molto
   diversi (stato dell'ottimizzatore incoerente con la traiettoria) o viceversa.
6. **Struttura dello scarto**: correlazione fra `Δm` dei tensori e scala comune (un fattore di clipping diverso produce `Δm`
   radiale e condiviso da tutti i tensori: `grad_clip_factor` vale 0,10–0,23 nei passi 545–576 e la sua differenza relativa
   arriva a ~4·10⁻⁴ al 550). Rumore: componente comune piccola, scarti indipendenti. Sistematico: un fattore di scala unico.

Nessuna di queste misure richiede nuovi job GPU: sono letture CPU dei tre checkpoint `.pt` di CN (e di CR per confronto) che
stanno su Leonardo. Limite [V da GPT, confermato dal codice]: i checkpoint non contengono i gradienti dei passi intermedi, quindi
queste letture misurano la **scala/rilevanza** dello scarto, non la sua **origine**.

## Osservazione a margine (fuori compito, da passare al compito 2) [V]

Dalle metriche `grad_norm` continuo vs CN_A: 545 → 2,2·10⁻¹², **546 → 0 esatto**, 547 → 2,2·10⁻⁵, 550 → 3,8·10⁻⁴. In CR: 545 → 3,4·10⁻¹³,
546 → 3,4·10⁻⁶. Il `total` loggato differisce gia' al 545 (9,8·10⁻⁸ in CN), mentre la norma del gradiente coincide a 10⁻¹²: il log
aggregato non e' un indicatore fedele (GPT nota scarti di log anche fra A e B identici). Il salto di sei ordini di grandezza fra 546 e 547
in CN e' compatibile con una differenza che nasce al 545–546 nella sola somma della norma o in pochi ulp e si amplifica dopo, ma da qui
non distingue DDP da altre fonti di non determinismo specifiche del processo (per esempio scelta di algoritmi cuBLAS/cuDNN).
