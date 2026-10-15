"""Calendario della finestra 1 con N nodi (stima, 15/10/2026): run della bozza di GPT (non firmata), passi per epoca dal corpus
(results/infra/nested_manifests_v1_1_20261015.json), s/passo misurati col prefetch (w1pf, w1gd); calibrazione col segnaposto di
docs/finestra1_fattibilita.md. Le pause di coda fra i job di una catena non sono contate. Uso: python scripts/finestra1_calendario.py"""
import heapq

EPOCH = {1.0: 24528, 0.5: 12478, 0.25: 6403, 0.125: 3412}  # passi per epoca col batch 32 (nested_manifests_v1_1_20261015.json)
SPS = {"30m": 0.70, "100m": 1.39, "300m": 2.77}            # s/passo, DDP 4 GPU, batch 32 (w1pf, w1gd)
OVER = 0.03  # avvio, valutazioni e checkpoint (~1 min ogni 24 h di job, valutazioni periodiche): margine del 3%


def hours(preset, steps):
    return steps * SPS[preset] * (1 + OVER) / 3600


runs = []  # (nome, ore, prerequisito: None | "calib" | (tronco, ore del tronco a cui il ramo puo' partire))
# calibrazione (segnaposto della fattibilita'): 30M 3 LR x 2 batch x 1 epoca; 100M 2 x 2 x 0,5 epoche (batch 64 = meta' dei passi, ~1,95x s/passo)
for i in range(3):
    runs.append((f"cal30-b32-{i}", hours("30m", EPOCH[1.0]), None))
    runs.append((f"cal30-b64-{i}", EPOCH[1.0] / 2 * 1.37 * (1 + OVER) / 3600, None))
for i in range(2):
    runs.append((f"cal100-b32-{i}", hours("100m", EPOCH[1.0] / 2), None))
    runs.append((f"cal100-b64-{i}", EPOCH[1.0] / 4 * 2.69 * (1 + OVER) / 3600, None))
CAL = [r[0] for r in runs]
# ablation a 30M: riferimento + 13 bracci, 2 semi, D = 4 epoche
for i in range(28):
    runs.append((f"abl30-{i}", hours("30m", 4 * EPOCH[1.0]), "calib"))
# pilot a 100M: tronco a 8 epoche e rami di decay da 0,8 / 1,6 / 3,2 epoche lunghi 0,2 / 0,4 / 0,8 epoche, 2 semi
for s in range(2):
    trunk = f"pilot100-s{s}"
    runs.append((trunk, hours("100m", 8 * EPOCH[1.0]), "calib"))
    for start, length in ((0.8, 0.2), (1.6, 0.4), (3.2, 0.8)):
        runs.append((f"{trunk}-E{start / 0.8:g}", hours("100m", length * EPOCH[1.0]), (trunk, hours("100m", start * EPOCH[1.0]))))
# asse dei soggetti, E = 4, 2 semi; il 100M al 100% e' il ramo E4 del pilot
for s in range(2):
    for f in (0.125, 0.25, 0.5):
        runs.append((f"subj100-{f}-s{s}", hours("100m", 4 * EPOCH[f]), "calib"))
    for f in (0.125, 0.25, 0.5, 1.0):
        runs.append((f"subj300-{f}-s{s}", hours("300m", 4 * EPOCH[f]), "calib"))
# controllo d'esposizione: 25% agli stessi passi del 100% (4 epoche piene), 2 semi; solo 100M (variante: anche 300M)
EXPO = [(f"expo100-s{s}", hours("100m", 4 * EPOCH[1.0]), "calib") for s in range(2)]
EXPO300 = [(f"expo300-s{s}", hours("300m", 4 * EPOCH[1.0]), "calib") for s in range(2)]
EVAL_NODE_HOURS = 50  # ~200 GPU-ora di valutazione


def simulate(job_list, nodes):
    """Lista: i run piu' lunghi per primi; un run occupa un nodo per tutta la sua durata (le pause di coda fra i job non sono contate)."""
    free = [0.0] * nodes
    heapq.heapify(free)
    end = {}
    pending = sorted(job_list, key=lambda r: -r[1])
    cal_end = None
    order = [r for r in pending if r[2] is None] + [r for r in pending if r[2] == "calib"] + [r for r in pending if isinstance(r[2], tuple)]
    starts = {}
    for name, h, pre in order:
        t0 = heapq.heappop(free)
        if pre == "calib":
            cal_end = cal_end if cal_end is not None else max(end[c] for c in CAL)
            t0 = max(t0, cal_end)
        elif isinstance(pre, tuple):
            t0 = max(t0, starts[pre[0]] + pre[1])
        starts[name] = t0
        end[name] = t0 + h
        heapq.heappush(free, end[name])
    return max(end.values()), sum(r[1] for r in job_list), max(end[c] for c in CAL)


for label, extra in (("controllo solo 100M", EXPO), ("controllo 100M e 300M", EXPO + EXPO300)):
    jobs = runs + extra
    for n in (4, 8, 12):
        span, total, cal = simulate(jobs, n)
        days_train = span / 24
        days_eval = EVAL_NODE_HOURS / n / 24
        print(f"{label:24s} N={n:2d}: ore di nodo {total + EVAL_NODE_HOURS:6.0f} (GPU-ora {4 * (total + EVAL_NODE_HOURS):6.0f}), "
              f"calibrazione {cal:4.1f} h, training {days_train:4.1f} giorni, + valutazione {days_eval:3.1f} -> {days_train + days_eval:4.1f} giorni")
longest = max(r[1] for r in runs)
print("run piu' lungo (ore):", round(longest, 1), [r[0] for r in runs if r[1] == longest][:2])
