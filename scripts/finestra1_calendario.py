"""Calendario e costi della finestra 1 con N nodi sulla QoS normal (stima, aggiornata il 15/10/2026 con l'accordo GPT-Claude): run della
bozza di GPT (non firmata), passi dei manifest derivati (floor sul totale, `results/infra/derived_manifests_v1_1_build_cluster_20261015.json`),
s/passo misurati col prefetch, calibrazione C1-C4 dell'accordo in sequenza (C1 -> decisione -> C2 -> C3 -> C4) prima dei run che ne usano
il LR. Separa il tempo di calcolo dalle attese: `gap` = attesa in coda prima di ogni job da 24 h (una catena da k job ne paga k), che dipende
anche dal consumo degli altri 16 utenti del progetto; le attese delle decisioni di GPT fra le fasi della calibrazione sono a parte.

Uso: python scripts/finestra1_calendario.py [rapporto_dei_manifest_derivati.json]
"""

import heapq
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "infra" / "derived_manifests_v1_1_build_cluster_20261015.json"
WPE = {f: v["windows_per_epoch"] for f, v in json.loads(REPORT.read_text())["fractions"].items()}  # finestre per epoca, per frazione
SPS = {"30m": 0.70, "100m": 1.39, "300m": 2.77}  # s/passo, DDP 4 GPU, batch 32, prefetch (w1pf, w1gd)
OVER = 0.03  # avvio, valutazioni e checkpoint del trainer
JOB_H = 23.3  # ore utili di un job da 24 h (margine di 40 minuti)


def steps(frac: str, epochs: float, batch: int = 32) -> int:
    return math.floor(epochs * WPE[frac] / batch)  # un solo floor sul totale


def hours(preset, n):
    return n * SPS[preset] * (1 + OVER) / 3600


def runs():
    """(nome, ore di calcolo, prerequisito) dei run dopo la calibrazione."""
    out = []
    for i in range(28):  # ablation a 30M: riferimento + 13 bracci, 2 semi, D = 4 epoche
        out.append((f"abl30-{i}", hours("30m", steps("1", 4)), None))
    for s in range(2):  # pilot a 100M: tronco a 8 epoche e rami di decay da 0,8 / 1,6 / 3,2 epoche lunghi 0,2 / 0,4 / 0,8
        trunk = f"pilot100-s{s}"
        out.append((trunk, hours("100m", steps("1", 8)), None))
        for start, length in ((0.8, 0.2), (1.6, 0.4), (3.2, 0.8)):
            out.append((f"{trunk}-E{start / 0.8:g}", hours("100m", steps("1", length)), (trunk, hours("100m", steps("1", start)))))
    for s in range(2):  # asse dei soggetti a E = 4 (il 100M al 100% e' il ramo E4 del pilot)
        for f in ("0.125", "0.25", "0.5"):
            out.append((f"subj100-{f}-s{s}", hours("100m", steps(f, 4)), None))
        for f in ("0.125", "0.25", "0.5", "1"):
            out.append((f"subj300-{f}-s{s}", hours("300m", steps(f, 4)), None))
    return out


EXPO100 = [(f"expo100-s{s}", hours("100m", steps("1", 4)), None) for s in range(2)]  # 25% agli stessi passi del 100%
EXPO300 = [(f"expo300-s{s}", hours("300m", steps("1", 4)), None) for s in range(2)]
CALIBRATION = [  # fasi in sequenza, run di una fase in parallelo (accordo del 15/10)
    ("C1", [hours("30m", steps("1", 1))] * 3), ("C2", [hours("30m", steps("1", 1))] * 2),
    ("C3", [hours("100m", steps("1", 0.5))] * 2), ("C4", [hours("300m", steps("1", 0.25))])]
EVAL_NODE_HOURS = 50


def calibration_hours(nodes, gap):
    """Durata della calibrazione (solo calcolo e code, senza le attese delle decisioni) e ore di nodo."""
    total, node_h = 0.0, 0.0
    for _, hs in CALIBRATION:
        waves = math.ceil(len(hs) / nodes)
        total += waves * (max(hs) + gap)
        node_h += sum(hs)
    return total, node_h


def simulate(job_list, nodes, gap):
    free = [0.0] * nodes
    heapq.heapify(free)
    start, end = {}, {}
    for name, h, pre in sorted(job_list, key=lambda r: (isinstance(r[2], tuple), -r[1])):
        t0 = heapq.heappop(free)
        if isinstance(pre, tuple):
            t0 = max(t0, start[pre[0]] + pre[1] + gap)
        k = max(1, math.ceil(h / JOB_H))  # job da 24 h della catena, ciascuno con la sua attesa
        start[name] = t0 + gap
        end[name] = start[name] + h + (k - 1) * gap
        heapq.heappush(free, end[name])
    return max(end.values()), sum(r[1] for r in job_list)


def main():
    print(f"passi a D (4 epoche, batch 32): 100% {steps('1', 4)}, 50% {steps('0.5', 4)}, 25% {steps('0.25', 4)}, 12,5% {steps('0.125', 4)}")
    for label, extra in (("controllo solo 100M", EXPO100), ("controllo 100M e 300M", EXPO100 + EXPO300)):
        jobs = runs() + extra
        for n in (4, 8):
            for gap in (0.0, 1.0, 3.0):
                span, node_h = simulate(jobs, n, gap)
                cal, cal_h = calibration_hours(n, gap)
                tot_h = node_h + cal_h + EVAL_NODE_HOURS
                days = (cal + span) / 24 + EVAL_NODE_HOURS / n / 24
                print(f"{label:22s} N={n} attesa per job {gap:3.1f} h: GPU-ora {4 * tot_h:6.0f} (calibrazione {4 * cal_h:4.0f}); "
                      f"calibrazione {cal:4.1f} h + campagna {span / 24:4.1f} g + valutazione {EVAL_NODE_HOURS / n / 24:3.1f} g = {days:4.1f} giorni")


if __name__ == "__main__":
    main()
