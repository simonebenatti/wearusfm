"""Numeri dell'analisi scientifica di Claude sul confronto 300M contro 30M a 5.000 passi (linea 1, 06/11/2026).

    python -I -B claude_analysis_scaling_300m_5000_20261106_numbers.py <report.json>

Legge soltanto il report del confronto gia' raccolto e validato (ramo claude/scaling-300m-exec-20261106, commit 7fdb2e8,
results/infra/scaling_300m_20261104/final_stages_20261106/collected_outcome_20261106_1441/report.json, sha256 4c10e7bb...).
Prima parte: primari, secondari e posa come li ha scritti il comparatore congelato. Seconda parte: descrittive post hoc
(per soggetto e per classe) ricavate dai conteggi appaiati del report. Solo libreria standard, nessun calcolo nuovo sui dati.

Rettifica del 06/11/2026 (terza review di GPT, commit 5e15009): i conteggi per soggetto «meglio, uguale, peggio» si fanno
con frazioni esatte. Prima le medie erano in virgola mobile e il confronto con zero contava come differenze alcuni pareggi
(residui di circa 1e-16): su EPN-612 usciva 200/13/101 e 148/11/155 al posto di 199/15/100 e 148/13/153. Gli altri numeri
non cambiano.
"""
import json
import statistics
import sys
from fractions import Fraction

r = json.load(open(sys.argv[1]))
f = lambda x, n=3: ("%." + str(n) + "f") % x
ci = lambda c, n=3: "[%s, %s]" % (f(c[0], n), f(c[1], n))
print("quality:", r["quality"])
rd = r["reading"]
print("reading:", {k: rd[k] for k in ("reading_valid", "continuation_authorized", "primary_reading", "pose_practical_signal",
                                      "classification_noninferiority", "recommendation")})
print("pose_relative_point_reduction:", f(rd["pose_relative_point_reduction"], 5))
print("\nPRIMARI (300M-natural meno 30M-natural)")
for k in ("epn", "p1", "p2"):
    print("  %-4s delta_pp %s  ci95 %s" % (k, f(r[k]["delta_pp"]), ci(r[k]["ci95"])))
p = r["pose"]
print("  pose E_dyn: controllo %s  candidato %s  delta %s  ci95 %s" % (f(p["control_E_dyn"], 5), f(p["candidate_E_dyn"], 5),
                                                                      f(p["delta_E_dyn"], 5), ci(p["ci95"], 5)))
c = r["secondary"]["classification"]
cells = ["T100-G-30M-natural", "T100-G-300M-natural", "T100-G-300M-matched"]
names = ["natural_300m_minus_30m", "matched_300m_minus_30m", "natural_300m_minus_matched_300m"]
print("\nCLASSIFICAZIONE: balanced accuracy (%) e ci95 per cella; contrasti in punti percentuali")
for label, block in [(ds, c["datasets"][ds]) for ds in sorted(c["datasets"])] + [("task " + t, c["tasks"][t]) for t in ("p1", "p2")]:
    print("  " + label + ("  (%d soggetti, %d classi)" % (len(block["subjects"]), len(block["classes"])) if "subjects" in block else ""))
    for cell in cells:
        v = block["cells"][cell]
        print("     %-22s %s  ci95 %s" % (cell, f(100 * v["balanced_accuracy"], 2), ci(v["ci95_percent"], 2)))
    for n in names:
        v = block["contrasts"][n]
        print("     %-34s %s  ci95 %s" % (n, f(v["delta_pp"], 2), ci(v["ci95_pp"], 2)))
print("\nPOSA: rapporto d'errore per cella (piu' basso e' meglio) e contrasti")
for g in ("dynamic", "all", "static"):
    b = r["secondary"]["pose"][g]
    print("  gruppo %s (%d utenti)" % (g, len(b["users"])))
    for cell in cells:
        print("     %-22s %s  ci95 %s" % (cell, f(b["cell_ratio"][cell], 5), ci(b["cell_ratio_ci95"][cell], 5)))
    for n in names:
        v = b["contrasts"][n]
        better = sum(1 for u in v["per_user"].values() if u["delta_centered_sse"] < 0)
        print("     %-34s delta %s  ci95 %s  riduzione relativa %s  segnale2%%: %s  utenti con delta<0: %d/%d" % (
            n, f(v["delta_ratio"], 5), ci(v["ci95"], 5), f(v["relative_error_reduction"], 4), v["signal_2percent_and_ci_below_zero"],
            better, len(v["per_user"])))

# --- descrittive post hoc: per soggetto e per classe ---
c = r["secondary"]["classification"]["datasets"]
cells = ["T100-G-30M-natural", "T100-G-300M-natural", "T100-G-300M-matched"]
one = next(iter(c["epn612"]["paired_counts"][cells[0]]["subjects"].values()))
print("chiavi per soggetto:", sorted(one), "| lunghezze:", {k: len(v) if isinstance(v, list) else type(v).__name__ for k, v in one.items()})
keys = sorted(one)
def per_subject(ds, cell):
    out = {}
    for s, v in c[ds]["paired_counts"][cell]["subjects"].items():
        a, b = v[keys[0]], v[keys[1]]
        # individua quale chiave e' il denominatore: e' sempre >= dell'altra
        den, hit = (a, b) if all(x >= y for x, y in zip(a, b)) else (b, a)
        rec = [Fraction(h, d) for h, d in zip(hit, den) if d > 0]  # esatto: un pareggio resta un pareggio
        out[s] = (sum(rec) / len(rec), hit, den)
    return out
for ds in sorted(c):
    base, nat, mat = (per_subject(ds, cell) for cell in cells)
    for label, other in (("300M-natural meno 30M", nat), ("300M-matched meno 30M", mat)):
        d = [100 * (other[s][0] - base[s][0]) for s in base]
        print("%-12s %-22s soggetti %3d | meglio %3d, uguale %3d, peggio %3d | mediana %+.2f pp, media %+.2f pp" % (
            ds, label, len(d), sum(x > 0 for x in d), sum(x == 0 for x in d), sum(x < 0 for x in d),
            float(statistics.median(d)), float(statistics.mean(d))))
print("\nEPN612, recall per classe (%) sommando i conteggi dei soggetti")
classes = c["epn612"]["classes"]
for cell in cells:
    ps = per_subject("epn612", cell)
    hit = [sum(v[1][i] for v in ps.values()) for i in range(len(classes))]
    den = [sum(v[2][i] for v in ps.values()) for i in range(len(classes))]
    print("  %-22s %s" % (cell, "  ".join("%s %.1f" % (k, 100 * h / d) for k, h, d in zip(classes, hit, den))))
