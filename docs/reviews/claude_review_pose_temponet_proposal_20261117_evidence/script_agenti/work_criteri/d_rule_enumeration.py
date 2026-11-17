"""Punto (d): la regola a quattro esiti del protocollo (righe 95-98) e' esaustiva e senza sovrapposizioni?
Enumerazione di tutte le combinazioni di stima, limite inferiore L e limite superiore U rispetto a -2, 0, +2.
Le quattro etichette sono applicate alla lettera, ciascuna per conto suo (nessun ordine di precedenza)."""
import itertools
vals = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]   # rappresentanti: sotto -2, -2, fra -2 e 0, 0, fra 0 e 2, 2, sopra 2
def labels(est, L, U):
    out = []
    if est >= 2 and L > 0: out.append("B")            # beneficio pratico risolto
    if est <= -2 and U < 0: out.append("D")           # deterioramento pratico
    if U < 2: out.append("N")                         # beneficio di almeno 2% non sostenuto dall'intervallo
    if U >= 2 and not (est >= 2 and L > 0): out.append("I")   # inconclusivo: l'IC include >=2% ma il criterio di beneficio non passa
    return out
def old(est, L, U):   # practical_assessment di scripts/consolidation_long_pose.py:353-361 (IC95 allora)
    if est >= 2 and L > 0: return "practical_benefit"
    if est <= -2 and U < 0: return "practical_worsening"
    return "small" if abs(est) < 2 and (L > 0 or U < 0) else "inconclusive"
tot = {"coerenti": [0, 0, 0], "stima_fuori_IC": [0, 0, 0]}
overl, empty, table = {}, [], {}
for est, L, U in itertools.product(vals, vals, vals):
    if L > U: continue
    kind = "coerenti" if L <= est <= U else "stima_fuori_IC"
    lab = labels(est, L, U)
    tot[kind][0] += 1
    if len(lab) == 0: empty.append((est, L, U)); tot[kind][1] += 1
    if len(lab) > 1:
        tot[kind][2] += 1
        overl.setdefault((kind, "+".join(lab)), []).append((est, L, U))
    if kind == "coerenti":
        table.setdefault(("+".join(lab), old(est, L, U)), []).append((est, L, U))
for kind, (n, e, o) in tot.items():
    print(f"{kind}: combinazioni={n}, senza etichetta={e}, con piu' di una etichetta={o}")
print("combinazioni senza etichetta:", empty)
for (kind, lab), cases in sorted(overl.items()):
    print(f"sovrapposizione {lab} [{kind}]: {len(cases)} combinazioni, esempi (stima, L, U): {cases[:4]}")
print()
print("Confronto con la regola storica practical_assessment (solo combinazioni coerenti L<=stima<=U):")
for (new, o), cases in sorted(table.items()):
    print(f"  nuova={new:4s} storica={o:20s} n={len(cases):3d} esempio (stima, L, U)={cases[0]}")
