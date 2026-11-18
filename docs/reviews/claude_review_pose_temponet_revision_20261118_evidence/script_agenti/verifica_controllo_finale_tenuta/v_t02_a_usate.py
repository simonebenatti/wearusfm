"""Verifica avversaria di T-02 (ii): con sigma <= 3 le a oltre 1,5 sono davvero inutili? Modello di
work_controlli/conti_temporale_dettaglio.py, ancora ridge su y, banda su E_LS* (tabella della bozza), griglia di ipotesi del
revisore (81 casi). Conto, per ogni E_real, quali a hanno almeno una coppia ammissibile e quale a sceglie la regola di P:92.
Solo aritmetica. Scrive solo in questa cartella."""
import itertools
from collections import Counter
OUT = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/verifica-tenuta/v_t02_a_usate.out.txt"
c_n = 0.9372; p = 384; n_out = 118000.0; n_in = n_out / 2
grid = list(itertools.product((0.3, 0.5, 0.8), (0.3, 0.6, 0.9), (0.2, 0.4, 0.6), (2.0, 3.0, 5.0)))
a3 = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
def fine(lo, hi, st=0.05):
    n = int(round((hi - lo) / st)); return [round(lo + st * i, 2) for i in range(n + 1)]
L = []
for smax in (3.0, 6.0):
    ss = fine(0.5, smax)
    for Er in (0.85, 0.90, 0.94):
        usable = Counter(); chosen = Counter(); only_gt15 = 0
        for (fL, fT, r, Kd) in grid:
            cand = []
            for a in a3:
                for s in ss:
                    N = s * s * c_n; Xi = s * s * p * Kd / n_in; RT = a * a * r * fT; S = fL + a * a * fT; flin = S - RT
                    Xa = Xi * flin / (flin + Xi)
                    Ecal = (N + RT + Xa) / (S + N); Els = (N + RT) / (S + N); Eor = N / (S + N); g = (Els - Eor) / Els
                    if abs(Ecal - Er) <= 0.05 and 0.06 <= g <= 0.10:
                        cand.append((abs(g - 0.08), abs(Ecal - Er), a, s))
            aset = sorted(set(c[2] for c in cand))
            for a in aset: usable[a] += 1
            if cand:
                chosen[min(cand)[2]] += 1
                if all(a > 1.5 for a in aset): only_gt15 += 1
        L.append("sigma 0,5-%.0f E_real %.2f: casi con almeno una coppia per a %s; a scelta %s; casi calibrabili solo con a>1,5: %d"
                 % (smax, Er, dict(sorted(usable.items())), dict(sorted(chosen.items())), only_gt15))
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
