"""Lente applicazione (controllo 2). Stesso modello in forma chiusa di v_scan_lineare.py e bl1_griglia_fine.py (nessun dato).
Controlla bozza_controllo2.md:102-104: con sigma continua la WLS esatta smette di passare fra ~0,89 e 0,94; la bozza dice che
«sotto quella soglia» l'esito dipende dalla griglia. Qui si cercano anche i casi SOPRA la soglia in cui la griglia attuale passa."""
OUT = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/applicazione/soglia_griglia.out.txt"
c_n = 0.9372; p = 384; n_out = 118000.0; n_in = n_out / 2
grid = [1, 1.5, 2, 2.5, 3]
def inner(f, v, Kd):
    N = v * c_n; return (N + v * p * Kd / n_in) / (f + N)
def rec(f, v, Kd, Kt):
    return 1 - v * p * Kd / n_out / f, 1 - v * p * Kt / n_out
def v_cont(f, Er, Kd):
    a = c_n + p * Kd / n_in
    return Er * f / (a - Er * c_n)
def passa(Rd, Rm): return min(Rd, Rm) >= 0.9
L = []
for Kd, Kt in ((2.5, 5.0), (3.5, 7.0)):
    for f in (0.3, 0.4, 0.5, 0.6):
        # soglia con sigma continua
        Er = 0.80; thr = None
        while Er < 1.0:
            Rd, Rm = rec(f, v_cont(f, Er, Kd), Kd, Kt)
            if not passa(Rd, Rm): thr = Er; break
            Er += 0.0005
        sopra_passa = []; sotto_fallisce = []
        Er = 0.80
        while Er <= 1.0001:
            c = [(abs(inner(f, s * s, Kd) - Er), s) for s in grid if abs(inner(f, s * s, Kd) - Er) <= 0.05]
            if c:
                d, s = min(c); Rd, Rm = rec(f, s * s, Kd, Kt)
                g = passa(Rd, Rm)
                if Er > thr and g: sopra_passa.append("%.3f(s=%.1f,R=%.4f)" % (Er, s, min(Rd, Rm)))
                if Er < thr and not g: sotto_fallisce.append("%.3f(s=%.1f)" % (Er, s))
            Er = round(Er + 0.005, 4)
        L.append("K=(%.1f,%.1f) f'=%.1f soglia continua %.4f | griglia attuale PASSA sopra la soglia: %s" % (
            Kd, Kt, f, thr, ", ".join(sopra_passa) if sopra_passa else "nessuno"))
        L.append("      griglia attuale FALLISCE sotto la soglia (E_real >= 0,80): %s" % (
            ", ".join(sotto_fallisce[:8]) + (" ..." if len(sotto_fallisce) > 8 else "") if sotto_fallisce else "nessuno"))
# caso puntuale citato
f, Er, Kd, Kt = 0.5, 0.92, 3.5, 7.0
c = [(abs(inner(f, s * s, Kd) - Er), s) for s in grid if abs(inner(f, s * s, Kd) - Er) <= 0.05]
d, s = min(c); Rd, Rm = rec(f, s * s, Kd, Kt)
Rdc, Rmc = rec(f, v_cont(f, Er, Kd), Kd, Kt)
L.append("")
L.append("Caso: K=(3,5;7) f'=0,5 E_real=0,92 -> griglia attuale sigma=%.1f inner=%.4f R=%.4f (%s); sigma continua R=%.4f (%s)" % (
    s, inner(f, s * s, Kd), min(Rd, Rm), "PASSA" if passa(Rd, Rm) else "FALLISCE", min(Rdc, Rmc), "PASSA" if passa(Rdc, Rmc) else "FALLISCE"))
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
