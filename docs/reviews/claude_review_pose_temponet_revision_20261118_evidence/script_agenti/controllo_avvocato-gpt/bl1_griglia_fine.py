"""Avvocato di GPT, BL-1. Stesso modello in forma chiusa del verificatore (v_scan_lineare.py, nessun dato):
controesempio della bozza (f'=0,5, E_real=0,92) con la griglia di sigma attuale e con sigma continua,
cioe' dopo la correzione (d) primo punto della stessa bozza. Scrive solo nella cartella avvocato-gpt."""
OUT = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo/avvocato-gpt/bl1_griglia_fine.out.txt"
c_n = 0.9372; p = 384; n_out = 118000.0; n_in = n_out / 2
grid = [1, 1.5, 2, 2.5, 3]
L = []
def inner(f, v, Kd):  # v = sigma^2
    N = v * c_n; return (N + v * p * Kd / n_in) / (f + N)
def outer(f, v, Kd, Kt):
    N = v * c_n; Eor = N / (f + N); Ewls = (N + v * p * Kd / n_out) / (f + N)
    return Eor, Ewls, 1 - v * p * Kd / n_out / f, 1 - v * p * Kt / n_out
def v_cont(f, Er, Kd):  # sigma^2 tale che inner == Er (inner cresce con v)
    a = c_n + p * Kd / n_in
    return Er * f / (a - Er * c_n)
for Kd, Kt in ((2.5, 5.0), (3.5, 7.0)):
    for f, Er in ((0.5, 0.92), (0.3, 0.92), (0.6, 0.92), (0.5, 0.90), (0.5, 0.94)):
        # griglia attuale
        c = [(abs(inner(f, s * s, Kd) - Er), s) for s in grid if abs(inner(f, s * s, Kd) - Er) <= 0.05]
        if c:
            d, s = min(c); Eor, Ew, Rd, Rm = outer(f, s * s, Kd, Kt)
            g = "sigma=%.1f inner=%.4f R_dyn=%.4f R_MSE=%.4f %s" % (s, inner(f, s * s, Kd), Rd, Rm, "PASSA" if min(Rd, Rm) >= 0.9 else "FALLISCE")
        else:
            g = "non calibrato"
        v = v_cont(f, Er, Kd); Eor, Ew, Rd, Rm = outer(f, v, Kd, Kt)
        cc = "sigma=%.3f inner=%.4f R_dyn=%.4f R_MSE=%.4f %s" % (v ** 0.5, inner(f, v, Kd), Rd, Rm, "PASSA" if min(Rd, Rm) >= 0.9 else "FALLISCE")
        L.append("K=(%.1f,%.1f) f'=%.1f E_real=%.2f | griglia attuale: %s | sigma continua: %s" % (Kd, Kt, f, Er, g, cc))
# soglia di E_real oltre la quale la WLS esatta non passa, con sigma continua
L.append("")
for Kd, Kt in ((2.5, 5.0), (3.5, 7.0)):
    for f in (0.3, 0.4, 0.5, 0.6):
        Er = 0.70
        while Er < 1.2:
            v = v_cont(f, Er, Kd)
            if v <= 0: break
            Eor, Ew, Rd, Rm = outer(f, v, Kd, Kt)
            if min(Rd, Rm) < 0.9: break
            Er += 0.0005
        L.append("sigma continua K=(%.1f,%.1f) f'=%.1f: la WLS esatta smette di passare a E_real ~ %.4f" % (Kd, Kt, f, Er))
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))

# Griglia fitta a passi di 0,1 (la proposta di controlli-3: sigma da 0,5 a 4), stessa regola di scelta di P:92
fine = [round(0.5 + 0.1 * i, 1) for i in range(36)]
L2 = [""]
for Kd, Kt in ((2.5, 5.0), (3.5, 7.0)):
    for f, Er in ((0.5, 0.92), (0.5, 0.94)):
        c = [(abs(inner(f, s * s, Kd) - Er), s) for s in fine if abs(inner(f, s * s, Kd) - Er) <= 0.05]
        d, s = min(c); Eor, Ew, Rd, Rm = outer(f, s * s, Kd, Kt)
        L2.append("griglia 0,1 K=(%.1f,%.1f) f'=%.1f E_real=%.2f: sigma=%.1f inner=%.4f R_dyn=%.4f R_MSE=%.4f %s" % (
            Kd, Kt, f, Er, s, inner(f, s * s, Kd), Rd, Rm, "PASSA" if min(Rd, Rm) >= 0.9 else "FALLISCE"))
open(OUT, "a").write("\n".join(L2) + "\n"); print("\n".join(L2))
