"""Ricalcola i numeri del controesempio di BL-1 (bozza:73-78) con le formule di
script_agenti/work_verifica_architettura-controlli/v_scan_lineare.py (modello in forma chiusa, nessun dato),
per entrambi i parametri di inflazione usati dal verificatore. Scrive solo in rilettura_riferimenti."""
OUT = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/rilettura_riferimenti/ricalcola_bl1.out.txt"
c_n = 0.9372; p = 384; n_out = 118000.0; n_in = n_out / 2; grid = [1, 1.5, 2, 2.5, 3]
lines = []
for Kd, Kt in ((2.5, 5.0), (3.5, 7.0)):
    for f, Er in ((0.5, 0.92), (0.4, 0.78), (0.4, 0.80)):
        cands = []
        for s in grid:
            N = s * s * c_n
            lin_in = (N + s * s * p * Kd / n_in) / (f + N)
            if abs(lin_in - Er) <= 0.05:
                cands.append((abs(lin_in - Er), s, lin_in))
        if not cands:
            lines.append("K=(%.1f,%.1f) f'=%.1f E_real=%.2f: nessuna sigma ammissibile -> non calibrato" % (Kd, Kt, f, Er))
            continue
        d, s, lin_in = min(cands)
        N = s * s * c_n
        oracle = N / (f + N)
        wls_out = (N + s * s * p * Kd / n_out) / (f + N)
        Rd = 1 - s * s * p * Kd / n_out / f
        Rm = 1 - s * s * p * Kt / n_out
        esito = "PASSA" if (Rd >= 0.9 and Rm >= 0.9) else "WLS esatta sotto 90%"
        lines.append("K=(%.1f,%.1f) f'=%.1f E_real=%.2f: sigma=%.1f, WLS inner %.5f, oracolo %.5f, WLS outer %.5f, "
                     "R_dyn %.4f, R_MSE %.4f -> %s" % (Kd, Kt, f, Er, s, lin_in, oracle, wls_out, Rd, Rm, esito))
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
