"""Verifica avversaria di T-02 (solo aritmetica sul modello di work_controlli/conti_temporale_dettaglio.py, nessun dato).
Il controllore conta solo i «non calibrato» (inner). Qui conto anche il secondo arresto in forma chiusa del temporale: la banda
outer di P:96 («non diagnostico»). Con la banda sulla LS fittata su y l'eccesso di stima si dimezza da inner (10 utenti) a outer
(20 utenti) e la banda si sposta in basso (controlli-4); con la banda su E_LS* (LS su y*) non si sposta.
Regola di scelta di P:92: distanza minima da 0,08, poi da E_real, poi a e sigma crescenti.
Ancora: WLS su y (protocollo attuale) o ridge su y (tabella della bozza, eccesso X*flin/(flin+X)).
Scrive solo in questa cartella."""
import itertools
OUT = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo2/verifica-tenuta/v_t02_arresti.out.txt"
c_n = 0.9372; p = 384; n_out = 118000.0; n_in = n_out / 2
grid = list(itertools.product((0.3, 0.5, 0.8), (0.3, 0.6, 0.9), (0.2, 0.4, 0.6), (2.0, 3.0, 5.0)))
a1 = [0.2, 0.4, 0.6, 0.8, 1.0]; a3 = a1 + [1.5, 2.0, 2.5, 3.0]
def fine(lo, hi, st=0.05):
    n = int(round((hi - lo) / st)); return [round(lo + st * i, 2) for i in range(n + 1)]
sg = {"P:90 {1..3}": [1, 1.5, 2, 2.5, 3], "fitta 1-3": fine(1, 3), "fitta 0,5-3": fine(0.5, 3), "fitta 0,5-6": fine(0.5, 6)}

def esito(fL, fT, r, Kd, Er, ag, ss, anchor, band):
    cand = []
    for a in ag:
        for s in ss:
            N = s * s * c_n; Xi = s * s * p * Kd / n_in; RT = a * a * r * fT; S = fL + a * a * fT; flin = S - RT
            Xa = Xi if anchor == "wls" else Xi * flin / (flin + Xi)
            Xb = Xi if band == "y" else 0.0
            Ecal = (N + RT + Xa) / (S + N); Els = (N + RT + Xb) / (S + N); Eor = N / (S + N); g = (Els - Eor) / Els
            if abs(Ecal - Er) <= 0.05 and 0.06 <= g <= 0.10:
                cand.append((abs(g - 0.08), abs(Ecal - Er), a, s, N, RT, Xi))
    if not cand:
        return "NC"
    _, _, a, s, N, RT, Xi = min(cand)
    Xo = (Xi / 2) if band == "y" else 0.0
    go = (RT + Xo) / (N + RT + Xo)
    return "OK" if 0.06 <= go <= 0.10 else "ND"

L = ["Arresti in forma chiusa del controllo temporale su 81 casi (griglia di ipotesi del revisore): NC = non calibrato,",
     "ND = calibrato ma banda outer di P:96 fuori intervallo («non diagnostico»). Formato NC+ND=totale per E_real 0,80/0,85/0,90/0,94."]
for anchor in ("wls", "ridge"):
    for an, ag in (("a<=1", a1), ("a<=3", a3)):
        for sn, ss in sg.items():
            for band in ("y", "ystar"):
                row = []
                for Er in (0.80, 0.85, 0.90, 0.94):
                    res = [esito(fL, fT, r, Kd, Er, ag, ss, anchor, band) for (fL, fT, r, Kd) in grid]
                    nc = res.count("NC"); nd = res.count("ND")
                    row.append("%d+%d=%d" % (nc, nd, nc + nd))
                L.append("ancora %-5s %-4s sigma %-12s banda su %-5s | %s" % (anchor, an, sn, band, " | ".join(row)))
open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
