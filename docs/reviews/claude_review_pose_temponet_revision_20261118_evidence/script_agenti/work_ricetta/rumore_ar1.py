"""Stima in forma chiusa: contributo del rumore AR(1) del controllo a pose_var di una finestra (n tempi validi, 20 giunti)."""
import math
rho = math.exp(-0.25)
for n in (148, 160):
    k = (1 + rho) / (1 - rho)            # tempo di correlazione integrato (fattore)
    mean_fac = 1 - k / n                 # E[s^2]/sigma^2 per la varianza campionaria population
    rel_sd_joint = math.sqrt(2 * (1 + rho**2) / (1 - rho**2) / n)   # SD relativa di s^2 per giunto (n grande)
    rel_sd_win = rel_sd_joint / math.sqrt(20)                       # media su 20 giunti indipendenti
    print(f"n={n}: rho={rho:.4f}, E[s2]/sigma2={mean_fac:.3f}, SD rel per giunto={rel_sd_joint:.3f}, SD rel della media su 20 giunti={rel_sd_win:.3f}")
    for s in (1, 1.5, 2, 2.5, 3):
        m = mean_fac * s * s; sd = rel_sd_win * s * s
        print(f"   sigma={s}: parte di rumore in pose_var ~ {m:.2f} +- {sd:.3f} (fra finestre); soglia storica dei target reali 0.257")
