# Forma chiusa (nessuna simulazione): varianza population dentro finestra di un AR(1) gaussiano stazionario, varianza 1.
import numpy as np
n, rho = 160, np.exp(-0.25)
k = np.arange(n)
S = rho ** np.abs(k[:, None] - k[None, :])          # covarianza
A = (np.eye(n) - np.ones((n, n)) / n) / n            # s2 = x' A x
mean_s2 = np.trace(A @ S)
var_s2 = 2 * np.trace(A @ S @ A @ S)
print(f"rho={rho:.6f}  E[s2]={mean_s2:.4f}  SD[s2] un giunto={np.sqrt(var_s2):.4f}  SD media su 20 giunti indipendenti={np.sqrt(var_s2/20):.4f}")
for sigma in (1, 1.5, 2, 2.5, 3):
    print(f"sigma={sigma}: media rumore in pose_var = {mean_s2*sigma**2:.3f}, SD fra finestre = {np.sqrt(var_s2/20)*sigma**2:.3f}")
