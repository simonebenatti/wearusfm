"""Ricalcolo indipendente per F13 (solo aritmetica, nessun dato).
Modello dei revisori: p=384 direzioni, segnale isotropo s_k=1/p (somma 1 = margine costante->oracolo in MSE normalizzata),
rumore effettivo sig^2*K con K_MSE=5, n_out=118000 finestre-giunto (20 utenti), n_in=n_out/2 (10 utenti).
Disegno X^T X = n I. Ridge in due parametrizzazioni:
  somma dei quadrati: min ||y-Xb||^2 + lam ||b||^2  -> shrink h = n/(n+lam); lam* = sig^2 K / s  (non dipende da n)
  loss media:         min (1/n)||y-Xb||^2 + alpha ||b||^2 -> lam = n*alpha; alpha* = lam*/n
Rischio in eccesso per direzione: (1-h)^2 s + h^2 v, v = sig^2 K / n.
Recupero del margine R = 1 - rischio totale (margine = 1)."""
from fractions import Fraction as F
p = 384; n_out = 118000; n_in = n_out // 2
for sig, K in ((2.5, 5.0), (3.0, 5.0)):
    s = 1.0 / p
    lam_star = sig * sig * K / s
    def R(n, lam):
        v = sig * sig * K / n
        h = n / (n + lam)
        return 1 - p * ((1 - h) ** 2 * s + h ** 2 * v)
    r_wls = 1 - p * sig * sig * K / n_out
    # somma dei quadrati: lam scelto a n_in (ottimo a n_in) e usato a n_out senza riscalare
    r_sum = R(n_out, lam_star)
    # loss media: alpha scelto a n_in (ottimo a n_in) e usato a n_out senza riscalare -> lam = n_out * lam_star / n_in
    alpha_in = lam_star / n_in
    r_mean_noresc = R(n_out, alpha_in * n_out)
    # loss media con alpha riscalato di n_in/n_out
    r_mean_resc = R(n_out, alpha_in * n_in / n_out * n_out)
    # ottimo vero a n_out (controllo)
    best = max(R(n_out, lam_star * f / 100) for f in range(10, 1000))
    print(f"sigma={sig} K={K}: WLS={r_wls:.4f}  somma_quadrati_non_riscalato={r_sum:.4f}  "
          f"loss_media_non_riscalato(lam x{alpha_in*n_out/lam_star:.1f})={r_mean_noresc:.4f}  "
          f"loss_media_riscalato={r_mean_resc:.4f}  ottimo_griglia={best:.4f}")
# soglia del cancello di P:95 e intervallo in cui una L0 neurale fallisce pur con la ridge sopra il 90%
sig, K = 2.5, 5.0
print("cancello P:95 = 0.90; con ridge 0.9077: L0 neurale in [0.8983, 0.90) -> 'ricetta non ammessa' con controllo diagnostico;"
      " L0 in [0.90, 0.9077] -> passa")
print(f"quota dell'intervallo WLS-ridge che fallisce: con ridge 0,9077 = {(0.90-0.8983)/(0.9077-0.8983):.2f}; "
      f"con ridge 0,9012 = {(0.90-0.8983)/(0.9012-0.8983):.2f}")
