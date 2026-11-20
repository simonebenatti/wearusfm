"""P1 (F13): nel modello dei revisori (spettro isotropo, sigma 2,5, K_MSE 5) confronta la ridge con lambda scelto a 10 utenti
(n_in = n_out/2) e portato a 20 utenti senza riscalare, con penalita' sulla somma dei quadrati e con loss media.
Stesse formule di EV/script_agenti/rilettura_fedelta/conti_rilettura.py. Solo aritmetica."""
n_out = 118000.0; n_in = n_out / 2; p = 384
spec = [1 / p] * p                      # segnale per direzione (isotropo), somma 1
sig, K = 2.5, 5.0
def risk(n, lam_sum):
    # d_k = sig_eff2 / v proporzionale a n; varianza OLS v = sig^2 K / n; shrink = d/(d + lam) = 1/(1 + lam*v/sig_eff2)
    v = sig * sig * K / n
    tot = 0.0
    for s in spec:
        sh = 1.0 / (1.0 + lam_sum * v / (sig * sig * K))
        tot += s * (1 - sh) ** 2 + v * sh ** 2
    return tot
# ottimo con penalita' sulla somma: lam* = sig_eff2 / s (indipendente da n)
lam_star = (sig * sig * K) / spec[0]
r_ols = 1 - p * sig * sig * K / n_out
r_sum = 1 - risk(n_out, lam_star)                 # lambda scelto a n_in, usato a n_out: stesso valore
alpha_in = lam_star / n_in                        # loss media: alpha* a n_in
r_mean = 1 - risk(n_out, alpha_in * n_out)        # usato a n_out senza riscalare: lambda = 2 lam*
print(f"R_MSE WLS/OLS = {r_ols:.4f}")
print(f"ridge, penalita' sulla somma, lambda degli inner non riscalato = {r_sum:.4f}")
print(f"ridge, loss media, alpha degli inner non riscalato (lambda x{alpha_in*n_out/lam_star:.1f}) = {r_mean:.4f}")
