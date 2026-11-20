# Stima d'ordine di grandezza di E_real (WLS NON regolarizzata, fit su 10 utenti) dai numeri pinnati della ricevuta storica
# LONG-FIT-G49046-pose-20261105.json (commit 2930d7d, sha256 b08e6e32...): percorso della ridge sui due fold interni 20/20.
# Ipotesi: eccesso classico della LS non regolarizzata E[MSE] = M* (1 + p/(n_eff - p - 1)); n_eff proporzionale agli utenti;
# quota dinamica dell'eccesso phi in [0,1; 0,5]; varianza centrata del target nelle finestre dinamiche ~0,525 per elemento
# (ricevuta, gruppo dynamic, means/vary, sui 15 val); E_dyn all'ottimo ~0,786 (ridge post storica sui val).
p = 384
mse_lowalpha = 0.7697254997148791   # alpha 0,01, fold 20/20
mse_opt = 0.6561153689170307        # alpha 1e5
vary = 0.5253446914952473
print("eccesso relativo a 20 utenti rispetto all'ottimo:", mse_lowalpha/mse_opt - 1)
for Mstar in (0.62, 0.64, 0.6561):
    ratio = mse_lowalpha/Mstar - 1           # = p/(n_eff-p-1)
    neff20 = p/ratio + p + 1
    neff10 = neff20/2
    mse10 = Mstar*(1 + p/(neff10 - p - 1))
    exc10 = mse10 - Mstar
    out = [f"M*={Mstar}: n_eff20={neff20:.0f} (per giunto, ~{neff20/953:.1f} per finestra), MSE_LS a 10 utenti={mse10:.3f} (eccesso {exc10:.3f})"]
    for phi in (0.1, 0.2, 0.35, 0.5):
        out.append(f"phi={phi}: E_real~{0.786 + exc10*phi/vary:.3f}")
    print(" | ".join(out))
