from pose_temponet import selection as S
def curve(lr, horizon, mse_fn, edyn_fn):
    exps = [0.0, 0.25, 0.5, 0.75] + [float(e) for e in range(1, horizon+1)]
    return S.CandidateCurve(lr, horizon, tuple(S.CurvePoint(e, mse_fn(e), edyn_fn(e)) for e in exps))
def mk(lr, best, horizon=150):
    return curve(lr, horizon, lambda e: (best if e == 10 else best + 1 + (e/1000)), lambda e: (e-10)**2 + 1)
others = [mk(3e-4, 5.0), mk(1e-3, 5.1), mk(3e-3, 5.2), mk(3e-5, 6.0)]
def s2(peak):
    c = curve(1e-4, 150, lambda e: (e-20)**2+0.5, lambda e: (1.02 if e == 20 else ((e-peak)**2/1e6 + 1.0)))
    return S.assessment([c] + others)
for peak in (25, 26):
    r = s2(peak)
    t = r["objective_tradeoff"]
    print("S2 peak", peak, "selected", r["selected"]["lr"], r["selected"]["exposure"], "flag", t["flag"], "dist", t["exposure_distance"], "regret", round(t["dynamic_relative_regret"], 6))
# regret just under 2%
c = curve(1e-4, 150, lambda e: (e-20)**2+0.5, lambda e: (1.0199 if e == 20 else ((e-40)**2/1e6 + 1.0)))
t = S.assessment([c]+others)["objective_tradeoff"]; print("regret 1.99%:", t["flag"], t["dynamic_relative_regret"])
# zero minimum E_dyn
c = curve(1e-4, 150, lambda e: (e-20)**2+0.5, lambda e: (0.0 if e == 40 else 0.5))
t = S.assessment([c]+others)["objective_tradeoff"]; print("zero min:", t["flag"], t["regret_infinite"], t["dynamic_relative_regret"])
# both zero
c = curve(1e-4, 150, lambda e: (e-20)**2+0.5, lambda e: 0.0)
t = S.assessment([c]+others)["objective_tradeoff"]; print("both zero:", t["flag"], t["regret_infinite"], t["dynamic_relative_regret"], t["dynamic_minimum"]["exposure"])
# last five epochs at max horizon
c = curve(1e-4, 300, lambda e: (e-297)**2+0.5, lambda e: 1.0 + e/1e3)
print("last five mse@297:", S.assessment([c]+others)["optimizer_alarms"])
c = curve(1e-4, 300, lambda e: (e-295)**2+0.5, lambda e: 1.0 + e/1e3)
print("mse@295:", S.assessment([c]+others)["optimizer_alarms"])
c = curve(1e-4, 300, lambda e: (e-100)**2+0.5, lambda e: (e-296)**2 + 1)
print("edyn@296:", S.assessment([c]+others)["optimizer_alarms"])
# step 0 excluded from both selection and dynamic minimum
c = curve(1e-4, 150, lambda e: 0.0 if e == 0 else (e-20)**2+0.5, lambda e: 0.0 if e == 0 else (e-20)**2+1)
r = S.assessment([c]+others); print("step0 excluded:", r["selected"]["exposure"], r["objective_tradeoff"]["dynamic_minimum"]["exposure"])
