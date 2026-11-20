import math
from pose_temponet import selection as S

# checkpoint schedule production-like
print("sched 30/31:", S.checkpoint_schedule([30, 31], 2))
print("sched 1/1:", S.checkpoint_schedule([1, 1], 2))
print("sched 2/3:", S.checkpoint_schedule([2, 3], 2))
print("sched 3/2 after 1:", S.checkpoint_schedule([3, 2], 3, after_exposure=1))

def curve(lr, horizon, mse_fn, edyn_fn):
    exps = [0.0, 0.25, 0.5, 0.75] + [float(e) for e in range(1, horizon+1)]
    pts = tuple(S.CurvePoint(e, mse_fn(e), edyn_fn(e)) for e in exps)
    return S.CandidateCurve(lr, horizon, pts)

# extension trigger at 146 (MSE) and at 145 (none)
c = curve(1e-3, 150, lambda e: (e-146)**2 + 1, lambda e: (e-10)**2 + 1)
print("ext trigger mse at 146:", S.needs_epoch_extension(c))
c = curve(1e-3, 150, lambda e: (e-145)**2 + 1, lambda e: (e-10)**2 + 1)
print("ext trigger mse at 145:", S.needs_epoch_extension(c))
c = curve(1e-3, 150, lambda e: (e-10)**2 + 1, lambda e: (e-150)**2 + 1)
print("ext trigger edyn at 150:", S.needs_epoch_extension(c))

# lower-edge alarm: gain exactly 0.005 -> no alarm, just above -> alarm
base = 1.0
def mk(lr, best, horizon=150, edyn_best_epoch=10):
    return curve(lr, horizon, lambda e: (best if e == 10 else best + 1 + (e/1000)), lambda e: (e-edyn_best_epoch)**2 + 1)
cands = [mk(3e-5, 0.995), mk(1e-4, 1.0), mk(3e-4, 1.2), mk(1e-3, 1.3), mk(3e-3, 1.4)]
a = S.assessment(cands)
print("lower gain=", a["lower_edge_relative_mse_gain"], a["optimizer_alarms"])
cands = [mk(3e-5, 0.99499), mk(1e-4, 1.0), mk(3e-4, 1.2), mk(1e-3, 1.3), mk(3e-3, 1.4)]
a = S.assessment(cands)
print("lower gain=", a["lower_edge_relative_mse_gain"], a["optimizer_alarms"])
# zero neighbor
cands = [mk(3e-5, 0.0), mk(1e-4, 0.0), mk(3e-4, 1.2), mk(1e-3, 1.3), mk(3e-3, 1.4)]
try:
    a = S.assessment(cands)
    print("zero/zero:", a["selected"], a["lower_edge_relative_mse_gain"], a["optimizer_alarms"])
except Exception as e:
    print("zero/zero error", type(e).__name__, e)

# next LR logic
init = [mk(1e-4, 0.9), mk(3e-4, 1.0), mk(1e-3, 1.1), mk(3e-3, 1.2)]
print("next lower:", S.next_learning_rate(init))
init2 = init + [mk(3e-5, 0.8)]
print("after lower win:", S.next_learning_rate(init2))
init3 = [mk(1e-4, 1.3), mk(3e-4, 1.0), mk(1e-3, 1.1), mk(3e-3, 0.9)]
print("next upper:", S.next_learning_rate(init3))
init4 = init3 + [mk(1e-2, 0.8)]
a = S.assessment(init4); print("upper after ext:", a["selected"]["lr"], a["optimizer_alarms"])

# ties: exact mse tie across LRs -> earlier exposure, then smaller lr
t = [curve(1e-4, 150, lambda e: 1.0 if e in (20., 30.) else 2.0, lambda e: 1.0),
     curve(3e-4, 150, lambda e: 1.0 if e == 20. else 2.0, lambda e: 1.0),
     curve(1e-3, 150, lambda e: 1.0 if e == 15. else 2.0, lambda e: 1.0),
     curve(3e-3, 150, lambda e: 2.0, lambda e: 1.0)]
print("tie:", S.choose(t))

# first quarter alarm
q = [curve(1e-4, 150, lambda e: 0.5 if e == 0.25 else 2.0, lambda e: 1.0)] + init[1:]
print("first quarter:", S.assessment(q + [mk(3e-5, 1.5)])["optimizer_alarms"])

# S-2: distance >5 and regret >=2%
s2 = [curve(1e-4, 150, lambda e: (e-20)**2+1, lambda e: (1.02 if e == 20 else ((e-26)**2/1e6 + 1.0)))] + init[1:] + [mk(3e-5, 1.5)]
r = S.assessment(s2)["objective_tradeoff"]; print("S2 26 vs 20:", r["flag"], r["exposure_distance"], r["dynamic_relative_regret"])
s2 = [curve(1e-4, 150, lambda e: (e-20)**2+1, lambda e: (1.02 if e == 20 else ((e-25)**2/1e6 + 1.0)))] + init[1:] + [mk(3e-5, 1.5)]
r = S.assessment(s2)["objective_tradeoff"]; print("S2 25 vs 20:", r["flag"], r["exposure_distance"], r["dynamic_relative_regret"])
# S-2 zero minimum
s2 = [curve(1e-4, 150, lambda e: (e-20)**2+1, lambda e: (0.0 if e == 40 else 0.5))] + init[1:] + [mk(3e-5, 1.5)]
r = S.assessment(s2)["objective_tradeoff"]; print("S2 zero min:", r["flag"], r["regret_infinite"], r["dynamic_relative_regret"])
# last-five at max horizon
lf = [curve(1e-4, 300, lambda e: (e-297)**2+1, lambda e: 1.0 + e/1e3)] + init[1:] + [mk(3e-5, 1e9)]
print("last five:", S.assessment(lf)["optimizer_alarms"])
lf = [curve(1e-4, 300, lambda e: (e-295)**2+1, lambda e: 1.0 + e/1e3)] + init[1:] + [mk(3e-5, 1e9)]
print("not last five (295):", S.assessment(lf)["optimizer_alarms"])
