"""Effect of observer seconds on the real measured_allocation (fbfde797), synthetic 576-update benchmark.

Assumptions (declared, not measured): normal update 1.10 s, diagnostic update 1.90 s
(grad_every=100, grad_steps 256/512, dense 513..576), 50 s outside updates (startup,
evals, saves). Observer cost X split equally between C0 updates 128 and 256.
"""
import contrastive_benchmark_readout as B
from wearusfm.training.contrastive_objectives import ContrastiveConfig

cc = ContrastiveConfig(arm="C0", nce_weight=1., rnc_weight=1., grad_every=100, grad_dense_start=513, grad_dense_end=576)


def series(extra=None):
    extra = extra or {}
    rows = []
    for step in range(1, 577):
        diag = cc.wants_gradients(step - 1)
        r = {"step": step, "t_step_s": (1.90 if diag else 1.10) + extra.get(step, 0.)}
        if diag:
            r["contrast_global_grad_base"] = 1.
        rows.append(r)
    return rows


def project(rows):
    return B.measured_allocation({"elapsed_s": sum(r["t_step_s"] for r in rows) + 50.}, rows)


base = project(series())
print("diagnostic updates in 1..576:", sum(cc.wants_gradients(s - 1) for s in range(1, 577)),
      "; 128 diagnostic?", cc.wants_gradients(127), "; 256 diagnostic?", cc.wants_gradients(255))
print(f"no observer: elapsed {base['observed_end_to_end_s']:.1f}s per_update {base['conservative_update_s']:.4f}s "
      f"phases {base['phase_allocation_seconds']} GPUh {base['allocated_gpu_hours']:.3f}")
for X in (10., 20., 30., 40., 60.):
    got = project(series({128: X / 2, 256: X / 2}))
    print(f"C0 observer {X:4.0f}s: per_update {got['conservative_update_s']:.4f}s (+{100*(got['conservative_update_s']/base['conservative_update_s']-1):.2f}%) "
          f"phases {got['phase_allocation_seconds']} GPUh {got['allocated_gpu_hours']:.3f} (+{got['allocated_gpu_hours']-base['allocated_gpu_hours']:.3f})")
trace = {545: 1.0, **{s: .05 for s in range(546, 577)}}
got = project(series(trace))
print(f"CN/CR trace ~2.6s: per_update {got['conservative_update_s']:.4f}s phases {got['phase_allocation_seconds']} "
      f"GPUh {got['allocated_gpu_hours']:.3f} (+{got['allocated_gpu_hours']-base['allocated_gpu_hours']:.3f})")
which = "elapsed/576" if base["conservative_update_s"] == base["observed_end_to_end_s"] / 576 else "mean(normal)+premium"
print("branch selected by max():", which, "; mean(normal)+0.02*premium =",
      round(base["normal_update_mean_s"] + max(0., base["diagnostic_update_mean_s"] - base["normal_update_mean_s"]) * .02, 4))
