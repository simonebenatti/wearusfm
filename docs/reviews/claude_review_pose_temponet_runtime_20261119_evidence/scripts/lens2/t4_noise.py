import hashlib, json, numpy as np, torch
from pose_temponet import controls as C
meta = [{"user": "abc", "session": "s1", "start": 123, "n_patches": 160}, {"user": "def", "start": 9, "n_patches": 160}]
out = C.window_noise(20261118 + 2, meta, 160, 3)
# independent derivation for window 1, joint 2
key = json.dumps([20261120, meta[1], 2], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
eta = np.random.Generator(np.random.PCG64(seed)).standard_normal(160)
rho = np.exp(-0.25); e = np.empty(160); e[0] = eta[0]
for t in range(1, 160): e[t] = rho*e[t-1] + np.sqrt(1-rho**2)*eta[t]
print("noise independent match:", np.array_equal(e, out[1, :, 2]))
# stationarity of AR(1): long-run variance ~1
long = C.window_noise(5, [{"k": i} for i in range(200)], 160, 20)
print("var at t=0 / t=159:", round(long[:, 0].var(), 3), round(long[:, 159].var(), 3), "lag1 corr:", round(np.corrcoef(long[:, 10].ravel(), long[:, 11].ravel())[0, 1], 3), "rho", round(rho, 3))
# generator seeds and init distributions
rng = np.random.default_rng(1)
x = rng.normal(size=(40, 160, 384)).astype(np.float16)
sup = np.ones((40, 148, 20), dtype=bool)
users = [f"u{i:02d}" for i in range(40)]
meta = [{"user": u, "n_patches": 160} for u in users]
fit, held = np.arange(0, 40, 2), np.arange(1, 40, 2)
c0 = C.generate_components(x, meta, sup, fit, held, users, outer_fold=0, block_windows=16)
c1 = C.generate_components(x, meta, sup, fit, held, users, outer_fold=1, block_windows=16)
print("seeds:", c0["seed"], c1["seed"], "L std ~ 1/sqrt(384):", round(c0["generator_states"]["L"]["weight"].std().item()*np.sqrt(384), 3))
Ts = c0["generator_states"]["T"]
for k, v in Ts.items():
    if k.endswith("weight"):
        fan_in = v.shape[1]*v.shape[2]
        print(" T", k, tuple(v.shape), "std*sqrt(fan_in)=", round(v.std().item()*np.sqrt(fan_in), 3))
torch.manual_seed(20261118+1)
print("hidden active fraction:", [round(v, 3) for v in c0["hidden_active_fraction_fitting"]])
# normalization on fitting: weighted mean 0, std 1 on fit crop
for k in ("L", "T"):
    y = c0[k][fit, 6:154]
    print(k, "fit mean/std per joint (max abs dev):", float(np.abs(y.mean(axis=(0, 1))).max()), float(np.abs(y.std(axis=(0, 1)) - 1).max()))
