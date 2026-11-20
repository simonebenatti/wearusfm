"""Eterogeneita' fra utenti osservata in contrasti storici gia' pubblicati (finestre dinamiche, stessi 15 utenti val).
Serve solo a calibrare la simulazione. Solo lettura di JSON gia' committati."""
import json, sys
import numpy as np
SNAP, W = sys.argv[1], sys.argv[2]
SEED, N_BOOT = 20261102, 2000
def per_user(cell_analysis, group="dynamic"):
    pu = cell_analysis["groups"][group]["centered"]["per_user"]
    us = sorted(pu)
    return us, np.array([pu[u]["sums"]["centered"] for u in us]), np.array([pu[u]["sums"]["vary"] for u in us]), np.array([pu[u]["n_windows"] for u in us])
def report(name, us, n_c, n_r, den):
    draws = np.random.default_rng(SEED).integers(0, len(us), (N_BOOT, len(us)))
    point = 100*(1-n_c.sum()/n_r.sum())
    boot = 100*(1-n_c[draws].sum(1)/n_r[draws].sum(1))
    r = 100*(1-n_c/n_r)
    w = n_r/n_r.sum()
    big = w > 0.01
    print(f"{name}: rel={point:+.3f}%  sd_boot={boot.std(ddof=1):.3f}  IC95={np.percentile(boot,[2.5,97.5]).round(3).tolist()} IC99={np.percentile(boot,[0.5,99.5]).round(3).tolist()}  sd_fra_utenti={r[big].std(ddof=1):.3f} (n={big.sum()}) min={r[big].min():+.2f} max={r[big].max():+.2f}  n_eff={1/np.sum(w**2):.2f}")
# 1) lettore temporale contro istantaneo (B student 19618, crop4)
d = json.load(open(f"{SNAP}/results/infra/pose_temporal_20261102/execution/run.json"))
for model in ("B", "R3c"):
    us, ni, den, nw = per_user(d["cells"][f"{model}/instantaneous"]["analysis"])
    us2, nt, den2, _ = per_user(d["cells"][f"{model}/temporal"]["analysis"])
    assert us == us2 and np.array_equal(den, den2)
    report(f"lettore lineare con contesto vs istantaneo, {model}", us, nt, ni, den)
# 2) tre bracci del tokenizer a 19618 e 5000 (ricette diverse, stesso seme)
for step in ("19618", "5000"):
    d = json.load(open(f"{SNAP}/results/infra/tokenizer_evaluation_20261103/cpu_{step}/final/report.json"))
    cells = d["cells"]
    arms = {}
    for a in cells:
        arms[a] = per_user(cells[a]["analysis"])
    for cand, ctrl in (("T100-L","T25-L"),("T100-G","T100-L"),("T100-G","T25-L")):
        us, nc, den, _ = arms[cand]; us2, nr, den2, _ = arms[ctrl]
        assert us == us2 and np.array_equal(den, den2)
        report(f"tokenizer {step}: {cand} vs {ctrl}", us, nc, nr, den)
# 3) valutazione lunga: testa allenata contro ridge alle tre dosi; dosi
fits = {s: json.load(open(f"{W}/LONG-FIT-G{s}-pose.json")) for s in (19618, 49046)}
for s in fits:
    us, nr, den, _ = per_user(fits[s]["cells"]["post_ridge"]["analysis"])
    _, nh, _, _ = per_user(fits[s]["cells"]["trained_head"]["analysis"])
    report(f"lungo {s}: testa allenata vs ridge post", us, nh, nr, den)
for reader in ("post_ridge", "trained_head"):
    us, n1, den, _ = per_user(fits[49046]["cells"][reader]["analysis"])
    _, n0, _, _ = per_user(fits[19618]["cells"][reader]["analysis"])
    report(f"lungo dose 49046 vs 19618, {reader}", us, n1, n0, den)
us, npost, den, _ = per_user(fits[49046]["cells"]["post_ridge"]["analysis"])
_, npre, _, _ = per_user(fits[49046]["cells"]["pre_ridge"]["analysis"])
report("lungo 49046: ridge post vs ridge pre", us, npost, npre, den)
