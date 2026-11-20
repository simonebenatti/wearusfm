import json, sys
W, SNAP = sys.argv[1], sys.argv[2]
r = json.load(open(W + "/RIC_49046_mine.json"))
p = json.load(open(SNAP + "/results/infra/pose_temponet_proposal_20261117/proposal.json"))
chk = []
def eq(name, a, b): chk.append((name, a == b, str(a)[:70], str(b)[:70]))
eq("checkpoint_sha256", p["checkpoint_sha256"], r["checkpoint_sha256"])
eq("checkpoint_step", p["checkpoint_step"], r["checkpoint_step"])
for split in ("train", "val"):
    rc = r["caches"][split]["cache"]
    eq(split + " cache sha256", p["caches"][split]["sha256"], rc["sha256"])
    eq(split + " cache bytes", p["caches"][split]["bytes"], rc["bytes"])
    eq(split + " cache basename", p["caches"][split]["path"].split("/")[-1], rc["path"].split("/")[-1])
for ref, art in (("post_ridge_coefficients", "post_ridge_coefficients"), ("post_ridge_predictions", "post_ridge_predictions"),
                 ("trained_head_predictions", "trained_head_predictions")):
    eq(ref + " sha256", p["references"][ref]["sha256"], r["artifacts"][art]["sha256"])
    eq(ref + " bytes", p["references"][ref]["bytes"], r["artifacts"][art]["bytes"])
    eq(ref + " source_path", p["references"][ref]["source_path"], r["artifacts"][art]["path"])
eq("n_users fit", p["caches"]["train"]["users"], r["cells"]["post_ridge"]["fit"]["n_users"])
eq("n_users val", p["caches"]["val"]["users"], r["cells"]["post_ridge"]["analysis"]["groups"]["dynamic"]["centered"]["n_users"])
eq("dynamic threshold", p["support"]["dynamic_threshold"], r["cells"]["post_ridge"]["analysis"]["dynamic_threshold"])
for name, ok, a, b in chk:
    print("%-34s %s  %s | %s" % (name, "UGUALE" if ok else "DIVERSO", a, b))
print("totale uguali: %d su %d" % (sum(c[1] for c in chk), len(chk)))
# utenti dei fold
fu = r["cells"]["post_ridge"]["fit"]["selection"]["fold_users"]
allu = sorted(fu[0] + fu[1])
alt = [allu[0::2], allu[1::2]]
print("fold = alternanza degli ID ordinati:", sorted(fu[0]) == alt[0] and sorted(fu[1]) == alt[1], "| dimensioni", len(fu[0]), len(fu[1]))
# esistono le viste pre/lat nella cache del train? (ricevuta di estrazione)
t = json.load(open(SNAP + "/results/infra/consolidation_long_20261105/execution/receipts/LONG-G49046-pose-train-20261105.json"))
s = json.dumps(t)
for key in ("\"pre\"", "\"lat\"", "pose_head", "loss_per_window", "pose_weight", "which"):
    print(key, "presente nella ricevuta train:", key in s)
