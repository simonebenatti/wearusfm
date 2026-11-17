"""Controllo ESEGUITO (seconda stesura): i pin storici di proposal.json (snapshot c15aac9) contro la ricevuta storica a 49.046
estratta da git (2930d7d). Confronto campo per campo; nessun accesso a Leonardo."""
import hashlib, json, sys
SNAP = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/gpt_c15aac9"
p = json.load(open(SNAP + "/results/infra/pose_temponet_proposal_20261117/proposal.json"))
raw = open(sys.argv[1], "rb").read()
r = json.loads(raw)
ok = []
ok.append(("sha256 ricevuta", hashlib.sha256(raw).hexdigest() == p["source_provenance"]["historical_fit_receipt_sha256"]))
ok.append(("checkpoint", p["checkpoint_sha256"] == r["cells"]["post_ridge"]["checkpoint_sha256"] == r["cells"]["trained_head"]["checkpoint_sha256"]))
for split in ("train", "val"):
    a, b = p["caches"][split], r["caches"][split]["cache"]
    ok.append((f"cache {split} sha/byte", a["sha256"] == b["sha256"] and a["bytes"] == b["bytes"]))
    ok.append((f"cache {split} nome file", a["path"].split("/")[-1] == b["path"].split("/")[-1]))
for name, key in (("post_ridge_coefficients", "post_ridge_coefficients"), ("post_ridge_predictions", "post_ridge_predictions"),
                  ("trained_head_predictions", "trained_head_predictions")):
    a, b = p["references"][name], r["artifacts"][key]
    ok.append((f"{name} sha/byte/origine", a["sha256"] == b["sha256"] and a["bytes"] == b["bytes"] and a["source_path"] == b["path"]))
ok.append(("utenti train 40", p["caches"]["train"]["users"] == r["cells"]["post_ridge"]["fit"]["n_users"] == 40))
ok.append(("utenti val 15", p["caches"]["val"]["users"] == r["cells"]["post_ridge"]["analysis"]["groups"]["dynamic"]["centered"]["n_users"] == 15))
ok.append(("soglia dinamica", p["support"]["dynamic_threshold"] == r["cells"]["post_ridge"]["analysis"]["dynamic_threshold"]))
for k, v in ok:
    print(("OK  " if v else "NO  ") + k)
