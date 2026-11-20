"""Does the delivered historical_producer refuse a receipt that lists one user directory less / one more? (reviewer TESTS)"""
import json, sys
from pathlib import Path
sys.path[:0] = ["scripts", "src"]
from pose_temponet import provenance as P
proposal = json.loads(Path("results/infra/pose_temponet_proposal_20261117/proposal.json").read_text())
for split in ("train", "val"):
    receipt = json.loads(Path(P.HISTORICAL_RECEIPTS[split]["local_copy"]).read_text())
    arguments = {"split": split, "cache_pin": proposal["caches"][split], "stats_sha256": proposal["stats_sha256"]}
    good = P.historical_producer(receipt, **arguments)
    first = good["storage_directories"][0]
    states = receipt["raw_inputs"]["file_states"]
    for label, changed in (("one directory removed", {k: v for k, v in states.items() if "/emg2pose/"+first+"/" not in k}),
                           ("one directory added", {**states, next(iter(states)).replace("/"+first+"/", "/uffffffffff/"): {}})):
        receipt["raw_inputs"]["file_states"] = changed
        try:
            out = P.historical_producer(receipt, **arguments)
            print(split, label, "-> ACCEPTED with", len(out["users"]), "users")
        except ValueError as error:
            print(split, label, "-> refused:", error)
    receipt["raw_inputs"]["file_states"] = states
