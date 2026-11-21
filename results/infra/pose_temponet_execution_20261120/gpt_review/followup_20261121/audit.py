"""Verify Claude's final documentary/test follow-up, without release or cluster I/O."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from pose_temponet import provenance as P, execution as E

OLD = "0e548bf8218f15ba07fc731efcc2ead512e65034"
TIP = "5146ecc2ca7215f739bce54ed064d44aa48ad50f"
CODE = "5d6fb2dc855bd3c12788bb6d3c3a66043c34f1ba"
HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    assert git("diff", OLD, TIP, "--", "src", "scripts") == b""
    assert git("diff", TIP, "HEAD", "--", "src", "scripts") == b""
    plan_path = CAMPAIGN / "plan.json"
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    assert hashlib.sha256(plan_bytes).hexdigest() == "1e8e2b46b046072c706738995e745f608f009d33406d970729a3eb22a73f563e"
    assert plan_bytes == git("show", OLD + ":" + str(plan_path.relative_to(ROOT)))
    manifest = P.source_manifest()
    source = P.source_identity()
    assert source == "37985d28c75bf82a6a784106d414825b727b9ba78575b675dd2dac1d4ccfc8de"
    assert len(manifest) == 246 and source == plan["reader"]["source_sha256"]
    for path, digest in manifest.items():
        assert hashlib.sha256(git("show", CODE + ":" + path)).hexdigest() == digest
    assert P.canonical_bytes(P.build_plan(plan["proposal"]["path"], jobs=E.job_graph())) == P.canonical_bytes(plan)
    assert plan["status"] == "REVIEW_REQUIRED" and plan["submission_released"] is False
    pins = json.loads((CAMPAIGN / "delivery_pin.json").read_bytes())
    verified = []
    for name, value in pins.items():
        if isinstance(value, dict) and {"path", "sha256", "bytes"} <= value.keys():
            path = ROOT / value["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == value["sha256"], name
            assert path.stat().st_size == value["bytes"], name
            verified.append(name)
    summaries = {}
    for filename in ("review2_realdata.json", "review2_tests.json"):
        evidence = json.loads((CAMPAIGN / "independent_review" / filename).read_bytes())
        assert all(row["severity"] in ("minor", "note") for row in evidence["findings"])
        summaries[filename] = {"findings": len(evidence["findings"]),
                               "severity": [row["severity"] for row in evidence["findings"]]}
    result = {"reviewed_tip": TIP, "reviewed_code": CODE, "previous_delivery": OLD,
              "source_sha256": source, "source_files": 246, "source_diff_empty": True,
              "same_unreleased_plan": True, "delivery_pins_verified": verified,
              "archived_reviews": summaries, "release_performed": False, "cluster_operations": False}
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
