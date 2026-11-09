"""Attestation of the jobs already run (v1 checkout, failed v1 training, four diagnoses) against the signed envelope.

Usage (repository root): python -B <this folder>/attest_spent.py
Reads the raw `sacct -P` section of remote_sha256.txt next to this file and the `spent` rows of the signed envelope
(continuation19618_20261109_v2/budget_envelope.json); writes attestation.json here (never over an existing file).
Every parent row must give the envelope's elapsed seconds, CPUs, GPUs and state; any difference stops the recovery.
"""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
ENVELOPE = ROOT / "results/infra/scaling_300m_20261104/continuation19618_20261109_v2/budget_envelope.json"
SECTION = "=== sacct -P dei sei job gia' eseguiti (passi compresi)"


def sacct_rows(text):
    lines = text.splitlines()
    start = lines.index(SECTION) + 1
    block = []
    for line in lines[start:]:
        if line.startswith("=== "):
            break
        block.append(line)
    header = block[0].split("|")
    return [dict(zip(header, line.split("|"))) for line in block[1:] if line.strip()]


def main():
    rows = sacct_rows((HERE / "remote_sha256.txt").read_text())
    parents = {r["JobIDRaw"]: r for r in rows if "." not in r["JobIDRaw"]}
    envelope = json.loads(ENVELOPE.read_text())
    jobs = []
    for spent in envelope["spent"]:
        row = parents.get(spent["slurm_job_id"], {})
        tres = dict(x.split("=", 1) for x in row.get("AllocTRES", "").split(",") if "=" in x)
        measured = {"logical_id": row.get("JobName"), "elapsed_seconds": int(row["ElapsedRaw"]) if row else None,
                    "cpus": int(row["AllocCPUS"]) if row else None, "gpus": int(tres.get("gres/gpu", 0)) if row else None,
                    "state": "%s %s" % (row.get("State"), row.get("ExitCode")) if row else None}
        expected = {key: spent[key] for key in ("logical_id", "elapsed_seconds", "cpus", "gpus")}
        expected["state"] = spent["state"] or "COMPLETED 0:0"
        peaks = [r["MaxRSS"] for r in rows if r["JobIDRaw"].startswith(spent["slurm_job_id"] + ".") and r["MaxRSS"]]
        jobs.append({"slurm_job_id": spent["slurm_job_id"], "expected": expected, "sacct": measured,
                     "max_rss": peaks, "match": measured == expected})
    report = {"protocol": "scaling-continuation-recovery-attestation-20261109-v1", "envelope_sha256": envelope_digest(),
              "jobs": jobs, "all_match": all(job["match"] for job in jobs) and len(jobs) == 6}
    with (HERE / "attestation.json").open("x") as stream:
        json.dump(report, stream, indent=1)
        stream.write("\n")
    print(json.dumps({"all_match": report["all_match"], "jobs": [(j["slurm_job_id"], j["match"]) for j in jobs]}))
    return 0 if report["all_match"] else 1


def envelope_digest():
    import hashlib
    return hashlib.sha256(ENVELOPE.read_bytes()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
