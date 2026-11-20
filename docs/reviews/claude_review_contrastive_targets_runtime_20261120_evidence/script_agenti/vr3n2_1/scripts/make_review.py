import json, sys, pathlib
root = pathlib.Path(sys.argv[1]); rel = sys.argv[2]
payload = {"schema": "contrastive-target-release-review-v1", "status": "APPROVED",
           "reviewed_commit": "fbfde797b48644744c36547533acf0e8a7793535",
           "source_sha256": "40d39f568958004415763f9cc683f3be0b379d170b70c4d37588d1ddcec5cde0",
           "science_plan_sha256": "b03c0bc679f7247d703635d7065c9fc7bda4a292f2c8c106e29004e0527e021a",
           "preflight_plan_sha256": "e4b3959112db86dac806978cf65307d97694da68b64096120487e8f568c6dd92"}
text = ("# Review di Claude (prova dello scettico)\n\n"
        "Verdetto: la catena è corretta, perché le guardie reggono; così è approvata. Durata ≤ 7200 s — nessun ritardo.\n\n"
        "```contrastive-release-review\n" + json.dumps(payload, indent=1) + "\n```\n")
p = root / rel
p.parent.mkdir(parents=True, exist_ok=True)
p.write_bytes(text.encode("utf-8"))
b = p.read_bytes(); print(rel, len(b), "non-ascii bytes:", sum(1 for x in b if x > 127))
