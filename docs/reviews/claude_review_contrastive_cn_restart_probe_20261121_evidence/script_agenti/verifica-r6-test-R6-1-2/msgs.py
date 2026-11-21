import sys, json, tempfile, pathlib
if sys.argv[1] == "shim":
    import shim22  # SIMULAZIONE torch 2.2
from contrastive_cn_restart_observe import load_observation
src = pathlib.Path(sys.argv[2])
for field in ["none", "pin", "name_collision", "rank", "shape", "gradient_hash", "slot", "bucket_members"]:
    d = pathlib.Path(tempfile.mkdtemp(dir=sys.argv[3]))
    rec = json.loads((src / "observation.rank0.json").read_text())
    (d / "vectors.rank0.pt").hardlink_to(src / "vectors.rank0.pt")
    if field == "pin": rec["vectors"]["sha256"] = "0" * 64
    elif field == "name_collision": rec["parameters"][1]["name"] = rec["parameters"][0]["name"]
    elif field == "rank": rec["rank"] = 1
    elif field == "shape": rec["parameters"][0]["shape"] = [99]
    elif field == "gradient_hash": rec["parameters"][0]["local_sha256"] = "0" * 64
    elif field == "slot": rec["parameters"][0]["bucket_position"] = 99
    elif field == "bucket_members": rec["buckets"][0]["parameter_ids"] = []
    (d / "observation.rank0.json").write_text(json.dumps(rec))
    try:
        load_observation(d, 0, expected_parameters=3); print(field, "NO ERROR")
    except ValueError as e:
        print(field, "->", e)
