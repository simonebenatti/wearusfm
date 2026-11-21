import json,sys,glob,hashlib
for d in sys.argv[1:]:
    for r in range(4):
        rows=[json.loads(l) for l in open(f"{d}/trace-{r}.ddp.jsonl")]
        out=[]
        for x in rows:
            out.append((x["update"],x.get("iteration"),x.get("has_rebuilt_buckets"),x.get("bucket_sizes"),hashlib.sha256((x.get("rebuilt_bucket_sizes","")+"|"+x.get("rebuilt_per_bucket_param_indices","")).encode()).hexdigest()[:8] if "rebuilt_bucket_sizes" in x else None))
        print(d,r,out[:4],out[9:12],out[-1])
