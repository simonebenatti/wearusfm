"""Summarize sk1trace output for the R5-2 guard lines (fbfde797 numbering)."""
import json
import sys

data = json.load(open(sys.argv[1]))
hits = {k: set(v) for k, v in data["hits"].items()}
by_test = data["by_test"]
GUARDS = [
    ("contrastive_job.py", "70 binding call", [70]),
    ("contrastive_job.py", "71-72 review vs binding", [71, 72]),
    ("contrastive_job.py", "76-79 cell", [76, 77, 79]),
    ("contrastive_job.py", "81 accounting call", [81]),
    ("contrastive_job.py", "84-86 cohort root", [84, 85, 86]),
    ("contrastive_job.py", "141-142 release==job", [141, 142]),
    ("contrastive_job.py", "145-146 cuda prereq presence", [145, 146]),
    ("contrastive_job.py", "154-155 resume pins", [154, 155]),
    ("contrastive_run.py", "182 binding call", [182]),
    ("contrastive_run.py", "183-184 review_pin", [183, 184]),
    ("contrastive_run.py", "193-194 sacct presence", [193, 194]),
    ("contrastive_run.py", "195 accounting call", [195]),
    ("contrastive_run.py", "209 cuda prereq call", [208, 209]),
    ("contrastive_run.py", "210-214 phase/arm", [210, 211, 213]),
    ("contrastive_run.py", "251-266 validate_target_fork", [251, 255, 263, 265]),
    ("contrastive_run.py", "317 target fork call", [317]),
    ("contrastive_run.py", "338-340 profile ctx", [338, 339, 340]),
    ("contrastive_run.py", "348 copy sha", [348]),
    ("contrastive_run.py", "359 numeric receipt", [359]),
    ("contrastive_run.py", "309 validate_release in main", [309]),
    ("contrastive_release.py", "107-134 validate_review_binding body", [107, 111, 116, 125, 132]),
    ("contrastive_release.py", "157-192 accounting body", [157, 159, 164, 187]),
    ("contrastive_release.py", "197-212 validate_cuda_prerequisite", [200, 203, 209, 211]),
    ("contrastive_release.py", "221 make_job binding", [221]),
    ("contrastive_release.py", "240-242 make_job accounting", [240, 241, 242]),
    ("contrastive_cuda_check.py", "219 _main body", [219, 223]),
    ("contrastive_cuda_check.py", "281 rank profile", [281]),
    ("contrastive_cuda_check.py", "301-303 main wrapper", [301, 302, 303]),
]
for name, label, lines in GUARDS:
    hit = [l for l in lines if l in hits.get(name, set())]
    tests = sorted({t.split("/")[-1] for l in hit for t in by_test.get(f"{name}:{l}", [])})
    print(f"{name}:{label}: hit {hit or 'NONE'} of {lines}; tests={tests}")
