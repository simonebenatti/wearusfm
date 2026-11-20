"""Skeptic R5-1: which rejection branches of contrastive_release.py do the 465 tests reach?"""
import collections
import sys

log = sys.argv[1]
hits = collections.defaultdict(set)      # tag -> tests
per_test = collections.defaultdict(list)
for line in open(log):
    test, _, tag = line.rstrip("\n").partition("|")
    test = test.split(" ")[0]
    hits[tag].add(test)
    per_test[test].append(tag)

files = sorted({t.split("::")[0] for t in per_test})
print("test files reaching contrastive_release:", files)
print("tests reaching it:", len(per_test))
# barrier raise lines named in R5-1 (original line numbers at fbfde797)
named = {"L48": "pin dentro il checkout (47)", "L69": "git_bytes forma (68)", "L73": "git show fallito (72-73)",
         "L78": "verify_reviewed_sources: commit completo (77)", "L84": "sorgenti al reviewed_commit (83)",
         "L87": "inventario/digest (86)", "L93": "un solo blocco (92)", "L97": "chiavi/schema/status (96)",
         "L99": "reviewed_commit completo (98)", "L102": "formato hash (101)", "L108": "schema binding (107)",
         "L110": "review_commit completo (109)", "L115": "piani di questa coorte (114)",
         "L119": "hash dichiarati sorgente/piani (117-118)", "L123": "legame interno preflight (121-122)",
         "L128": "byte dei piani al reviewed_commit (127)", "L133": "copia/payload binding (132)",
         "L143": "bind: byte-identita' copia (142)", "L225": "make_job cella unica (224)", "L228": "make_job modo (227)",
         "L231": "make_job input vs piano (230)", "L235": "make_job override (234)", "L42": "pin completo (41)",
         "L50": "pin mismatch (49)"}
for tag, label in named.items():
    tests = sorted(hits.get(tag, ()))
    print(f"{tag:5s} {label:45s} hit by {len(tests)}: {[t.split('::')[-1][:70] for t in tests]}")
print()
for tag in sorted(t for t in hits if t.startswith(("L91", "L94", "L116"))):
    print(tag, len(hits[tag]))
print()
for tag in sorted(t for t in hits if t.startswith("enter:")):
    print(tag, sorted({x.split('::')[0].split('/')[-1] for x in hits[tag]}))
# where L73 is raised from: the tag preceding it in the same test
print()
for test, tags in per_test.items():
    for i, tag in enumerate(tags):
        if tag in ("L73", "L50", "L93", "L143", "L235", "L225"):
            ctx = [t for t in tags[max(0, i - 6):i] if not t.startswith(("enter:file_sha", "enter:_full_sha"))]
            print(test.split("::")[-1][:80], tag, "after", ctx[-4:])
