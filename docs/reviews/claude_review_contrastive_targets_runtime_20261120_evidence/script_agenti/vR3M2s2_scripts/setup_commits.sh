#!/bin/bash
# Build, without touching the working tree, three review commits in the temp clone:
#  A  = APPROVED review on branch sim-claude-review (parent 0efa4d45)
#  B  = child of A that retracts the review (tip of sim-claude-review)
#  O  = orphan commit (no parent, no ref) with the same review bytes as A
set -euo pipefail
D="$1"; W="$2"
cd "$D"
GIT="git -c user.name=Sim -c user.email=sim@example.invalid"
REV=docs/reviews/claude_review_sim_20261120.md
python3 - "$W/review_A.md" "$W/review_B.md" <<'PY'
import json, sys
payload = {"schema": "contrastive-target-release-review-v1", "status": "APPROVED",
           "reviewed_commit": "fbfde797b48644744c36547533acf0e8a7793535",
           "source_sha256": "40d39f568958004415763f9cc683f3be0b379d170b70c4d37588d1ddcec5cde0",
           "science_plan_sha256": "b03c0bc679f7247d703635d7065c9fc7bda4a292f2c8c106e29004e0527e021a",
           "preflight_plan_sha256": "e4b3959112db86dac806978cf65307d97694da68b64096120487e8f568c6dd92"}
a = "# Review simulata (scettico R3-M2-2)\n\nTesto.\n\n```contrastive-release-review\n" + json.dumps(payload, indent=2) + "\n```\n"
b = "# Review simulata (scettico R3-M2-2)\n\nAPPROVAZIONE RITIRATA: non usare.\n"
open(sys.argv[1], "w").write(a); open(sys.argv[2], "w").write(b)
PY
blobA=$(git hash-object -w "$W/review_A.md"); blobB=$(git hash-object -w "$W/review_B.md")
export GIT_INDEX_FILE="$W/index.tmp"
rm -f "$GIT_INDEX_FILE"
git read-tree 0efa4d45
git update-index --add --cacheinfo 100644,$blobA,$REV
treeA=$(git write-tree)
A=$($GIT commit-tree $treeA -p 0efa4d45 -m "Sim: Claude review APPROVED")
git update-index --add --cacheinfo 100644,$blobB,$REV
treeB=$(git write-tree)
B=$($GIT commit-tree $treeB -p $A -m "Sim: review retracted")
git update-ref refs/heads/sim-claude-review $B
rm -f "$GIT_INDEX_FILE"; git read-tree --empty; rm -f "$GIT_INDEX_FILE"
unset GIT_INDEX_FILE
# orphan: tree with only the review file
export GIT_INDEX_FILE="$W/index2.tmp"; rm -f "$GIT_INDEX_FILE"
git update-index --add --cacheinfo 100644,$blobA,$REV
treeO=$(git write-tree)
O=$($GIT commit-tree $treeO -m "Sim: orphan review")
rm -f "$GIT_INDEX_FILE"; unset GIT_INDEX_FILE
echo "A=$A"; echo "B=$B"; echo "O=$O"
echo "branches containing O: [$(git branch -a --contains $O 2>/dev/null | tr -d '\n')]"
echo "branches containing A: [$(git branch -a --contains $A 2>/dev/null | tr -d '\n')]"
echo "tip sim-claude-review = $(git rev-parse sim-claude-review)"
git status --short | head
