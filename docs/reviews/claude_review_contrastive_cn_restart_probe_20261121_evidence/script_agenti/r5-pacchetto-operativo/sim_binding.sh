#!/bin/bash
# Simulation of the review binding flow in a private clone (never the reviewed clone).
set -u
S=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_probe/agents/r5-pacchetto-operativo/sim
PY=~/.venvs/wearusfm-tok/bin/python
CAND=9d34dc690cf60fcbfe405279dc17546655fa3f5b
REF=refs/remotes/origin/worktree-claude-training-engine
RP=docs/reviews/claude_review_contrastive_cn_restart_probe_20261121.md
BINDING=results/infra/contrastive_targets_20261118/cn_restart_probe_20261121/release_binding.json
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts:scripts/diagnostics
cd "$S" || exit 1
GOOD='{"schema":"cn-restart-review-v1","status":"APPROVED_FOR_DIAGNOSTIC","reviewed_commit":"9d34dc690cf60fcbfe405279dc17546655fa3f5b","package_sha256":"b0640ca6caded7d2c43b5ba804265b05a4c3e5494f0a05a8b8275822d26dcfa3","plan_sha256":"9d5811b6833f00d0698774ec0068fd0e6a4b32dc6815e979c45249167eb350d6"}'
BAD=${GOOD/b0640ca6/b0640ca7}

review_commit() {  # $1 = body file ; commit on a fake Claude branch (orphan), built with plumbing
  local blob t1 t2 c
  blob=$(git hash-object -w "$1")
  t1=$(printf '100644 blob %s\t%s\n' "$blob" "claude_review_contrastive_cn_restart_probe_20261121.md" | git mktree)
  t2=$(printf '040000 tree %s\treviews\n' "$t1" | git mktree)
  t3=$(printf '040000 tree %s\tdocs\n' "$t2" | git mktree)
  c=$(git commit-tree "$t3" -m review)
  git update-ref "$REF" "$c"
  echo "$c"
}
try() {  # $1 label, $2 commit
  rm -f "$BINDING"
  out=$($PY -B scripts/diagnostics/contrastive_cn_restart.py --bind-review "$2" 2>&1 | tail -1)
  b=$([ -f "$BINDING" ] && echo BOUND || echo NOT_BOUND)
  rq=$($PY -B -c "import contrastive_cn_restart as c; c.require_binding(); print('REQUIRE_OK')" 2>&1 | tail -1)
  echo "[$1] bind: $b | last: ${out:0:160} | require: ${rq:0:160}"
}
mkdir -p /tmp/none 2>/dev/null; D=$(mktemp -d "$S/../bodies.XXXX")
printf '# Review\n\nVerdetto àèìòù — “ok”.\n\n```cn-restart-review\n%s\n```\n' "$GOOD" > $D/good.md
printf '# Review\n\n```cn-restart-review\n%s\n```\n\nEsempio:\n\n```cn-restart-review\n%s\n```\n' "$GOOD" "$GOOD" > $D/double.md
printf '# Review\n\n```cn-restart-review\n%s\n```\n' "$BAD" > $D/badhash.md
printf '# Review\r\n\r\n```cn-restart-review\r\n%s\r\n```\r\n' "$GOOD" > $D/crlf.md
printf '# Review\n\n```cn-restart-review   \n%s   \n```  \n' "$GOOD" > $D/trailing.md
printf '# Review\n\n> ```cn-restart-review\n> %s\n> ```\n\n```cn-restart-review\n%s\n```\n' "$GOOD" "$GOOD" > $D/quoted_plus_real.md
printf '# Review\n\n    ```cn-restart-review\n    %s\n    ```\n\n```cn-restart-review\n%s\n```\n' "$GOOD" "$GOOD" > $D/indented_plus_real.md
printf '# Review\n\n````text\n```cn-restart-review\n%s\n```\n````\n\n```cn-restart-review\n%s\n```\n' "$GOOD" "$GOOD" > $D/fenced_example_plus_real.md
printf '# Review\n\n```contrastive-release-review\n{"schema":"contrastive-target-release-review-v1","status":"APPROVED","reviewed_commit":"fbfde797b48644744c36547533acf0e8a7793535"}\n```\n' > $D/oldblock.md
pretty=$(python3 -c "import json,sys;print(json.dumps(json.loads(sys.argv[1]),indent=2))" "$GOOD")
printf '# Review\n\n```cn-restart-review\n%s\n```\n' "$pretty" > $D/pretty.md
for k in good double badhash crlf trailing quoted_plus_real indented_plus_real fenced_example_plus_real oldblock pretty; do
  c=$(review_commit $D/$k.md); try "$k" "$c"
done
# not an ancestor of the review ref: commit good body, then move ref elsewhere
c=$(review_commit $D/good.md); git update-ref "$REF" $CAND; try "not_ancestor" "$c"
# abbreviated commit
c=$(review_commit $D/good.md); try "short_commit" "${c:0:12}"
# package file modified after review (working tree, as in an unreviewed later commit)
c=$(review_commit $D/good.md); rm -f "$BINDING"
$PY -B scripts/diagnostics/contrastive_cn_restart.py --bind-review "$c" >/dev/null 2>&1 && echo "[bound good for tamper tests]"
cp "$BINDING" $D/binding.json
echo '# x' >> scripts/diagnostics/contrastive_cn_restart_compare.py
echo "[tamper compare.py] require: $($PY -B -c 'import contrastive_cn_restart as c; c.require_binding(); print("REQUIRE_OK")' 2>&1 | tail -1 | cut -c1-160)"
git checkout -q -- scripts/diagnostics/contrastive_cn_restart_compare.py
echo '# x' >> tests/cpu/test_cn_restart_compare.py
echo "[tamper test file, outside package] require: $($PY -B -c 'import contrastive_cn_restart as c; c.require_binding(); print("REQUIRE_OK")' 2>&1 | tail -1 | cut -c1-160)"
git checkout -q -- tests/cpu/test_cn_restart_compare.py
echo '# x' >> scripts/contrastive_job.py
echo "[tamper production module] require: $($PY -B -c 'import contrastive_cn_restart as c; c.require_binding(); print("REQUIRE_OK")' 2>&1 | tail -1 | cut -c1-160)"
git checkout -q -- scripts/contrastive_job.py
echo '# x' >> scripts/submit_job.py
echo "[tamper scripts/submit_job.py (not run on cluster)] require: $($PY -B -c 'import contrastive_cn_restart as c; c.require_binding(); print("REQUIRE_OK")' 2>&1 | tail -1 | cut -c1-160)"
git checkout -q -- scripts/submit_job.py
echo '{}' > results/infra/contrastive_targets_20261118/reloads_execution_20261121/CN_A/start.json
echo "[tamper evidence start.json] require: $($PY -B -c 'import contrastive_cn_restart as c; c.require_binding(); print("REQUIRE_OK")' 2>&1 | tail -1 | cut -c1-160)"
git checkout -q -- results/infra/contrastive_targets_20261118/reloads_execution_20261121/CN_A/start.json
# forged binding: valid self-hash but review text not from Claude branch
python3 - "$BINDING" <<'EOF'
import json,hashlib,sys
p=sys.argv[1]; b=json.load(open(p)); b["review_commit"]="0"*40; b["review_text"]=b["review_text"].replace("Verdetto","Falso")
b["review_sha256"]=hashlib.sha256(b["review_text"].encode()).hexdigest(); open(p,"w").write(json.dumps(b,indent=2)+"\n")
EOF
echo "[forged binding, self-consistent] require: $($PY -B -c 'import contrastive_cn_restart as c; c.require_binding(); print("REQUIRE_OK")' 2>&1 | tail -1 | cut -c1-160)"
cp $D/binding.json "$BINDING"
# commit the binding as release commit and recheck from a clean tree
git add "$BINDING" && git commit -qm "bind" && echo "[release commit $(git rev-parse --short HEAD)] status: '$(git status --porcelain)' require: $($PY -B -c 'import contrastive_cn_restart as c; c.require_binding(); print("REQUIRE_OK")' 2>&1 | tail -1 | cut -c1-120)"
# second bind on existing binding
echo "[rebind existing] $($PY -B scripts/diagnostics/contrastive_cn_restart.py --bind-review "$c" 2>&1 | tail -1 | cut -c1-120)"
git checkout -qf --detach $CAND
rm -rf "$D"
