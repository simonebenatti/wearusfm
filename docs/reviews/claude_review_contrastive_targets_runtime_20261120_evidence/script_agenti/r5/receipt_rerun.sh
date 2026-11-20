#!/bin/zsh
R=/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/r5
W=$R/wt_m1
until [ "$(cat $R/results_full2_*.jsonl | wc -l | tr -d ' ')" -ge 29 ]; do sleep 5; done
sleep 3
git -C $W status --porcelain
cd $W
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$W/src:$W/scripts
start=$(date +%s)
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python -c "
import json, sys, subprocess
rec=json.load(open('results/infra/contrastive_targets_20261118/amendment_20261119/validation_final/local_validation.json'))
cmd=rec['runs'][0]['command']
assert cmd[0].endswith('python') and cmd[1:8]==['-m','pytest','-q','-p','no:cacheprovider','-o','tmp_path_retention_policy=none']
args=[sys.executable,'-B',*cmd[1:8],'--basetemp=$R/bt_receipt2',*cmd[8:]]
with open('$R/receipt_rerun2.log','wb') as out:
    rc=subprocess.run(args, stdout=out, stderr=subprocess.STDOUT).returncode
print('rc', rc)
"
end=$(date +%s)
echo "wall=$((end-start))s"
tail -2 $R/receipt_rerun2.log
diff <(grep -v "passed in" $R/receipt_rerun2.log) <(grep -v "passed in" $W/results/infra/contrastive_targets_20261118/amendment_20261119/validation_final/pytest.log) && echo "progress lines identical"
git -C $W status --porcelain | wc -l
rm -rf $R/bt_receipt2
