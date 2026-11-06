"""Descriptive dose comparison from verified teacher/crop6 aggregate reports."""
from pathlib import Path
import argparse,json,hashlib

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
base=a.root/'results/infra/tokenizer_evaluation_20261103'
old_path=base/'cpu_5000/final/report.json';new_path=a.directory/'report.json'
old=json.loads(old_path.read_text());new=json.loads(new_path.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert old['complete'] and new['complete'] and old['step']==5000 and new['step']==19618
assert old['recipe']==new['recipe'] and new['recipe']['offset']==6 and new['recipe']['which']=='teacher'
assert old['reference_pins']==new['reference_pins'] and old['runtime']['sources']==new['runtime']['sources']
summary={'description':'Same teacher/crop6/refitted ridge estimand at 5000 and 19618; descriptive dose comparison, not a randomized dose effect or new confirmatory test. Each checkpoint alpha selected on train only.','reports':{'5000':sha(old_path),'19618':sha(new_path)},'cells':{}}
for arm in ('T25-L','T100-L','T100-G'):
    summary['cells'][arm]={}
    for split in ('train','val'):
        assert old['inspections'][arm+'/'+split]['users']==new['inspections'][arm+'/'+split]['users']
        assert old['inspections'][arm+'/'+split]['targets']==new['inspections'][arm+'/'+split]['targets']
    for group in ('dynamic','all','static'):
        previous=old['cells'][arm]['analysis']['groups'][group]
        current=new['cells'][arm]['analysis']['groups'][group]
        assert previous['centered']['sums']['vary']==current['centered']['sums']['vary']
        assert previous['centered']['n_valid_elements']==current['centered']['n_valid_elements']
        before=previous['centered']['centered_mse_ratio'];after=current['centered']['centered_mse_ratio']
        summary['cells'][arm][group]={'E_5000':before,'E_19618':after,'E_delta':after-before,'relative_E_reduction':1-after/before,'variance_ratio_5000':previous['centered']['variance_ratio'],'variance_ratio_19618':current['centered']['variance_ratio'],'correlation_5000':previous['centered']['pooled_centered_correlation'],'correlation_19618':current['centered']['pooled_centered_correlation'],'mae_rad_5000':previous['mae_rad'],'mae_rad_19618':current['mae_rad']}
(a.directory/'dose_comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2,allow_nan=False))
