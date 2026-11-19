"""Paired secondary dose-by-size contrast from validated endpoint sufficient statistics."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path



def require(ok,message):
    if not ok:raise ValueError(message)


def analyse(old,new):
    require(old.get('checkpoint_step')==19618 and new.get('checkpoint_step')==49046,'wrong endpoint doses')
    require(all(r.get('complete') is True and set(r.get('quality',{}))=={'artifact_integrity','paired_support','solver_converged'} and all(v is True for v in r['quality'].values()) and
                r.get('candidate')=='T100-G-300M' and r.get('control')=='T100-G-30M' for r in (old,new)),
            'incomplete endpoint or wrong size contrast')
    result={'protocol':'scaling-dose-interaction-20261118-v1',
            'formula':'(300M49-300M19)-(30M49-30M19)', 'secondary':True,
            'status':'descriptive_only', 'comparable':False,
            'reason':'No four-cell artifact certificate for row/target/fold/preprocessing/reader/runtime parity. Within-dose pairing and equal counts do not establish between-dose pairing.',
            'epn':{'delta19_pp':old['epn']['delta_pp'],'delta49_pp':new['epn']['delta_pp'],
                   'descriptive_difference_pp':new['epn']['delta_pp']-old['epn']['delta_pp']},
            'pose':{'control19':old['pose']['control_E_dyn'],'candidate19':old['pose']['candidate_E_dyn'],
                    'control49':new['pose']['control_E_dyn'],'candidate49':new['pose']['candidate_E_dyn'],
                    'descriptive_difference_E_dyn':new['pose']['delta_E_dyn']-old['pose']['delta_E_dyn']},
            'inferential_interaction_computed':False,
            'scope':'single pretraining seed, development users; no interaction CI or decision from this secondary'}
    return result


def execute(old_path,expected_old_sha,new_path,out):
    old_bytes=Path(old_path).read_bytes()
    require(hashlib.sha256(old_bytes).hexdigest()==expected_old_sha,'19k reference report changed')
    result=analyse(json.loads(old_bytes),json.loads(Path(new_path).read_text()))
    with Path(out).open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result
