#!/usr/bin/env python3
"""Local EPN15 verification; no model inference, cluster access, or subject IDs in output."""
import csv, hashlib, io, json
from pathlib import Path
import numpy as np

REPO=Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-consolidation')
INPUT=Path('/private/tmp/wearusfm-ft15-claude-final-20261105')
OUT=Path('/private/tmp/wearusfm-ft15-gpt-science-review-20261106')
FINAL=REPO/'results/infra/finetune15_20261105/final'
REGIMES=('frozen','finetune','scratch')
report=json.loads((FINAL/'report.json').read_text())
plan=json.loads((INPUT/'executed_plan.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(OUT/'reproduced_report.json')==sha(FINAL/'report.json')
cells={}
choices=[]
for dose in (15,5):
    for regime in REGIMES:
        tables=[]
        for fold in range(4):
            base=INPUT/f'epn{dose}'/f'fold{fold}'/regime
            with np.load(base/'candidate1/held.npz',allow_pickle=False) as a:
                table={k:a[k].copy() for k in ('rows','y','pred','s','fold')}
                assert np.all(table['fold']==fold)
                assert a['input_sha256'].item()==plan['input_sha256']
            tables.append(table)
            if dose==15:
                candidates=[]
                for index in (0,1):
                    d=base/f'candidate{index}'
                    c=json.loads((d/f'candidate_{index}.json').read_text())
                    history=c['history']
                    assert len(history)==15 and [h['epoch'] for h in history]==list(range(1,16))
                    best=max(history,key=lambda h:h['val_bacc'])
                    assert c['best_epoch']==best['epoch'] and c['val_bacc']==best['val_bacc']
                    assert c['stopped']=='fixed_budget' and c['lr']==plan['lr_grid'][regime][index]
                    assert c['head_lr']==c['lr']*plan['head_lr_multiplier'][regime]
                    assert c['updates']==c['planned_updates']==history[-1]['step']
                    metric=[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines()]
                    assert metric==[{'candidate':index,'lr':c['lr'],**h} for h in history]
                    candidates.append(c)
                winner=max(candidates,key=lambda c:c['val_bacc'])
                published=next(x for x in report['choices'] if x['fold']==fold and x['regime']==regime)
                assert published['selected_candidate']==winner['candidate']
                assert published['selected_best_epoch']==winner['best_epoch']
                choices.append({'fold':fold,'regime':regime,'candidate':winner['candidate'],
                                'best_epoch':winner['best_epoch'],'lr':winner['lr'],
                                'head_lr':winner['head_lr'],'val_bacc':winner['val_bacc']})
        combined={k:np.concatenate([t[k] for t in tables]) for k in tables[0]}
        order=np.argsort(combined['rows'])
        combined={k:v[order] for k,v in combined.items()}
        assert np.array_equal(combined['rows'],np.arange(47100))
        cells[f'{dose}/{regime}']=combined
ref=cells['15/frozen']
for cell in cells.values():
    for k in ('rows','y','s','fold'):
        assert np.array_equal(cell[k],ref[k]),k
subjects=np.unique(ref['s'])
classes=np.unique(ref['y'])
assert len(subjects)==314 and len(classes)==6
support=np.array([[(ref['y'][ref['s']==s]==c).sum() for c in classes] for s in subjects])
assert np.all(support==25)
names=list(cells)
subject_hits=np.array([[(cell['pred'][cell['s']==s]==cell['y'][cell['s']==s]).sum() for s in subjects] for cell in cells.values()])
subject_accuracy=subject_hits/150
scores=subject_accuracy.mean(1)*100
class_recalls=np.array([[(cell['pred'][cell['y']==c]==c).mean()*100 for c in classes] for cell in cells.values()])
assert np.allclose(scores,class_recalls.mean(1),atol=1e-12)
# Independent implementation: direct sample indexing and mean; valid because all
# subject/class denominators are independently verified equal (25 windows).
picked=np.random.default_rng(20261102).integers(314,size=(2000,314))
boot=np.array([a[picked].mean(1)*100 for a in subject_accuracy])
contrasts={
'finetune_minus_scratch':(scores[1]-scores[2],boot[1]-boot[2]),
'finetune_minus_frozen':(scores[1]-scores[0],boot[1]-boot[0]),
'frozen15_minus_frozen5':(scores[0]-scores[3],boot[0]-boot[3]),
'finetune15_minus_finetune5':(scores[1]-scores[4],boot[1]-boot[4]),
'scratch15_minus_scratch5':(scores[2]-scores[5],boot[2]-boot[5]),
 'transfer_gap15_minus_gap5':((scores[1]-scores[2])-(scores[4]-scores[5]),(boot[1]-boot[2])-(boot[4]-boot[5]))}
checked={}
errors=[]
for name,(point,draws) in contrasts.items():
    actual={'delta_pp':float(point),'ci95_pp':np.quantile(draws,[.025,.975]).tolist()}
    published=(report['statistics'] if name in report['statistics']['contrasts'] else report['dose_comparison'])['contrasts'][name]
    errors.extend([abs(actual['delta_pp']-published['delta_pp']),*np.abs(np.array(actual['ci95_pp'])-published['ci95_pp'])])
    checked[name]=actual
assert max(errors)<1e-10
folds=[]
for fold in range(4):
    vals={name:float((cell['pred'][ref['fold']==fold]==ref['y'][ref['fold']==fold]).mean()*100) for name,cell in cells.items()}
    folds.append({'fold':fold,'n_subjects':int(len(np.unique(ref['s'][ref['fold']==fold]))),
                  'scores_percent':vals,'finetune_minus_scratch15_pp':vals['15/finetune']-vals['15/scratch']})
subject_delta=subject_hits[1]-subject_hits[2]
per_subject={'better':int((subject_delta>0).sum()),'equal':int((subject_delta==0).sum()),'worse':int((subject_delta<0).sum()),
             'median_ft_minus_scratch15_pp':float(np.median(subject_delta)/150*100)}
per_class=[{'class':str(c),'finetune15_recall_percent':float(class_recalls[1,i]),
            'scratch15_recall_percent':float(class_recalls[2,i]),
            'finetune_minus_scratch15_pp':float(class_recalls[1,i]-class_recalls[2,i])} for i,c in enumerate(classes)]
# Independently recompute resource totals from archived raw sacct, not accounting totals.
sacct=(FINAL/'sacct.txt').read_text().splitlines()
rows=list(csv.DictReader(io.StringIO('\n'.join(sacct[1:])),delimiter='|'))
assert len(rows)==27 and len({r['JobIDRaw'] for r in rows})==27
accounting=json.loads((FINAL/'accounting.json').read_text())
byid={str(j['job']):j for j in accounting['jobs']}
groups={}
for r in rows:
    assert r['State']=='COMPLETED' and r['ExitCode']=='0:0' and int(r['Restarts'])==0
    seconds=int(r['ElapsedRaw']);cpus=int(r['AllocCPUS'])
    tres=dict(x.split('=',1) for x in r['AllocTRES'].split(','));gpus=int(tres.get('gres/gpu',0))
    a=byid[r['JobIDRaw']]
    assert (seconds,cpus,gpus)==(a['seconds'],a['cpus'],a['gpus'])
    group=next((regime for regime in REGIMES if f'-{regime}-' in r['JobName']),None)
    group=group or ('benchmark' if '-BENCH-' in r['JobName'] else 'checkouts')
    v=groups.setdefault(group,{'jobs':0,'gpu_seconds':0,'core_seconds':0})
    v['jobs']+=1;v['gpu_seconds']+=seconds*gpus;v['core_seconds']+=seconds*cpus
for g in groups.values():
    g['gpu_hours']=g.pop('gpu_seconds')/3600;g['allocated_core_hours']=g.pop('core_seconds')/3600
sumgpu=sum(g['gpu_hours'] for g in groups.values());sumcore=sum(g['allocated_core_hours'] for g in groups.values())
assert abs(sumgpu-accounting['all_listed_gpu_hours'])<1e-10
assert abs(sumcore-accounting['all_listed_allocated_core_hours'])<1e-10
summary={
 'review_protocol':'gpt-local-independent-ft15-20261106-v1',
 'reviewed_commit':'3ba5e6437fcda6e5149000cee3fce5374e46b7de',
 'scope':'local published metadata and cached OOF; no cluster call, checkpoint load, inference, or fit',
 'official_aggregator_reproduction':{'byte_identical':True,'report_sha256':sha(FINAL/'report.json'),
    'manifest_sha256':sha(FINAL/'manifest.json'),'executed_plan_sha256':sha(INPUT/'executed_plan.json'),
    'code_sha':report['run_identity']['code_sha'],'source_sha256':report['run_identity']['source_sha256']},
 'independent_oof':{'rows':len(ref['rows']),'subjects':314,'classes':6,'windows_per_subject_class':25,
    'paired_all_six_cells':True,'bootstrap_seed':20261102,'bootstrap_replicates':2000,
    'method':'direct indexing of per-subject accuracy; equivalence to macro recall checked from equal class support',
    'scores_percent':{name:float(score) for name,score in zip(names,scores)},
    'contrasts':checked,'max_absolute_report_difference_pp':float(max(errors)),
    'folds':folds,'per_subject_ft_minus_scratch15':per_subject,'per_class':per_class},
 'validation_only_choices_recomputed':choices,
 'accounting_from_archived_sacct':{'all_27_completed_zero':True,'restarts':0,'groups':groups,
    'gpu_hours':sumgpu,'allocated_core_hours':sumcore,'gpu_cap':304.5,'core_cap':2436.6666666666665},
 'reading':{
    'primary':'Positive fixed-recipe FT minus scratch advantage at 15 epochs, smaller than at five.',
    'fine_tune_dose':'No improvement resolved: 15-minus-5 interval includes zero; this does not establish equivalence.',
    'gap_change':'Paired difference-in-differences, not subtraction of CI limits; descriptive interval narrowly below zero.',
    'limits':['one training seed and fixed fitted folds; subject bootstrap does not cover training randomness',
      'development held set seen twice; descriptive intervals, no multiplicity adjustment',
      'fresh fifteen-epoch cosine versus fresh five-epoch cosine; not isolated dose effect',
      'best of two candidates; selected FT and scratch encoder LR equal but head LR differ by 10x',
      'no claim of convergence, universal FM utility, or unseen-dataset transfer']}}
(OUT/'verification_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({'artifact':str(OUT/'verification_summary.json'),'max_difference_pp':max(errors),
 'subjects':per_subject,'accounting':summary['accounting_from_archived_sacct'],'fold_ft_minus_scratch15_pp':[f['finetune_minus_scratch15_pp'] for f in folds]},indent=2))
