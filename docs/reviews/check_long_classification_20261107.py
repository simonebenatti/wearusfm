#!/usr/bin/env python3
"""Local independent reading of embedded paired subject/class counts; no fits or inference."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--snapshot-root',type=Path,required=True,help='Root of the published repository snapshot')
parser.add_argument('--out',type=Path,required=True,help='Destination JSON for aggregate verification; must not exist')
args=parser.parse_args()
BASE=args.snapshot_root.resolve()
OUT=args.out.resolve()
if OUT.exists():
    parser.error('--out must name a new file')
if not OUT.parent.is_dir():
    parser.error('--out parent directory must already exist')
SUMMARY=BASE/'results/infra/consolidation_long_20261105/execution/receipts/LONG-SUMMARY-20261105.json'
s=json.loads(SUMMARY.read_text());r=s['classification_result']
assert r['complete'] and r['n_bootstrap']==2000 and r['seed']==20261102
names=[f'{dose}/{view}' for dose in (19618,39237,49046) for view in ('pre','post','pool')]
assert set(r['convergence'])==set(names)
assert all(v is True for cell in r['convergence'].values() for v in cell.values())
weights={'p1':{'ninapro_db2':8/13,'ninapro_db3':3/13,'ninapro_db6':2/13},'p2':{'epn612':.5,'uci_emg':.5}}
assert all(r['tasks'][t]['weights']==weights[t] for t in weights)
pairs={f'{v}:{a}-{b}':(f'{a}/{v}',f'{b}/{v}') for v in ('pre','post','pool') for a,b in ((49046,19618),(39237,19618),(49046,39237))}
pairs.update({f'{d}:post-pre':(f'{d}/post',f'{d}/pre') for d in (19618,39237,49046)})
points={};draws={};checks={};error=[]
rng=np.random.default_rng(20261102)
for ds,d in sorted(r['datasets'].items()):
    counts=d['paired_counts'];assert set(counts)==set(names)
    subjects=sorted(counts[names[0]]['subjects']);assert subjects==d['subjects']
    classes=counts[names[0]]['classes'];assert classes==d['classes']
    n=np.array([counts[names[0]]['subjects'][u]['n'] for u in subjects],dtype=float)
    assert np.isfinite(n).all() and (n>=0).all() and (n.sum(1)>0).all() and (n==np.floor(n)).all()
    hit=[]
    for name in names:
        c=counts[name];assert sorted(c['subjects'])==subjects and c['classes']==classes
        den=np.array([c['subjects'][u]['n'] for u in subjects],dtype=float)
        assert np.array_equal(den,n)
        h=np.array([c['subjects'][u]['correct'] for u in subjects],dtype=float)
        assert np.isfinite(h).all() and (h>=0).all() and (h<=n).all() and (h==np.floor(h)).all()
        hit.append(h)
    picked=rng.integers(0,len(subjects),size=(2000,len(subjects)))
    hh=hashlib.sha256();hh.update(json.dumps([picked.dtype.str,picked.shape]).encode());hh.update(picked.tobytes());hh.update(b'\0')
    assert hh.hexdigest()==d['draws_sha256']
    # Multiplicity-weighted matrix products, independent of original indexed reduction.
    mult=np.stack([np.bincount(row,minlength=len(subjects)) for row in picked]).astype(float)
    bd=mult@n;present=bd>0
    assert (present.sum(1)>0).all()
    draws[ds]=np.column_stack([np.divide(mult@h,bd,out=np.zeros_like(bd),where=present).sum(1)/present.sum(1) for h in hit])
    total=n.sum(0);valid=total>0
    points[ds]=np.array([(h.sum(0)[valid]/total[valid]).mean() for h in hit])
    missing=int((present.sum(1)<len(classes)).sum());single=int((present.sum(1)==1).sum())
    assert missing==d['replicates_missing_classes'] and single==d['single_class_replicates']
    checks[ds]={'subjects':len(subjects),'classes':len(classes),'windows':int(n.sum()),'paired_denominators_all_nine_cells':True,'draw_hash_match':True,'replicates_missing_classes':missing}
for task,w in weights.items():
    points[task]=sum(v*points[ds] for ds,v in w.items())
    draws[task]=sum(v*draws[ds] for ds,v in w.items())
recomputed={}
for key in points:
    published=r['tasks'][key] if key in weights else r['datasets'][key]
    out={'cells':{},'contrasts':{}}
    for i,name in enumerate(names):
        p=float(points[key][i]);ci=(100*np.quantile(draws[key][:,i],[.025,.975])).tolist()
        error.extend([abs(p-published['cells'][name]['balanced_accuracy'])*100,*np.abs(np.array(ci)-published['cells'][name]['ci95_percent'])])
        out['cells'][name]={'balanced_accuracy_percent':p*100,'ci95_percent':ci}
    for label,(a,b) in pairs.items():
        i,j=names.index(a),names.index(b)
        p=float(100*(points[key][i]-points[key][j]));ci=(100*np.quantile(draws[key][:,i]-draws[key][:,j],[.025,.975])).tolist()
        pub=published['contrasts'][label]
        error.extend([abs(p-pub['delta_pp']),*np.abs(np.array(ci)-pub['ci95_pp'])])
        reading='practical_benefit' if p>=1 and ci[0]>0 else 'practical_deterioration' if p<=-1 and ci[1]<0 else 'below_practical_margin' if abs(p)<1 and (ci[0]>0 or ci[1]<0) else 'inconclusive'
        ni='supported' if ci[0]>-1 else 'loss_beyond_margin' if ci[1]<-1 else 'unresolved'
        assert reading==pub['reading'] and ni==pub['noninferiority_1pp'] and pub['decision_valid'] is True
        out['contrasts'][label]={'delta_pp':p,'ci95_pp':ci,'reading':reading,'noninferiority_1pp':ni,'role':pub['role']}
    recomputed[key]=out
assert max(error)<1e-10
reading={}
for label in pairs:
    c=[recomputed[t]['contrasts'][label] for t in ('p1','p2')]
    gains=[x['delta_pp']>=1 and x['ci95_pp'][0]>0 for x in c]
    losses=[x['delta_pp']<=-1 and x['ci95_pp'][1]<0 for x in c]
    benefit=(gains[0] and c[1]['ci95_pp'][0]>-1) or (gains[1] and c[0]['ci95_pp'][0]>-1)
    status='tradeoff' if any(gains) and any(losses) else 'practical_deterioration' if any(losses) else 'practical_benefit' if benefit else 'small_or_inconclusive'
    reading[label]=status;assert status==r['reading']['contrasts'][label]['status']
a,b,inc=[reading['pool:'+x]=='practical_benefit' for x in ('39237-19618','49046-19618','49046-39237')]
reference=(49046 if b and inc else 39237) if a else (49046 if b else 19618)
assert reference==r['reading']['classification_reference_candidate']==19618
result={'scope':'Independent reaggregation of saved per-subject/class counts embedded in published summary; raw OOF and checkpoint payloads not read; no fit, inference, cluster call or alternate seed',
 'source_summary':str(SUMMARY),'source_summary_sha256':hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
 'classification_artifact_pin':s['reports']['classification']['artifacts']['comparison'],
 'raw_comparison_bytes_independently_verified':False,
 'bootstrap':{'replicates':2000,'seed':20261102,'max_abs_difference_pp':max(error)},
 'support_checks':checks,'tasks':{k:recomputed[k] for k in weights},
 'dataset_primary':{k:recomputed[k]['contrasts']['pool:49046-19618'] for k in checks},
 'dataset_cells':{k:recomputed[k]['cells'] for k in checks},
 'v_r1_reading':reading,'classification_reference_candidate':reference,
 'historical_replica':{k:v['contrast'] for k,v in s['historical_replica']['tasks'].items()},
 'limitations':['single pretraining seed, fixed fitted probes and checkpoints; bootstrap over held subjects only',
  'development datasets and many prespecified comparisons without multiplicity adjustment',
  'P2 averages EPN314subjects and UCI19subjects with equal weight; uncertainty is not proportional to subject count',
  'post/pre equal-width reader accessibility diagnostic; not a test of all information or a new architecture',
  'pool baseline19618 uses pinned historical OOF; compatible numerical context with extra OpenMP runtimes is not exact context equality',
  '39k-to49k combines additional training and scheduled decay; cannot isolate a decay-only effect']}
with OUT.open('x') as stream:
    stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'max_abs_difference_pp':max(error),'support':checks,'reference':reference,'artifact':str(OUT)},indent=2))
