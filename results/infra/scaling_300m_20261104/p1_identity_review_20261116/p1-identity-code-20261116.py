"""Offline review of frozen guards; synthetic stat/hash sequences, no cluster IO."""
import argparse, ast, hashlib, json, subprocess
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
root=a.repo
source=(root/'scripts/tokenizer_preflight.py').read_text()
tree=ast.parse(source)
chosen=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('file_identity','check_unchanged')]
assert len(chosen)==2
checks=[]
def require(ok,message):
    if not ok: raise ValueError(message)
def stat(**kw):
    return SimpleNamespace(**dict(dict(st_dev=1,st_ino=7,st_size=100,st_mtime_ns=200,st_ctime_ns=300,st_atime_ns=400),**kw))
class File:
    def __init__(self,states,hashes):self.states=iter(states);self.hashes=iter(hashes)
    def stat(self):return next(self.states)
    def resolve(self):return self
    def __str__(self):return '/synthetic/cache.npz'
ns={'Path':lambda p:p,'R':SimpleNamespace(sha256_file=lambda p:next(p.hashes)),'require':require}
exec(compile(ast.Module(body=chosen,type_ignores=[]),'<frozen guard AST>','exec'),ns)
def guard_case(name,states,hashes,expected):
    try:r=ns['file_identity'](File(states,hashes));got='pass'
    except ValueError:r=None;got='reject'
    assert got==expected,(name,got,expected)
    checks.append({'name':name,'result':got,'expected':expected})
guard_case('same metadata and bytes',[stat(),stat()],['a'*64],'pass')
for field,value in [('st_ctime_ns',301),('st_mtime_ns',201),('st_ino',8),('st_size',99)]:
    guard_case(field+' drift, same bytes',[stat(),stat(**{field:value})],['a'*64],'reject')
guard_case('atime only drift is ignored',[stat(),stat(st_atime_ns=401)],['a'*64],'pass')
guard_case('device only drift is not tracked',[stat(),stat(st_dev=2)],['a'*64],'pass')
# The verbal proposed acceptance needs more than hash1==hash2 and stable pass2.
def naive_recovery(first,second):return first['sha']==second['sha'] and second['before']==second['after']
first={'sha':'a'*64,'before':(1,7,100,200,300),'after':(1,8,100,200,301)}
second={'sha':'a'*64,'before':(1,8,100,200,301),'after':(1,8,100,200,301)}
assert naive_recovery(first,second)
checks.append({'name':'verbal fallback accepts same-content inode replacement','result':'accepted counterexample','expected':'must reject identity replacement'})
contract=json.loads((root/'results/infra/scaling_300m_20261104/training_source_contract.json').read_text())
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert contract['scientific_source_sha256']['scripts/tokenizer_preflight.py']==sha(root/'scripts/tokenizer_preflight.py')
checks.append({'name':'shared helper pinned in accepted scientific training source contract','result':'confirmed'})
paths=['scripts/tokenizer_preflight.py','scripts/scaling_evaluation19618.py','scripts/scaling_train.py','scripts/scaling_plans19618.py','scripts/scaling_derive19618.py','scripts/scaling_probe19618.py','scripts/scaling_pose_readout19618.py','scripts/tokenizer_pose_readout.py','scripts/pose_temporal_readout.py','scripts/channel_readout.py','results/infra/scaling_300m_20261104/training_source_contract.json']
report={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'checks':checks,'source_sha256':{x:sha(root/x) for x in paths},'limits':['Synthetic controlled stat sequences, not a reproduction of Lustre.','No actual historical before/after stat pair is available in failure logs.','No cluster or filesystem performance tests performed.'],'verdict':'Do not weaken shared frozen helper. Measure changed fields; change only output-finalization path after diagnosis, with explicit identity and hash invariants.'}
a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'checks':len(checks),'commit':report['commit']}))
