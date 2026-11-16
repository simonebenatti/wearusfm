"""Offline accounting/evidence review only. No remote operations or payload reads."""
import argparse,csv,datetime,hashlib,io,json,subprocess
from fractions import Fraction
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
r=a.repo
base=r/'results/infra/scaling_300m_20261104'
b=base/'continuation19618_20261106'
f=b/'failure_64572000_diagnosis_20261116'
g=b/'evidence/failure-p1-r1-64596915'
checks=[]
def check(label,ok):
    checks.append({'label':label,'pass':bool(ok)});assert ok,label
def load(path):return json.loads(path.read_text())
def q(n):return {'exact':str(n),'value':float(n)}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
check('review commit',subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip().startswith('c2ae7ee'))
rows=list(csv.DictReader(io.StringIO((f/'readings/sacct_extractions_19618.txt').read_text()),delimiter='|'))
text=(g/'reading_20261116_1606.txt').read_text();table=text[text.index('JobIDRaw|State|ExitCode'):]
rows+= [v for v in csv.DictReader(io.StringIO(table),delimiter='|') if v['JobIDRaw'].isdigit()]
check('four allocations, no batch or extern double count',len(rows)==4 and len({x['JobIDRaw'] for x in rows})==4)
expected={'64571999':('COMPLETED','0:0',7330,'lrdn2203'),'64572000':('FAILED','1:0',1882,'lrdn0248'),'64572001':('COMPLETED','0:0',5967,'lrdn2858'),'64596915':('FAILED','1:0',1793,'lrdn0707')}
for x in rows:
    check('allocation '+x['JobIDRaw'],(x['State'],x['ExitCode'],int(x['ElapsedRaw']),x['NodeList'])==expected[x['JobIDRaw']])
    check('resources '+x['JobIDRaw'],x['AllocCPUS']=='8' and 'gres/gpu=1' in x['AllocTRES'])
    seconds=(datetime.datetime.fromisoformat(x['End'])-datetime.datetime.fromisoformat(x['Start'])).total_seconds()
    check('elapsed independently '+x['JobIDRaw'],seconds==int(x['ElapsedRaw']))
spent=Fraction(sum(int(x['ElapsedRaw']) for x in rows),3600)
p1=Fraction(sum(int(x['ElapsedRaw']) for x in rows if x['JobIDRaw'] in ('64572000','64596915')),3600)
check('stage elapsed16972',spent==Fraction(16972,3600))
check('P1 elapsed3675',p1==Fraction(3675,3600))
slurm=[f/'slurm_logs/scaling_evaluation19618_64572000.txt',g/'slurm_log_64596915.txt']
for path,name in zip(slurm,['ninapro_db6.npz','ninapro_db3.npz']):
    s=path.read_text();check('failure before first output identity '+name,'line 385, in execute' in s and 'before = P.file_identity(path)' in s and 'input changed while hashing:' in s and name in s)
    check('same runtime code '+name,'df2de35c05381e7a5a69edde409171b210a3b5d8' in s)
cache_checks={'pose':{'pass':2,'fail':0,'not_reached':0},'p2':{'pass':2,'fail':0,'not_reached':0},'p1_original':{'pass':2,'fail':1,'not_reached':0},'p1_r1':{'pass':1,'fail':1,'not_reached':1}}
check('seven successes nine reached ten planned',sum(v['pass'] for v in cache_checks.values())==7 and sum(v['fail'] for v in cache_checks.values())==2 and sum(v['not_reached'] for v in cache_checks.values())==1)
freeze=load(base/'continuation19618_budget_freeze_20261105_v2.json')
amend=load(base/'continuation19618_20261111_checkout_cpus/budget_amendment.json')
check('frozen stage7GPU56core',freeze['extractions']['gpu_hours_cap']==7 and freeze['extractions']['allocated_cpu_core_hours_cap']==56)
check('frozen P1 onejob2GPU16core',next(x for x in freeze['extractions']['jobs'] if x['task']=='p1')['walltime_seconds']==7200)
check('amendment freeze identity',amend['freeze']['sha256']==sha(base/'continuation19618_budget_freeze_20261105_v2.json'))
check('amendment base envelope identity',amend['base_envelope']['sha256']==sha(r/amend['base_envelope']['path']))
cpu_envelope=Fraction(amend['cpu_only_core_hours_with_margin']['after']['exact'])
check('CPU prior349.917778 exact',cpu_envelope==Fraction(157463,450))
checkout=Fraction(2*(300+180),3600)
check('CPU next350.184444 exact',cpu_envelope+checkout==Fraction(157583,450))
retry=Fraction(7200+180,3600)
without_diag=spent+retry
with_diag_strict=spent+retry+Fraction(1,2)
with_diag_requested=spent+retry+Fraction(1800+180,3600)
check('proposed6.764444 omits diagnostics',without_diag==Fraction(6088,900))
check('including diagnostics exceeds stage even0.5',with_diag_strict>7 and with_diag_strict>Fraction(715,100))
source_paths=[f/'readings/sacct_extractions_19618.txt',g/'reading_20261116_1606.txt',*slurm,f/'readings/stat_p1_19618.txt',f/'readings/stat_pose_p2_19618.txt',f/'README.md',g/'README.md',base/'continuation19618_20261116_p1_relaunch/README.md',base/'continuation19618_budget_freeze_20261105_v2.json',base/'continuation19618_20261111_checkout_cpus/budget_amendment.json',r/'scripts/scaling_evaluation19618.py',r/'scripts/tokenizer_preflight.py',r/'scripts/channel_readout.py']
result={'reviewed_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'checks':checks,'n_checks':len(checks),'allocations':rows,'spent':{'gpu_hours':q(spent),'allocated_cpu_hours':q(8*spent),'p1_gpu_hours':q(p1),'p1_allocated_cpu_hours':q(8*p1)},'cache_checks':cache_checks,'interpretation_of7of9':'First output-identity outcomes by cache, not independent trials and not every hash call. Each accepted cache is hashed again after validation. One later cache was not reached.','timing_limit':'Times reconstructed from birth/mtime/provenance and end-job, integer-second fields. Not instrumented before/after hash timestamps, not measured flush/durability times.','metadata_limit':'No before/after stat tuple or failed digest persisted. Post-job stat cannot identify which field changed during the failed call.','budget':{'third_p1_nominal_gpu_hours':q(Fraction(2)),'third_p1_with180s_margin':q(retry),'stage_plus_p1_without_diagnostic':q(without_diag),'p1_cumulative_plus_retry_margin':q(p1+retry),'diagnostic_30min_nominal_gpu_hours':q(Fraction(1,2)),'diagnostic_30min_plus180s_margin_gpu_hours':q(Fraction(11,20)),'diagnostic27min_plus180s_margin_gpu_hours':q(Fraction(1,2)),'stage_with_diagnostic_strict0_5_plus_p1':{'gpu':q(with_diag_strict),'allocated_cpu':q(8*with_diag_strict)},'stage_with_diagnostic30min_plus_margin_plus_p1':{'gpu':q(with_diag_requested),'allocated_cpu':q(8*with_diag_requested)},'CPU_only_envelope_before':q(cpu_envelope),'one_additional_checkout':q(checkout),'CPU_only_envelope_after_one_checkout':q(cpu_envelope+checkout),'CPU_over_cap':q(cpu_envelope+checkout-350),'full_reextract3_nominal7_plus_margin0_15_with_diag30min_margin':q(spent+Fraction(715,100)+Fraction(11,20))},'salvage_limits':['No receipt in either failed P1 directory; some per-cache provenance only.','Embedded metadata may support checkpoint/code/input/fold checks, but original raw_before, final checkpoint/source guards and full runtime were not serialized because execute failed before receipt.','A later verification cannot honestly reconstruct contemporaneous historical evidence; fresh recovery attestation in separate output would require explicit protocol amendment and accepted reduced historical guarantee.','Do not write into failed folders or label old jobs COMPLETED. Existing policy forbids adopting their caches.'],'source_pins':{str(x.relative_to(r)):{'bytes':x.stat().st_size,'sha256':sha(x)} for x in source_paths},'remote_operations':False}
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'spent_gpu':float(spent),'with_diagnostic_full_margin':float(with_diag_requested),'cpu_new_envelope':float(cpu_envelope+checkout)}))
