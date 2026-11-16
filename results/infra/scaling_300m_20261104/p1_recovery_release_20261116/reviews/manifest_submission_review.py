from pathlib import Path
import hashlib,json,subprocess,sys,tempfile,xml.etree.ElementTree as ET,math,datetime
ROOT=Path('/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-p1-recovery');BASE=ROOT/'results/infra/scaling_300m_20261104/p1_recovery_release_20261116'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import scaling_p1_recovery as R
checks=[]
def ck(name,ok):
 checks.append({'name':name,'passed':bool(ok)})
 if not ok:raise AssertionError(name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
manifest=load(BASE/'recovery_manifest.json');budget=load(BASE/'budget.json');cpu=load(BASE/'cpu_tests.json');approval=load(BASE/'criterion_amendment.json');pre=load(BASE/'evidence/preflight/remote.json')
review=load('/private/tmp/p1-criterion-v2-code-review-20261116.json')
ck('manifest_sha',sha(BASE/'recovery_manifest.json')=='1a2104c59e55f5f3c46e03a984bdcda163ef26634c69cc1431ce3319bf878ee3')
remote=Path('/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/scaling_p1_recovery_20261116_v1');pins={}
for name in ['extraction_plan','diagnostic_report','cpu_tests']:
 pin=manifest[name];relative=Path(pin['path']).relative_to(remote);p=ROOT/relative
 ck(name+'_newcheckout',not relative.is_absolute() and '..' not in relative.parts)
 ck(name+'_bytes_digest',p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256'])
 pins[name]=p
ck('original_plan_pin',manifest['extraction_plan']['sha256']=='3cb94b050fa351c01e86184954f542b01b6c939169efc663d869d134e93b4556')
ck('diagnostic_pin',manifest['diagnostic_report']['sha256']==R.RELEASE_DIAGNOSTIC_SHA256=='c2e32e9cfab36a76c79b05913e350b9ec7f15484f73e2a271552c06f6e980908')
diag=load(pins['diagnostic_report']);plan=load(pins['extraction_plan'])
ck('real_manifest_gate',R.validate_manifest(manifest,diag,cpu,R.source_hashes()))
ck('plan_single_endpoint',plan['step']==19618 and set(plan['checkpoints'])=={'T100-G-300M'} and plan['protocol']==R.V.PROTOCOL)
ck('plan_same_original_bytes',subprocess.check_output(['git','show','6ebfaf4:'+str(pins['extraction_plan'].relative_to(ROOT))],cwd=ROOT)==pins['extraction_plan'].read_bytes())
with tempfile.TemporaryDirectory(prefix='p1-release-localpins-',dir='/private/tmp') as tmp:
 local=json.loads(json.dumps(manifest))
 for name,p in pins.items():local[name]['path']=str(p)
 p=Path(tmp)/'recovery.json';p.write_text(json.dumps(local))
 got,states=R.load_manifest(p,sha(p));ck('load_manifest_with_local_path_translation',got==local and len(states)==10)
 # The full executable load path must reject a changed diagnostic byte sequence.
 changed=Path(tmp)/'diag.json';changed.write_bytes(pins['diagnostic_report'].read_bytes()+b'\n')
 local['diagnostic_report']['path']=str(changed);p.write_text(json.dumps(local))
 try:R.load_manifest(p,sha(p))
 except ValueError as exc:ck('changed_report_byte_rejected','pin mismatch' in str(exc))
 else:ck('changed_report_byte_rejected',False)
for name,h in cpu['source_sha256'].items():
 ck('cpu_source_'+name,sha(ROOT/name)==h==review['scope_sha256'][name])
ck('cpu_163',cpu['counts']=={'errors':0,'failures':0,'passed':163,'skipped':0,'tests':163} and cpu['failed']==0 and cpu['exit']==0 and cpu['complete'] is True and cpu['passed'] is True)
ck('cpu_artifact_pins',sha(BASE/'cpu_tests.log')==cpu['log_sha256'] and sha(BASE/'cpu_tests.junit.xml')==cpu['junit_sha256'])
xml=ET.parse(BASE/'cpu_tests.junit.xml');cases=xml.findall('.//testcase');ck('cpu_junit_independent',len(cases)==163 and not xml.findall('.//failure') and not xml.findall('.//error') and not xml.findall('.//skipped'))
contract=load(ROOT/'results/infra/scaling_300m_20261104/training_source_contract.json')
ck('103_science_unchanged',len(contract['scientific_source_sha256'])==103 and all(sha(ROOT/p)==h for p,h in contract['scientific_source_sha256'].items()))
ck('helper_frozen',sha(ROOT/'scripts/scaling_output_identity.py')=='717362c25cb99151ab058b4f212cc73611a36899f67251952ca613a71058e656')
ck('approval_scope',approval['approved'] is True and approval['max_new_extractions']==1 and approval['retry_authorized'] is False and approval['cause_established'] is False and approval['one_logical_extraction_id']==manifest['logical_job_id'] and approval['release_criterion']==manifest['release_criterion'])
ck('approval_report',approval['diagnostic_report_sha256']==manifest['diagnostic_report']['sha256'] and approval['diagnostic_outcome']==manifest['diagnostic_outcome']=='NOT_REPRODUCED')
ck('budget_approval_pin',budget['criterion_amendment_sha256']==sha(BASE/'criterion_amendment.json'))
ck('preflight_all_reads',all(type(c['exit']) is int and c['exit']==0 for c in pre['commands'].values()))
ck('new_paths_absent',all(v['exists'] is False and v['is_symlink'] is False and v['canonical']==k for k,v in pre['new_paths'].items()) and set(pre['new_paths'])=={str(remote),R.OUTPUT})
ck('logical_ids_unused',all(v['unused'] is True for v in pre['local_ledger'].values()) and set(pre['local_ledger'])=={R.LOGICAL_ID,'SCALING-P1-recovery-checkout-20261116'})
ck('new_deploy_ref_absent',pre['commands']['deploy_ref']['stdout']=='')
ck('old_checkout_unchanged',pre['commands']['old_code_head']['stdout'].strip()=='df2de35c05381e7a5a69edde409171b210a3b5d8' and pre['commands']['old_code_status']['stdout']=='')
ck('scheduler_margin','OverTimeLimit           = 3 min' in pre['commands']['config']['stdout'])
ck('failed_receipts_absent',all('receipt.json' not in v for v in pre['failed_directories'].values()))
for key in ('pose_val','p2_epn'):
 entry=next(i for i in diag['manifest']['inputs'] if i['id']==key)
 ck('preserved_'+key,pre['files'][entry['receipt_path']]['sha256']==entry['receipt_sha256'])
rows={p[0]:p for line in (BASE/'evidence/preflight/sacct.txt').read_text().splitlines() if (p:=line.split('|'))}
ck('sacct_raw_matches_capture',(BASE/'evidence/preflight/sacct.txt').read_text()==pre['commands']['sacct']['stdout'])
expect={'64571999':('COMPLETED','0:0'),'64572000':('FAILED','1:0'),'64572001':('COMPLETED','0:0'),'64596915':('FAILED','1:0'),'64610004':('COMPLETED','0:0'),'64610181':('COMPLETED','0:0')}
ck('historical_states',all(tuple(rows[j][1:3])==state for j,state in expect.items()))
stage=sum(int(rows[j][3]) for j in expect if j!='64610004')/3600;p1=(1882+1793)/3600;maxgpu=(7200+180)/3600;checkoutmax=2*(300+180)/3600
ck('budget_stage',math.isclose(budget['stage_gpu_hours_consumed'],stage) and math.isclose(budget['stage_gpu_hours_max'],stage+maxgpu) and stage+maxgpu<=budget['stage_gpu_hours_cap']==7.3)
ck('budget_p1',math.isclose(budget['p1_cumulative_gpu_hours_max'],p1+maxgpu) and p1+maxgpu<=budget['p1_cumulative_gpu_hours_cap']==3.1)
ck('budget_cpu_attached',math.isclose(budget['stage_attached_core_hours_max'],8*(stage+maxgpu)) and budget['stage_attached_core_hours_max']<=budget['stage_attached_core_hours_cap'])
ck('budget_checkout',math.isclose(budget['one_remaining_checkout']['max_core_hours'],checkoutmax))
ck('budget_cpu_only',math.isclose(budget['cpu_only_envelope_max'],budget['cpu_only_prior_envelope']+41*2/3600+checkoutmax) and budget['cpu_only_envelope_max']<=budget['cpu_only_cap']==351)
ck('twohour_allocation',manifest['allocation']=={'nodes':1,'gpus':1,'cpus':8,'memory_gib':100,'walltime_seconds':7200} and manifest['output_barrier']=='none' and manifest['budget']['retry_authorized'] is False)
for file in ('budget.json','criterion_amendment.json','release_plan.md','recovery_manifest.json','cpu_tests.json'):
 ck(file+'_exists',(BASE/file).is_file())
out={'verdict':'APPROVATO manifest concreto e piano di rilascio; nessun blocco','all_checks_passed':True,'checks':checks,'artifact_sha256':{f:sha(BASE/f) for f in ('recovery_manifest.json','release_plan.md','criterion_amendment.json','cpu_tests.json','budget.json')},'source_hashes_match_code_review':True,'test_count':163,'scientific_sources_unchanged':103,'manifest_loading':'Caricamento eseguibile verificato localmente traducendo soltanto i tre prefissi di percorso del checkout remoto a file locali byte-identici; report alterato rifiutato. Nessun execute/inferenza.','accounting_recomputed':{'stage_gpu_hours_max':stage+maxgpu,'p1_gpu_hours_max':p1+maxgpu,'attached_core_hours_max':8*(stage+maxgpu),'cpu_only_max':budget['cpu_only_envelope_max']},'preflight_timestamp':pre['at'],'limits':['Nessun accesso remoto da questo revisore; i dati preflight sono quelli raccolti dal parent','Lo SHA commit del rilascio e i due argv finali sono il passo successivo, ancora da rivedere','La verifica runtime checkpoint/dati avviene nei cancelli originali del job; qui non sono stati caricati i grandi input remoti'],'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=Path('/private/tmp/p1-release-v2-manifest-review-20261116.json');p.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(p);print('checks',len(checks));print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))
