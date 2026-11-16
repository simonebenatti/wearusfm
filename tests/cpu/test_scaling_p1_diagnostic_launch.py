import hashlib
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('p1launch',ROOT/'scripts/scaling_p1_diagnostic_launch.py')
L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)

@pytest.fixture
def args(tmp_path):
    work=tmp_path/'work';code=work/'wearusfm_campaigns/test';code.mkdir(parents=True)
    (code/'plan.json').write_bytes(b'{}')
    return dict(code=code,manifest='plan.json',manifest_sha256=hashlib.sha256(b'{}').hexdigest(),
                out=tmp_path/'diag/new',job='123',work=work,diag=tmp_path/'diag',
                job_info='JobId=123 JobState=RUNNING NumNodes=1 NumCPUs=8 NumTasks=1 Partition=boost_usr_prod QOS=boost_qos_dbg TimeLimit=00:27:00 AllocTRES=cpu=8,mem=32G,node=1,gres/gpu=1')

def test_allocation_and_manifest(args):assert L.validate(**args)

@pytest.mark.parametrize('old,new',[('JobId=123','JobId=456'),('RUNNING','PENDING'),('NumCPUs=8','NumCPUs=16'),
 ('gres/gpu=1','gres/gpu=2'),('mem=32G','mem=64G'),('00:27:00','00:30:00'),('boost_qos_dbg','normal'),('NumNodes=1','NumNodes=2')])
def test_wrong_allocation(args,old,new):
    args['job_info']=args['job_info'].replace(old,new)
    with pytest.raises(ValueError):L.validate(**args)

def test_pin_and_new_output(args):
    original=args['manifest_sha256'];args['manifest_sha256']='0'*64
    with pytest.raises(ValueError):L.validate(**args)
    args['manifest_sha256']=original;args['out'].mkdir(parents=True)
    with pytest.raises(ValueError):L.validate(**args)

def test_escape_and_symlink(args):
    outside=args['work']/'external.json';outside.write_bytes(b'{}')
    (args['code']/'link.json').symlink_to(outside);args['manifest']='link.json'
    with pytest.raises(ValueError):L.validate(**args)
