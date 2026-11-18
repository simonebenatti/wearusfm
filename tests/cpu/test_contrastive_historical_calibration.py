"""Exact archived evidence can be reread, never relabelled or edited."""
import copy
import gzip
import json
import shutil
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import contrastive_prepare as P
from contrastive_source_compat import validate_historical_evidence
BASE=ROOT/'results/infra/contrastive_comparison_20261116'
CAL=BASE/'calibration_completed_20261117'


def test_original_calibration_authenticates_without_relabelling_producer():
    receipt=json.loads((CAL/'calibration_report.json').read_text())
    measured=P.authenticate_calibration(receipt,CAL/'run')
    assert measured['source_sha256']==receipt['source_sha256']!=P.source_digest()
    assert measured['reader_provenance']!=receipt['reader_provenance']
    assert measured['weights']=={'nce':0.11942686122572264,'rnc':0.33044627364850215}
    assert measured['selected_steps']==list(range(257,289))
    measured['reader_provenance']=receipt['reader_provenance']
    assert measured==receipt
    configs=P.calibrated_configs(receipt,CAL/'run')
    for arm,cfg in configs.items():
        assert cfg==json.loads((BASE/'calibrated_cohort_20261117'/(arm+'.config.json')).read_text())


@pytest.mark.parametrize('field',['weight','selection','reader','source'])
def test_modified_historical_receipt_is_rejected(field):
    receipt=json.loads((CAL/'calibration_report.json').read_text())
    if field=='weight':receipt['weights']['nce']+=1e-9
    if field=='selection':receipt['selected_steps'][0]+=1
    if field=='reader':receipt['reader_provenance']['source_manifest_sha256']='0'*64
    if field=='source':receipt['source_sha256']='0'*64
    with pytest.raises(ValueError):P.authenticate_calibration(receipt,CAL/'run')


def test_historical_metadata_must_be_byte_exact(tmp_path):
    folder=tmp_path/'run';shutil.copytree(CAL/'run',folder)
    with (folder/'metrics.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError):P.calibration_report(folder)


def test_only_exact_historical_audit_is_compatible(tmp_path):
    path=tmp_path/'audit.json'
    path.write_bytes(gzip.decompress((BASE/'audit_completed_20261117/audit.json.gz').read_bytes()))
    metadata=validate_historical_evidence('audit',path)
    assert metadata['historical_source_sha256']!=metadata['current_source_sha256']
    with path.open('a') as f:f.write('\n')
    with pytest.raises(ValueError):validate_historical_evidence('audit',path)
