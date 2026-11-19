"""Exact archived evidence can be reread, never relabelled or edited."""
import copy
import gzip
import json
import shutil
from pathlib import Path
import sys
import pytest
from frozen_campaign_support import run_frozen

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import contrastive_prepare as P
from contrastive_source_compat import validate_historical_evidence
BASE=ROOT/'results/infra/contrastive_comparison_20261116'
CAL=BASE/'calibration_completed_20261117'
FROZEN_READER='8905e346b03e5d173cccc124aa40032bf588a191'
FROZEN_PATHS=('src', 'scripts', 'results/infra/contrastive_comparison_20261116',
              'results/infra/tokenizer_execution_20261103/19618/T100-G/config.json')


def test_original_calibration_authenticates_without_relabelling_producer(tmp_path):
    # This release authenticated the old 25/50 calibration using its own exact
    # sources. New target recipes must neither relabel nor grandfather it.
    result = run_frozen(tmp_path, FROZEN_READER, FROZEN_PATHS, """
import json
import contrastive_prepare as P
base=root/'results/infra/contrastive_comparison_20261116'
cal=base/'calibration_completed_20261117'
receipt=json.loads((cal/'calibration_report.json').read_text())
measured=P.authenticate_calibration(receipt,cal/'run')
assert measured['source_sha256']==receipt['source_sha256']!=P.source_digest()
assert measured['reader_provenance']!=receipt['reader_provenance']
assert measured['weights']=={'nce':0.11942686122572264,'rnc':0.33044627364850215}
assert measured['selected_steps']==list(range(257,289))
measured['reader_provenance']=receipt['reader_provenance']
assert measured==receipt
configs=P.calibrated_configs(receipt,cal/'run')
for arm,cfg in configs.items():
    assert cfg==json.loads((base/'calibrated_cohort_20261117'/(arm+'.config.json')).read_text())
print(json.dumps({'unchanged':True,'source_sha256':P.source_digest()}))
""")
    assert result == {'unchanged': True, 'source_sha256':
                      '46938f319181d408ee852cfaacfdc92cc5e44eaa1cfa770825ad96956eec9320'}


def test_unmodified_historical_calibration_does_not_qualify_new_targets():
    receipt=json.loads((CAL/'calibration_report.json').read_text())
    with pytest.raises(ValueError, match='calibration run config differs'):
        P.authenticate_calibration(receipt,CAL/'run')


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
    result = run_frozen(tmp_path, FROZEN_READER, FROZEN_PATHS, """
import gzip,json
from contrastive_source_compat import validate_historical_evidence
base=root/'results/infra/contrastive_comparison_20261116'
path=root/'audit.json'
path.write_bytes(gzip.decompress((base/'audit_completed_20261117/audit.json.gz').read_bytes()))
metadata=validate_historical_evidence('audit',path)
assert metadata['historical_source_sha256']!=metadata['current_source_sha256']
with path.open('a') as f:f.write('\\n')
try:
    validate_historical_evidence('audit',path)
except ValueError:
    rejected=True
else:
    rejected=False
print(json.dumps({'tampered_rejected':rejected}))
""")
    assert result == {'tampered_rejected': True}


def test_historical_audit_does_not_qualify_new_sources(tmp_path):
    path=tmp_path/'audit.json'
    path.write_bytes(gzip.decompress((BASE/'audit_completed_20261117/audit.json.gz').read_bytes()))
    with pytest.raises(ValueError, match="current execution source inventory"):
        validate_historical_evidence('audit',path)
