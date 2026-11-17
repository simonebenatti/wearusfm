"""Assemble accepted readout only after all collected evidence passes."""
from pathlib import Path
import hashlib
import json

BASE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    stream=read(BASE/'evidence/payload_stream_verification.json')
    guard=read(BASE/'reviews/guard_review.json')
    lineage=read(BASE/'reviews/provenance_review.json')
    budget=read(BASE/'reviews/budget_review.json')
    receipt=read(BASE/'evidence/p1/receipt.json')
    assert stream['complete'] is True and len(stream['records'])==3
    assert {x['dataset'] for x in stream['records']}==set(receipt['outputs'])
    for row in stream['records']:
        expected=receipt['outputs'][row['dataset']]
        assert row['accepted'] is True and row['exit']==0
        assert row['bytes']==expected['bytes'] and row['sha256']==expected['sha256']
    assert guard['all_checks_passed'] is True
    assert all(c['passed'] for c in lineage['checks'])
    assert budget['failed']==0 and all(c['passed'] for c in budget['checks'])
    assert all(read(BASE/'collection_checks.json').values())
    pins={str(p.relative_to(BASE)):sha(p) for p in sorted(BASE.rglob('*')) if p.is_file() and
          p.suffix in ('.json','.jsonl','.log','.txt') and p.name!='acceptance.json'}
    acceptance=dict(protocol='scaling-p1-r2-acceptance-20261117-v1',accepted=True,job_id='64620434',
        scheduler_state='COMPLETED',exit_code='0:0',code_sha=receipt['code_sha'],
        receipt_sha256=sha(BASE/'evidence/p1/receipt.json'),events_sha256=receipt['output_identity_events_sha256'],
        payload_fresh_full_reads=3,payload_total_bytes=stream['total_bytes'],payload_hash_location='Mac',
        immutable_inputs_and_scientific_source_contract_preserved=True,
        new_guard=dict(normal_full_hashes=6,extra_reads=0,timestamp_transitions=0,recorded_errors=0),
        old_pose_p2_receipts_preserved=True,failed_jobs_remain_failed=True,
        reader_status='EXTRACTIONS_ACCEPTED_FOR_PLAN_RELEASE_AND_FROZEN_VALIDATION',
        ready_for_fit=False,reader_jobs_launched=False,
        remaining=['Connect the controller to published P1-r2 mixed plans; the historical controller still names failed P1-r1.',
                   'Fresh operational checkout/pin/ledger/output/queue/quota gates.',
                   'Frozen complete input validation and train-only benchmarks, then budget/packing before fits.'],
        causal_conclusion='Successful extraction; timestamp-settling branch not exercised; historical cause not proved.',
        evidence_sha256=pins)
    (BASE/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    print('ACCEPTED P1-r2; next stage: publish mixed plans, frozen validation and benchmarks.')

if __name__=='__main__':main()
