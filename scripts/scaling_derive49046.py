"""Declared derivation of terminal readers from the accepted 19k numerical chain.

The derivation preserves all numerical algorithms. Only dose, module/protocol
identity and endpoint certification change. The control extraction gets a
separate guard for its original consolidation receipt.
"""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
MODULES=('scaling_evaluation','scaling_probe','scaling_pose_readout','scaling_compare','scaling_reading')
CONTROL_OLD='a31c895061c1f273eaac2d796fc00cf76cffa4318b521a20be4d6ba65b63a056'
CONTROL_NEW='bfa6141fc2ed6505d2f582a46f0c04f36fd35f867e8cff2b11555b241947a490'

def derive(source,name):
    text=source
    for module in MODULES:text=text.replace(module+'19618',module+'49046')
    text=text.replace('19618','49046').replace('19,618','49,046').replace(CONTROL_OLD,CONTROL_NEW)
    if name=='scaling_evaluation':
        text=text.replace('STEP, START = 49046, 5000','STEP, START = 49046, 39237')
        text=text.replace('TRAIN_PROTOCOL = "scaling-continuation-training-20261106-v1"','TRAIN_PROTOCOL = "scaling-continuation49046-20261118-v1"')
        text=text.replace('checkpoint_step0049046.pt','checkpoint_postdecay.pt')
        text=text.replace('reused, never re-extracted.', 'loaded from its preserved native checkpoint; freshly extracted at this endpoint.')
        text=text.replace('the control is not loaded', 'both checkpoint pins are registered')
        text=text.replace('choices=ARMS[1:]','choices=ARMS')
        text=text.replace('require(arm == ARMS[1], "only the 300M endpoint is extracted at 49046; the 30M control is reused")',
                          'require(arm in ARMS, "registered terminal arm required")')
        text=text.replace('require(arm == ARMS[1] and set(plan["checkpoints"]) == {ARMS[1]} and',
                          'require(arm in ARMS and set(plan["checkpoints"]) == set(ARMS) and')
        text=text.replace('    return load_teacher_endpoint(pin, expected_config, arm, stats_sha256, data_signature, source_contract)',
                          '    if arm == ARMS[0]:\n        return load_control_endpoint(pin, expected_config, stats_sha256, data_signature)\n    return load_teacher_endpoint(pin, expected_config, arm, stats_sha256, data_signature, source_contract)')
        old = '                before = P.file_identity(path)'
        new = """                audit = None
                if stage == 'extract-classification' and task == 'p1':
                    from scaling_finalize49046 import certify_output
                    before, details, audit = certify_output(path, lambda q: verify_classification_cache(
                        q, plan['classification_references'][name], name, checkpoint,
                        guard['checkpoint_sha256'], code, states['splits'], arm), out)
                else:
                    before = P.file_identity(path)"""
        text=text.replace(old,new)
        a=text.index('                details = verify_pose_cache',text.index('def execute('))
        b=text.index('                provenance = ',a)
        original=text[a:b]
        text=text[:a]+"                if audit is None:\n"+''.join('    '+line+'\n' for line in original.splitlines())+text[b:]
        text=text.replace('                pp = Path(str(path) + ".provenance.json")',
                          '                if audit is not None: provenance["output_identity_audit"] = audit\n                pp = Path(str(path) + ".provenance.json")')
        text=text.replace('    P.check_unchanged(states)\n    # Revalidate',
                          '                if audit is not None: outputs[name]["output_identity_audit"] = audit\n    P.check_unchanged(states)\n    # Revalidate')
        helper=(ROOT/'scripts/scaling_control49046.py').read_text()
        text=text.replace('def validate_checkpoint(',helper+'\n\ndef validate_checkpoint(',1)
    if name=='scaling_pose_readout':
        text=text.replace('legacy = arm == ARMS[0]  # the 49046 control is always the reused tokenizer-campaign extraction',
                          'legacy = False  # both terminal arms use the same certified extraction pipeline')
    if name=='scaling_probe':
        start=text.index('    if arm == ARMS[0]:  # reused tokenizer-campaign extraction')
        end=text.index('    for k, v in common.items():',start)
        text=text[:start]+text[end:]
    return text

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');a=p.parse_args(argv)
    records={}
    expected=json.loads((ROOT/"results/infra/scaling_49046_20261118/launcher/derivation_sources.json").read_text())
    for name in MODULES:
        src=ROOT/'scripts'/f'{name}19618.py';dst=ROOT/'scripts'/f'{name}49046.py'
        assert hashlib.sha256(src.read_bytes()).hexdigest()==expected[name], "accepted source changed: "+name
        data=derive(src.read_text(),name).encode()
        if a.write:dst.write_bytes(data)
        else:assert dst.read_bytes()==data,dst
        records[str(dst.relative_to(ROOT))]={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'derived_sha256':hashlib.sha256(data).hexdigest()}
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
