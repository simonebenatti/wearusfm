# This function is included verbatim by scaling_derive49046.py.
def load_control_endpoint(pin, expected_config, stats_sha256, signature):
    """Certify the preserved consolidation control using its original receipt."""
    path = Path(pin['path'])
    require(path.is_absolute() and path.name == 'checkpoint_postdecay.pt' and
            pin['sha256'] == CONTROL_CHECKPOINT_SHA256 and not (path.parent/'STOP').exists(),
            'wrong preserved control checkpoint')
    states = {'checkpoint':pin_file(pin), 'receipt':pin_file({'path':pin['receipt_path'], 'sha256':pin['receipt_sha256']})}
    receipt = json.loads(Path(states['receipt']['path']).read_text())
    preserved = receipt.get('persistent_checkpoint', {})
    require(receipt.get('protocol') == 'consolidation-t100g-20261104-v1' and receipt.get('complete') is True and
            receipt.get('step') == STEP and receipt.get('checkpoint_sha256') == CONTROL_CHECKPOINT_SHA256 and
            preserved.get('path') == str(path) and preserved.get('sha256') == CONTROL_CHECKPOINT_SHA256 and
            preserved.get('bytes') == states['checkpoint']['bytes'] == 454381728,
            'control preservation receipt differs')
    require(receipt.get('summary',{}).get('stopped') == 'max_steps' and receipt['summary'].get('steps') == STEP,
            'control training did not finish')
    model_recipe(expected_config['model'], ARMS[0])
    state = torch.load(path,map_location='cpu',weights_only=False)
    validate_training_state(state,expected_config,stats_sha256,signature)
    model = FF.load_model(str(path),device='cpu',which='teacher')
    weights = model.state_dict()
    require(canonical(asdict(model.cfg)) == expected_config['model'] and weights.keys() == state['teacher'].keys() and
            all(torch.equal(weights[k],state['teacher'][k]) for k in weights), 'control teacher not restored exactly')
    gain=float(model.tokenizer.fe_gain)
    require(np.isfinite(gain) and gain>0 and torch.equal(model.tokenizer.fe_gain,state['teacher']['tokenizer.fe_gain']),
            'control gain differs')
    config=canonical(asdict(model.cfg))
    del weights,model,state
    gc.collect()
    P.check_unchanged(states)
    return {'states':states,'model_config':config,'fe_gain':gain,'checkpoint_sha256':pin['sha256'],
            'checkpoint_step':STEP,'training_code_sha':receipt['code_sha'],'postload_recipe_passed':True,
            'gain_policy':'exact saved buffer; no evaluation-data calibration'}
