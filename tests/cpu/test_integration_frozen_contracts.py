"""Published freezes remain valid at their SHA and fail on unreviewed source bytes."""
import json
from pathlib import Path
import sys
import pytest
from frozen_campaign_support import run_frozen

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import scaling_train as S
CONTRACT='results/infra/scaling_300m_20261104/training_source_contract.json'


def test_published_scaling_contract_at_its_actual_source_snapshot(tmp_path):
    data=json.loads((ROOT/CONTRACT).read_text())
    paths=['src','scripts',CONTRACT]+[data[k]['path'] for k in ('comparison','config_30m','reference_training_receipt')]
    observed=run_frozen(tmp_path,'97125bd8bb8b14590958501ba91986c8456037f9',paths,
        """import json
import scaling_train as S
from wearusfm.training import run as R
assert Path(R.__file__).resolve().is_relative_to(root)
ctx=S.check_source_contract(root/sys.argv[2],root=root)
print(json.dumps({'sources':len(ctx['contract']['scientific_source_sha256']),
                  'reference':ctx['contract']['reference_code_sha'],
                  'dim':ctx['config_300m'].model.dim,'steps':ctx['config_300m'].max_steps}))
""",CONTRACT)
    assert observed=={'sources':103,'reference':S.BASE_CODE_SHA,'dim':896,'steps':5000}


def test_published_scaling_contract_is_not_silently_rebased_to_merged_code():
    with pytest.raises(ValueError,match='source inventory|artifact SHA|base CLI'):
        S.check_source_contract(ROOT/CONTRACT)
