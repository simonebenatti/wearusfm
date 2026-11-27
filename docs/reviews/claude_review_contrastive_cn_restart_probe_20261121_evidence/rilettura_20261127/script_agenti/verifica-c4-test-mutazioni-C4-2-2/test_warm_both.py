import copy, torch
import contrastive_cn_restart_compare as compare
from test_cn_restart_compare import saved_endpoint
def test_dormant_differs_from_both():
    continuous, ordinary = saved_endpoint(24), saved_endpoint(12)
    warm = copy.deepcopy(ordinary)
    warm["cuda_rng"][2] = torch.tensor([99, 5, 9], dtype=torch.uint8)
    r = compare.warm_endpoint_differences(continuous, warm, ordinary["cuda_rng"])
    assert r["scientific_metadata_or_schema_different"] == ["cuda_rng.2"]
    assert not r["exact_scientific_payload"]
