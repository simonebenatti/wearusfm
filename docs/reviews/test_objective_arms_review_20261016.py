"""Review indipendente W1-OBJ: equivalenza dei bracci con riduzioni distribuite.

Esecuzione CPU: PYTHONPATH=src:tests/cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 python -m pytest -q docs/reviews/test_objective_arms_review_20261016.py
"""

from dataclasses import replace

import pytest
import torch

import test_ddp_equivalence as E


@pytest.mark.parametrize("jepa_weight,anchor_weight", [(0.0, 0.2), (1.0, 0.0)], ids=["anchors_only", "jepa_only"])
def test_each_arm_matches_global_batch_with_accumulation_and_ddp(tmp_path, jepa_weight, anchor_weight):
    cfg = E._config(True, True)
    cfg = replace(cfg, jepa=replace(cfg.jepa, jepa_weight=jepa_weight, anchor_weight=anchor_weight))
    batch = E._global_batch(tmp_path, cfg)
    torch.manual_seed(17)
    init = E.WearUsFM(cfg.model).double().state_dict()
    reference = E._reference(cfg, init, batch)
    E._same(reference, E._update(cfg, init, batch, 2, E.R.DistContext(), lambda s: None), "1 processo x 2")
    distributed = E._ddp(tmp_path, "review", cfg, init, batch, 2, (1, 2))
    for rank, results in enumerate(distributed):
        for accumulation, result in results.items():
            # Loss, gradienti, pesi dopo clipping/AdamW e teacher dopo EMA: 1e-10.
            # Il batch ha conteggi eterogenei e RMS senza target sul processo 0.
            E._same(reference, result, f"rank {rank}, 2 processi x {accumulation}")


@pytest.mark.parametrize("jepa_weight,anchor_weight", [(0.0, 0.2), (1.0, 0.0)], ids=["anchors_only", "jepa_only"])
def test_resume_cannot_change_either_arm_weight(jepa_weight, anchor_weight):
    cfg = E._config(True, True)
    cfg = replace(cfg, jepa=replace(cfg.jepa, jepa_weight=jepa_weight, anchor_weight=anchor_weight))
    state = {"config": E.R.config_to_dict(cfg)}
    E.R._check_resume_recipe(cfg, state)
    for field in ("jepa_weight", "anchor_weight"):
        changed = replace(cfg, jepa=replace(cfg.jepa, **{field: getattr(cfg.jepa, field) + 0.1}))
        with pytest.raises(ValueError, match="jepa"):
            E.R._check_resume_recipe(changed, state)
