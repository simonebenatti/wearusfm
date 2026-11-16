import copy
from pathlib import Path
import json
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from contrastive_prepare import build_review_configs,BASELINE
from contrastive_run import validate_recipe,validate_release


def recipes():
    return build_review_configs(json.loads(BASELINE.read_text()))


def test_review_configs_cannot_accidentally_be_training_recipes():
    cfg=recipes()
    assert validate_recipe(cfg["calibration"],"calibration").contrastive["calibration_only"]
    for a in ("C0","CN","CR"):
        with pytest.raises(ValueError,match="calibrated"):
            validate_recipe(cfg[a],"training")
    with pytest.raises(ValueError,match="execution release"):
        validate_release({"status":"REVIEW_REQUIRED"},"calibration",Path("not_read"),{})
    with pytest.raises(ValueError,match="measured three-arm benchmark"):
        validate_release({"status":"APPROVED_FOR_TRAINING"},"training",Path("not_read"),{})


@pytest.mark.parametrize("arm",["C0","CN","CR"])
def test_paired_recipes_change_only_arm_and_preserve_measured_both_weights(arm):
    cfg=recipes()[arm]
    report={"status":"PASSED_FOR_REVIEW","weights":{"nce":.123,"rnc":.456}}
    cfg["contrastive"].update(nce_weight=.123,rnc_weight=.456)
    assert validate_recipe(cfg,"training",report).contrastive["arm"]==arm
    for field,value in (("rnc_stride",4),("fill_steps",0),("enforce_coverage",False),("rnc_weight",.45)):
        bad=copy.deepcopy(cfg);bad["contrastive"][field]=value
        with pytest.raises(ValueError):
            validate_recipe(bad,"training",report)
    cfg["max_steps"]=576
    assert validate_recipe(cfg,"benchmark",report).max_steps==576
    cfg["max_steps"]=64
    with pytest.raises(ValueError,match="64 full-weight"):
        validate_recipe(cfg,"benchmark",report)


def test_calibration_cannot_update_or_start_from_new_schedule():
    for key,value in (("arm","CN"),("calibration_only",False),("fill_steps",128)):
        cfg=recipes()["calibration"];cfg["contrastive"][key]=value
        with pytest.raises(ValueError):
            validate_recipe(cfg,"calibration")


def test_execution_release_rejects_changed_inputs_code_or_time(tmp_path, monkeypatch):
    import contrastive_run as runner
    cfg=tmp_path/"cfg.json";cfg.write_text("{}")
    data=tmp_path/"manifest.json";data.write_text("original")
    monkeypatch.setattr(runner,"source_digest",lambda:"source")
    release={"status":"APPROVED_FOR_PREPARATION","stage":"calibration",
             "source_sha256":"source","config_sha256":runner.sha(cfg),
             "world_size":4,"workers_per_rank":6,"inputs":{"manifest":runner.sha(data)},
             "review_pin":"reviewed-commit","time_limit_s":900.}
    runner.validate_release(release,"calibration",cfg,{"manifest":data},time_limit_s=900.)
    for key,value in (("source_sha256","changed"),("config_sha256","changed"),
                      ("world_size",2),("workers_per_rank",0),("review_pin",None),
                      ("time_limit_s",float("nan")),("time_limit_s",float("inf"))):
        bad=copy.deepcopy(release);bad[key]=value
        with pytest.raises(ValueError):
            runner.validate_release(bad,"calibration",cfg,{"manifest":data},time_limit_s=900.)
    data.write_text("changed")
    with pytest.raises(ValueError,match="data pins"):
        runner.validate_release(release,"calibration",cfg,{"manifest":data},time_limit_s=900.)
