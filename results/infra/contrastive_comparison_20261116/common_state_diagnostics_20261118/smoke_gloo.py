"""Cross-check the new restart observer and reader under four local Gloo ranks."""
from pathlib import Path
import tempfile, sys, json, shutil
from contextlib import nullcontext
from dataclasses import replace
from datetime import timedelta
import torch
root = Path.cwd()
sys.path[:0] = [str(root/'src'), str(root/'scripts/diagnostics'), str(root/'tests/cpu')]
import contrastive_common_checkpoint as ck
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.training import run as tr

def worker(rank, store, where, watch, cfg, manifest, data):
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=4, timeout=timedelta(seconds=50))
    try:
        with ck.trace_updates(where/f"trace-{rank}.jsonl", 4) if watch else nullcontext():
            summary=tr.train(cfg, manifest, [data], where, dist=tr.DistContext(rank,4,rank), log=lambda *_: None)
        assert summary["stopped"]=="max_steps"
    finally:
        torch.distributed.destroy_process_group()

def main():
    torch.set_num_threads(1)
    folder=Path(tempfile.mkdtemp(prefix="common-diag-gloo-"))
    data, manifest, pose=_pose_setup(folder)
    base=_config(pose,"CN")
    base=replace(base,max_steps=2,model=replace(base.model,muscle_dropout=.4), alarm=replace(base.alarm,patience=20), schedule=replace(base.schedule,total_steps=20,keep_checkpoints=()))
    def launch(phase,cfg,watch):
        out=folder/phase
        out.mkdir(exist_ok=True)
        torch.multiprocessing.spawn(worker,args=(folder/(phase+".store"),out,watch,cfg,manifest,data),nprocs=4)
    launch("old",base,False)
    old=folder/"old/checkpoint.pt"
    (folder/"continuous").mkdir()
    ck.diagnostic_seed(old,folder/"continuous/checkpoint.pt",ck.file_sha(old),4)
    cfg=replace(base,max_steps=5,schedule=replace(base.schedule,keep_checkpoints=(4,)))
    launch("continuous",cfg,True)
    for phase in ("reload_a","reload_b"):
        (folder/phase).mkdir()
        shutil.copyfile(folder/"continuous/checkpoint_step0000004.pt",folder/phase/"checkpoint.pt")
        launch(phase,cfg,True)
    analysis=ck.cpu_analysis(folder,4,5,4)
    analysis["ddp_at_split_rank0"] = {
        phase: json.loads((folder/phase/"trace-0.ddp.jsonl").read_text().splitlines()[0])
        for phase in ck.PHASES}
    (folder/"smoke_result.json").write_text(json.dumps(analysis, indent=2)+"\n")
    print("Evidence:",folder/"smoke_result.json")
    assert not analysis["boundary_state_or_rng_mismatches"]
    assert all(not p["input_rng_boundary_differences"] for p in analysis["pairs"].values())

if __name__=="__main__":
    main()
