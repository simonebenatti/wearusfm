"""Lens 4: measured neural walltimes are arbitrary seconds; Slurm keeps --time in whole minutes
(sbatch(1): 'Time resolution is one minute and second values are rounded up to the next minute').
Simulate scontrol's TimeLimit for the submitted --time and run runner._allocation on it."""
import os, sys, math
sys.path.insert(0, "tests/cpu")
from test_pose_temponet_resources import receipts
from pose_temponet import resources, execution as E, runner as R
cpu, gpu = receipts()
for head in resources.HEADS:
    for part, c in gpu["cells"][head].items():
        c["train_seconds"] = [.3137]*10; c["eval_seconds"] = [.4519]*10
gate = resources.gate_and_pack(cpu, gpu, work_free_bytes=20*resources.GIB, pending_write_bytes=0,
                               persistent_output_bytes=7*resources.GIB)
jobs = E.apply_training_pack(E.job_graph(), R._training_pack(gate))
neural = [j for j in jobs if j["stage"] in ("controls", "selection", "refit")]
print("gate all_passed:", gate["all_passed"], " neural requested_seconds % 60:", sorted({j["resources"]["requested_seconds"] % 60 for j in neural}))
def slurm_timelimit(seconds):            # sbatch rounds seconds up to the next minute
    m = math.ceil(seconds/60); return "%02d:%02d:00" % (m//60, m%60)
job = neural[0]
sec = job["resources"]["requested_seconds"]
print("job", job["id"], "requested", sec, "s -> --time=%02d:%02d:%02d" % (sec//3600, sec//60%60, sec%60), "-> scontrol TimeLimit", slurm_timelimit(sec))
os.environ.update(SLURM_JOB_ID="123", SLURM_CPUS_PER_TASK="8", SLURM_GPUS_ON_NODE="1", SLURM_NNODES="1",
                  SLURM_MEM_PER_NODE="32768", SLURM_RESTART_COUNT="0")
fields = {"TimeLimit": slurm_timelimit(sec), "Partition": "boost_usr_prod", "NumCPUs": "8", "NumNodes": "1",
          "AllocTRES": "cpu=8,mem=32G,node=1,billing=8,gres/gpu=1"}
R.subprocess.check_output = lambda *a, **k: " ".join(k+"="+v for k, v in fields.items())
R.torch.cuda.device_count = lambda: 1
try:
    R._allocation(job); print("allocation accepted")
except ValueError as e:
    print("runner._allocation raised:", e)
extra = sum(60*math.ceil(j["resources"]["requested_seconds"]/60) - j["resources"]["requested_seconds"] for j in neural)
print("granted-minus-accounted seconds over the 16 neural jobs:", extra, "(max possible 16*59 =", 16*59, "s =", round(16*59/3600,3), "GPUh)")
