"""Independent check of L4-1 with the UNMODIFIED resources test fixture.

Slurm stores job time_limit in minutes (time_str2mins rounds seconds up) and
scontrol prints TimeLimit via mins2time_str, so seconds are always ':00'.
"""
import math, os, sys
sys.path.insert(0, "tests/cpu")
from test_pose_temponet_resources import receipts
from pose_temponet import resources, execution as E, runner as R

cpu, gpu = receipts()
gate = resources.gate_and_pack(cpu, gpu, work_free_bytes=12*resources.GIB, pending_write_bytes=0,
                               persistent_output_bytes=7*resources.GIB)
jobs = E.apply_training_pack(E.job_graph(), R._training_pack(gate))
print("gate all_passed:", gate["all_passed"])
for j in jobs:
    s = j["resources"]["requested_seconds"]
    print(f'{j["stage"]:17s} {str(j["head"]):9s} {str(j["world"]):9s} req={s:6d}  s%60={s%60:2d}')

def slurm(seconds):
    m = (seconds + 59)//60              # time_str2mins
    return "%02d:%02d:00" % (m//60, m%60)  # mins2time_str for < 1 day

os.environ.update(SLURM_JOB_ID="1", SLURM_CPUS_PER_TASK="8", SLURM_GPUS_ON_NODE="1", SLURM_NNODES="1",
                  SLURM_MEM_PER_NODE="32768", SLURM_RESTART_COUNT="0")
R.torch.cuda.device_count = lambda: 1
fails = ok = 0
for j in jobs:
    if not j["resources"]["gpus"]:
        continue
    fields = {"TimeLimit": slurm(j["resources"]["requested_seconds"]), "Partition": "boost_usr_prod",
              "NumCPUs": "8", "NumNodes": "1", "AllocTRES": "cpu=8,mem=32G,node=1,billing=8,gres/gpu=1"}
    R.subprocess.check_output = lambda *a, _f=fields, **k: " ".join(k2+"="+v for k2, v in _f.items())
    try:
        R._allocation(j); ok += 1
    except ValueError as e:
        fails += 1
        last = (j["id"], j["resources"]["requested_seconds"], fields["TimeLimit"], str(e))
print("GPU jobs accepted:", ok, "rejected:", fails, "example:", last)
neural = [j for j in jobs if j["stage"] in ("controls", "selection", "refit")]
extra = sum(60*math.ceil(j["resources"]["requested_seconds"]/60) - j["resources"]["requested_seconds"] for j in neural)
print("granted-minus-charged seconds (16 neural):", extra)
