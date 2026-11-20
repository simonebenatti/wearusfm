#!/bin/bash
set -euo pipefail
hostname
date -Is
lscpu -J
python3 - <<'PYCPU'
import json,os,platform
from pathlib import Path
cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
print(json.dumps({'kind':'serial_cpu_inventory','hostname':platform.node(),'cpu_model':cpu,'machine':platform.machine(),'slurm_job_id':os.environ['SLURM_JOB_ID'],'slurm_job_nodelist':os.environ['SLURM_JOB_NODELIST'],'requested_cpus':os.environ.get('SLURM_CPUS_PER_TASK')}))
PYCPU
