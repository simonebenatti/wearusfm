import sys, time, subprocess, json
sys.path[:0]=["src","scripts","scripts/diagnostics"]
from contrastive_split_benchmark import checked_command
caught=None
t=time.monotonic()
try:
    try:
        checked_command([sys.executable,"-c","import time; time.sleep(30)"], time.monotonic()+1)
    except (RuntimeError, OSError, ValueError, TimeoutError) as exc:  # tuple copied from contrastive_cn_restart.py:174
        caught=("caught", type(exc).__name__)
        raise
except BaseException as e:
    print("propagated:", type(e).__name__, "caught_by_run_except:", caught, "elapsed %.1fs"%(time.monotonic()-t))
    print("MRO:", [c.__name__ for c in type(e).__mro__])
# pre-phase exhaustion path
try:
    checked_command(["true"], time.monotonic()-1)
except TimeoutError as e: print("pre-phase path is TimeoutError (caught):", e)
