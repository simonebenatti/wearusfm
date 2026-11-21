import sys, time, subprocess
sys.path[:0]=[sys.argv[1]+"/scripts", sys.argv[1]+"/src"]
from contrastive_split_benchmark import checked_command  # codice vero
deadline=time.monotonic()+1
try:
    try:
        checked_command([sys.executable,"-c","import time;time.sleep(30)"], deadline)
    except (RuntimeError, OSError, ValueError, TimeoutError) as exc:   # tupla di run() riga 174
        print("CAUGHT", type(exc).__name__)
except subprocess.TimeoutExpired as e:
    print("ESCAPED run() except:", type(e).__name__)
