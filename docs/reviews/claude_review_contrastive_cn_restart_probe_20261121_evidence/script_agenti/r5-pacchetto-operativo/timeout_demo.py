# Reproduces run()'s try/except around checked_command with a child exceeding the deadline.
import sys, time
sys.path[:0] = ["scripts"]
from contrastive_split_benchmark import checked_command
deadline = time.monotonic() + 1.0
try:
    try:
        checked_command([sys.executable, "-c", "import time; time.sleep(30)"], deadline)
    except (RuntimeError, OSError, ValueError, TimeoutError) as exc:
        print("CAUGHT -> execution_failure.json would be written:", type(exc).__name__)
        raise
except Exception as exc:
    print("NOT CAUGHT by run(): ", type(exc).__name__, "-> no execution_failure.json")
