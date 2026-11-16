"""Load the archived proposal for isolated review; no production installation."""
import importlib.util
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
p = Path(__file__).parent / "rank_contrast.py"
spec = importlib.util.spec_from_file_location("wearusfm.model.rank_contrast", p)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
