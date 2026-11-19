import sys, json
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
import contrastive_prepare as P
print(json.dumps({'root': str(root), 'source_digest': P.source_digest()}))
