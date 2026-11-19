import json, hashlib, sys
from pathlib import Path
root = Path(sys.argv[1])
def walk(obj, trail=()):
    if isinstance(obj, dict):
        if isinstance(obj.get('path'), str) and any(k in obj for k in ('sha256','canonical_sha256')):
            yield trail, obj
        for k, v in obj.items():
            yield from walk(v, trail + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, trail + (i,))
for name in sys.argv[2:]:
    data = json.loads((root/name).read_text())
    print('==', name)
    for trail, pin in walk(data):
        p = root / pin['path']
        exists = p.exists()
        h = hashlib.sha256(p.read_bytes()).hexdigest() if exists and p.is_file() else None
        want = pin.get('sha256')
        print('/'.join(map(str, trail)), '|', pin['path'], '|', 'exists' if exists else 'MISSING', '|',
              'match' if h == want else ('canonical?' if want is None else 'DIFF'))
