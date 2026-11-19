import json, sys
a=json.load(open(sys.argv[1]))
b=json.load(open(sys.argv[2]))
def walk(x, y, path):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in list(x.keys()) + [k for k in y.keys() if k not in x]:
            p = path + '.' + str(k) if path else str(k)
            if k not in y:
                print('REMOVED', p, '=', json.dumps(x[k])[:300])
            elif k not in x:
                print('ADDED  ', p, '=', json.dumps(y[k])[:2000])
            else:
                walk(x[k], y[k], p)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            print('LISTLEN', path, len(x), '->', len(y))
            print('   old:', json.dumps(x)[:1500])
            print('   new:', json.dumps(y)[:1500])
        else:
            for i,(u,v) in enumerate(zip(x,y)):
                walk(u, v, f'{path}[{i}]')
    else:
        if x != y or type(x) != type(y):
            print('CHANGED', path)
            print('   old:', json.dumps(x)[:1500])
            print('   new:', json.dumps(y)[:1500])
walk(a, b, '')
