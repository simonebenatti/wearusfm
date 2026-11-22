import json, sys
def shape(v, depth=0, maxdepth=4, prefix=""):
    if isinstance(v, dict):
        if depth >= maxdepth:
            print(prefix + "{...%d keys}" % len(v)); return
        for k, x in v.items():
            if isinstance(x, (dict, list)):
                print(prefix + str(k) + ": " + type(x).__name__ + "[%d]" % len(x))
                shape(x, depth+1, maxdepth, prefix + "  ")
            else:
                s = repr(x)
                print(prefix + str(k) + " = " + (s if len(s) < 100 else s[:100] + "..."))
    elif isinstance(v, list):
        if v and isinstance(v[0], (dict, list)) and depth < maxdepth:
            print(prefix + "[0]:")
            shape(v[0], depth+1, maxdepth, prefix + "  ")
        else:
            s = repr(v)
            print(prefix + (s if len(s) < 100 else s[:100] + "..."))
path = sys.argv[1]
maxdepth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
shape(json.load(open(path)), maxdepth=maxdepth)
