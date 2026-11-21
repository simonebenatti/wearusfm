import json,sys,re
E=sys.argv[1]
def load(p): return {r['step']:r for r in map(json.loads,open(p))}
pat=re.compile(r'(count|eligible|replaced|negative_age_(0|1_32|33_128|129_256|257_plus|sum)|excluded|negative_current$|negative_current__|same_recording$|same_recording__|same_subject$|same_subject__|queries|discarded|rnc_pairs|rnc_anchors|rnc_windows|rnc_recordings|rnc_subjects|rnc_candidate|duplicate_instants|multiplicity|cross_window_rank_pairs|contrast_windows|^windows$|pose_windows)')
for arm in ['CN','CR']:
  c=load(f'{E}/benchmark_execution_20261121/{arm}/metrics.jsonl'); a=load(f'{E}/reloads_execution_20261121/{arm}_A/metrics.jsonl')
  n=0;bad=[];keysets=0
  for s in sorted(a):
    if set(a[s])!=set(c[s]): keysets+=1
    for k,v in a[s].items():
      if pat.search(k) and not k.endswith('fraction') and 'mean' not in k:
        n+=1
        if c[s].get(k)!=v: bad.append((s,k,c[s].get(k),v))
  print(arm,'count-like comparisons',n,'differing',len(bad),'steps with different key sets',keysets, bad[:5])
  print(arm,'origin equal all steps:',all(a[s]['origin']==c[s]['origin'] for s in a))
