import json,sys,re,statistics as st
d=json.load(open(sys.argv[1]))
out={}
for arm in ['CN','CR']:
  p=d['arms'][arm]['resume']['pairs']['continuous__reload_a']
  f=p['floating_tensor_relative_to_left']
  kinds={}
  for k,v in f.items():
    m=re.match(r'optimizer\.state\.(\d+)\.(\w+)$',k)
    kind=m.group(2) if m else k.split('.')[0]+':'+k.split('.')[-1] if not k.startswith('optimizer') else k
    kinds.setdefault(kind,[]).append((k,v))
  print('==',arm,len(f))
  for kind,l in kinds.items():
    vals=[v for _,v in l]; nz=[v for v in vals if v>0]
    print(f'  {kind:30s} n={len(vals)} nz={len(nz)}', 'med=%.4g max=%.4g'%(st.median(nz),max(nz)) if nz else '')
  ea={int(re.match(r'optimizer\.state\.(\d+)\.',k).group(1)):v for k,v in f.items() if k.endswith('.exp_avg')}
  sq={int(re.match(r'optimizer\.state\.(\d+)\.',k).group(1)):v for k,v in f.items() if k.endswith('.exp_avg_sq')}
  out[arm]=(ea,sq)
  top=sorted(ea.items(),key=lambda x:-x[1])[:15]
  print('  top exp_avg', [(i,round(v*100,3)) for i,v in top])
  print('  max id', max(ea))
ea_n,sq_n=out['CN']; ea_r,sq_r=out['CR']
common=sorted(set(ea_n)&set(ea_r))
ratios=[ea_n[i]/ea_r[i] for i in common if ea_r[i]>0 and ea_n[i]>0]
print('common',len(common),'ratio CN/CR exp_avg median %.3f q10 %.3f q90 %.3f'%(st.median(ratios),sorted(ratios)[len(ratios)//10],sorted(ratios)[9*len(ratios)//10]))
r2=[ea_n[i]/ea_r[i] for i in common if ea_r[i]>0 and ea_n[i]>0 and i<266]
print('ids<266 ratio median %.3f'%st.median(r2), 'n',len(r2))
for arm,(ea,sq) in out.items():
  rr=[ea[i]/sq[i] for i in ea if sq.get(i,0)>0 and ea[i]>0]
  print(arm,'exp_avg/exp_avg_sq ratio median %.2f'%st.median(rr))
  print(arm,'exp_avg ids<266 med %.4g max %.4g; sq med %.4g'%(st.median([ea[i] for i in ea if i<266 and ea[i]>0]),max(ea[i] for i in ea if i<266),st.median([sq[i] for i in sq if i<266 and sq[i]>0])))
  for i in range(260,max(ea)+1): print('   ',arm,i,'%.4f%% sq %.4f%%'%(ea[i]*100,sq.get(i,0)*100))
