import json,sys,re,math,statistics as st
d=json.load(open(sys.argv[1]))
def get(arm,kind):
  f=d['arms'][arm]['resume']['pairs']['continuous__reload_a']['floating_tensor_relative_to_left']
  return {int(re.match(r'optimizer\.state\.(\d+)\.',k).group(1)):v for k,v in f.items() if k.endswith('.'+kind)}
def rank(x):
  s=sorted(range(len(x)),key=lambda i:x[i]); r=[0]*len(x)
  for j,i in enumerate(s): r[i]=j
  return r
def pear(a,b):
  ma,mb=st.mean(a),st.mean(b); return sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
for kind in ['exp_avg','exp_avg_sq']:
  n=get('CN',kind); r=get('CR',kind); ids=[i for i in range(266) if r[i]>0 and n[i]>0]
  a=[math.log(n[i]) for i in ids]; b=[math.log(r[i]) for i in ids]
  print(kind,'n',len(ids),'pearson log %.3f'%pear(a,b),'spearman %.3f'%pear(rank(a),rank(b)))
  # sq/ea ratio distribution
ea=get('CN','exp_avg'); sq=get('CN','exp_avg_sq')
rat=sorted(sq[i]/ea[i] for i in range(266)); print('CN sq/ea encoder: median %.4f q05 %.4f q95 %.4f max %.4f'%(st.median(rat),rat[13],rat[252],rat[-1]))
print('CN sq/ea proj', [round(sq[i]/ea[i],4) for i in range(266,270)])
eaR=get('CR','exp_avg'); sqR=get('CR','exp_avg_sq')
ratR=sorted(sqR[i]/eaR[i] for i in range(266)); print('CR sq/ea encoder: median %.4f q95 %.4f max %.4f'%(st.median(ratR),ratR[252],ratR[-1]))
# periodicity: top CN ids mod 18
top=sorted(range(266),key=lambda i:-ea[i])[:30]; print('CN top30 ids',top); print('mod18',sorted(i%18 for i in top))
topR=sorted(range(266),key=lambda i:-eaR[i])[:30]; print('CR top30 ids',topR)
print('CN rank of projector among 270:',sorted(range(270),key=lambda i:-ea[i]).index(266))
# W vs b ratio
print('CN W1/b1 %.3f W2/b2 %.3f'%(ea[266]/ea[267],ea[268]/ea[269]))
# key errors
for arm in ['CN','CR']:
  f=d['arms'][arm]['resume']['pairs']['continuous__reload_a']['floating_tensor_relative_to_left']
  ks=[v for k,v in f.items() if not k.startswith('optimizer')]
  print(arm,'queue keys n',len(ks),'nz',sum(v>0 for v in ks),'median nz %.4f p90 %.4f max %.4f'%(st.median([v for v in ks if v>0]),sorted([v for v in ks if v>0])[int(.9*sum(v>0 for v in ks))],max(ks)))
  print(' sample key names', [k for k in f if not k.startswith('optimizer')][:3])
