# Estrae riferimenti P:/J:/R:/DEC:/prompt: e citazioni «» dalla bozza congelata, con il testo della riga citata.
import re
B='/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/controllo/bozza_controllo.md'
G='/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/gpt_3e007c8/'
files={'P':G+'docs/pose_temponet_frozen_protocol_20261117.md','J':G+'results/infra/pose_temponet_proposal_20261117/proposal.json',
       'R':G+'docs/reviews/pose_temponet_response_20261118.md','DEC':G+'docs/decisioni.md','prompt':G+'docs/prompt_claude_pose_temponet_review_20261117.md'}
src={k:open(v).read().split('\n') for k,v in files.items()}
b=open(B).read().split('\n')
# sezioni riscritte: 1-58 (intestazione, Esito, metodo), 60-276 (sez. 1-2), 338-402 (tab. 4-5), 404-459 (6-8), 471-585 (prompt)
ranges=[(1,58),(60,276),(338,402),(404,459),(471,585)]
inr=lambda n: any(a<=n<=z for a,z in ranges)
refs=[];quotes=[]
for i,l in enumerate(b,1):
    if not inr(i): continue
    for m in re.finditer(r'\b(P|J|R|DEC|prompt):(\d+)(?:-(\d+))?',l):
        refs.append((i,m.group(0)))
    for m in re.finditer(r'«([^»]+)»',l):
        quotes.append((i,m.group(1)))
print('riferimenti',len(refs),'citazioni',len(quotes))
seen=set()
for i,r in refs:
    k,rest=r.split(':'); a=int(rest.split('-')[0])
    if r in seen: continue
    seen.add(r)
    print(f"[{i}] {r}: {src[k][a-1][:150]}")
print('---- citazioni: trovate letterali in una fonte?')
allsrc='\n'.join('\n'.join(v) for v in src.values())
extra=open('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/gpt_richiesta_20261118.txt').read()
c15=open('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview/gpt_c15aac9/docs/pose_temponet_frozen_protocol_20261117.md').read()
rev32=open('/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-temponet/docs/reviews/claude_review_pose_temponet_proposal_20261117.md').read()
mem=open('/Users/simonebenatti/.claude/projects/-Users-simonebenatti-dev-wearusfm/memory/feedback_limit_review_rounds.md').read()
def norm(s): return re.sub(r'\s+','',s.replace('`','').replace('*','')).lower()
corpus=norm(allsrc+extra+c15+rev32+mem)
for i,q in quotes:
    ok=norm(q) in corpus
    print(f"[{i}] {'LETTERALE' if ok else 'non trovata':11s} «{q[:110]}»")
