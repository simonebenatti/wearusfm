import json,sys
F=sys.argv[1]; out=sys.argv[2]
d=json.load(open(F))
r=d['result']
with open(out,'w') as o:
    for x in r:
        o.write('='*100+'\nPAIR '+x['pair']+'\n')
        for rv in x['reviews']:
            o.write('-'*100+'\nREVIEW dimension=%s point_verdict=%s\n'%(rv['dimension'],rv['point_verdict']))
            o.write('SUMMARY: '+rv['summary']+'\n')
            for f in rv['findings']:
                o.write('\n  FINDING '+json.dumps(f,ensure_ascii=False,indent=2)+'\n')
            for c in rv['confirmations']:
                o.write('\n  CONFIRM '+json.dumps(c,ensure_ascii=False)+'\n')
        vf=x['verify']
        o.write('-'*100+'\nVERIFY pair=%s\n'%vf['pair'])
        for k in ['verdicts','seeded_claims','new_findings','spot_checks','point_verdicts']:
            o.write('\n### '+k+'\n')
            v=vf[k]
            if isinstance(v,list):
                for it in v:
                    o.write('  '+json.dumps(it,ensure_ascii=False,indent=2)+'\n')
            else:
                o.write(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
        o.write('report_path '+vf['report_path']+'\n')
