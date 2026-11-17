import json, sys
src, outdir = sys.argv[1], sys.argv[2]
d = json.load(open(src))
for b in d['result']:
    lines = []
    for rv in b['reviews']:
        lines.append('#'*100)
        lines.append(f"REVIEW {rv['key']} verdict={rv['verdict']} report={rv.get('report_path')}")
        lines.append('SUMMARY:\n' + rv['summary'])
        for f in rv['findings']:
            lines.append('-'*80)
            lines.append(f"FINDING {f['id']} [{f['severity']}] {f['title']}")
            for k in ('claim','evidence','how','fix'):
                lines.append(f"  {k.upper()}: {f.get(k)}")
        for i, v in enumerate(rv.get('verified_ok', [])):
            lines.append('-'*80)
            lines.append(f"VOK {rv['key']}#{i+1}: {v.get('claim')}")
            lines.append(f"  HOW: {v.get('how')}")
            lines.append(f"  EVIDENCE: {v.get('evidence')}")
        for i, v in enumerate(rv.get('not_verifiable', [])):
            lines.append(f"NOTVER {rv['key']}#{i+1}: {json.dumps(v, ensure_ascii=False)}")
    v = b['verification']
    lines.append('#'*100)
    lines.append(f"VERIFICATION {b['block']} report={v.get('report_path')}")
    for j in v['judgments']:
        lines.append('-'*80)
        lines.append(f"JUDG {j['finding_id']} outcome={j['outcome']} after={j['severity_after']}")
        for k in ('reasoning','evidence','corrected_statement'):
            lines.append(f"  {k.upper()}: {j.get(k)}")
    for i, a in enumerate(v['verified_ok_attacked']):
        lines.append('-'*80)
        lines.append(f"ATT {b['block']}#{i+1} holds={a['holds']}: {a['claim']}")
        lines.append(f"  EVIDENCE: {a['evidence']}")
    for f in v['new_findings']:
        lines.append('-'*80)
        lines.append(f"NEW {f['id']} [{f['severity']}] {f['title']}")
        for k in ('claim','evidence','how','fix'):
            lines.append(f"  {k.upper()}: {f.get(k)}")
    lines.append('VERDICT_OPINION: ' + str(v['verdict_opinion']))
    open(f"{outdir}/{b['block']}.txt", 'w').write('\n'.join(lines))
print('ok')
