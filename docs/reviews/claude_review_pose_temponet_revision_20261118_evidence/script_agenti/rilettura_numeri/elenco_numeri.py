# Elenca, riga per riga, i token numerici della bozza (esclusi SHA e riferimenti di riga P:/J:/R:/DEC:), per contarli.
import re
B = '/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-temponet/docs/reviews/claude_review_pose_temponet_revision_20261118.md'
tot = 0
for i, l in enumerate(open(B, encoding='utf-8'), 1):
    s = re.sub(r'`[0-9a-f]{7,40}`', ' ', l)
    s = re.sub(r'\b[0-9a-f]{7,40}\b', ' ', s)
    s = re.sub(r'\b(P|J|R|DEC|prompt):\d+(-\d+)?', ' ', s)
    s = re.sub(r'(controlli|ricetta|criteri|costo|consegna|architettura|selezione)-\d+', ' ', s)
    s = re.sub(r'V-[a-z-]+-\d', ' ', s)
    s = re.sub(r'\b(BL|OB|S|A|C|K|R|N|I)-\d+', ' ', s)
    s = re.sub(r'§\d+(\.\d+)?', ' ', s)
    s = re.sub(r'\bp1-p10\b', ' ', s)
    toks = re.findall(r'[−+-]?\d[\d.,:/]*%?(?:e-?\d+)?', s)
    toks = [t for t in toks if t not in ('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')]
    if toks:
        tot += len(toks)
        print(f'{i:3d}: ' + ' | '.join(toks))
print('TOTALE token numerici:', tot)
