# Conta le occorrenze numeriche con significato nella bozza (esclusi SHA, ID dei rilievi, nomi di file, tipi FP16/FP32 ecc.).
# I riferimenti di riga (P:n, J:n, R:n, DEC:n, prompt:n) sono contati a parte.
import re
B = '/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-temponet/docs/reviews/claude_review_pose_temponet_revision_20261118.md'
txt = open(B, encoding='utf-8').read()
refs = re.findall(r'\b(?:P|J|R|DEC|prompt):\d+(?:-\d+)?', txt)
s = txt
for pat in [r'`[^`]*`', r'\b[0-9a-f]{7,40}\b', r'\b(?:P|J|R|DEC|prompt):\d+(?:-\d+)?',
            r'(?:controlli|ricetta|criteri|costo|consegna|architettura|selezione)-\d+', r'V-[a-z-]+-\d',
            r'\b(?:BL|OB|S|A|C|K|R|N|I)-\d+', r'§\d+(?:\.\d+)?', r'\bp1-p10\b', r'\b(?:FP|float)(?:16|32|64)\b',
            r'\bA100\b', r'\bsha256\b', r'\bS[01]\b', r'\bp95\b', r'\bdb8\b', r'\bL(?:0|13)\b', r'\bq_i\b', r'\b20261\d{3}\b',
            r'^\d+\. ', r'^#+ \d+\.', r'\bpunto \d\b', r'\bPunto \d\b', r'^\| \d\. ', r'\bcontrast[oi] \d(?: e \d)?\b',
            r'\betichett[ae] \d\b', r'\bv2\b', r'\bIC(?:99|95)\b', r'\bupper99\b', r'\blower99\b']:
    s = re.sub(pat, ' ', s, flags=re.M)
toks = re.findall(r'[−+]?\d[\d.,:/]*\d%?|[−+]?\d%?', s)
print('occorrenze numeriche con significato:', len(toks))
print('riferimenti di riga:', len(refs))
print('totale controllato:', len(toks) + len(refs))
