# Conta le disposizioni di R (risposta di GPT) e le classifica: «Accolta.» semplice o qualificata.
import re
R = '/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/gpt_3e007c8/docs/reviews/pose_temponet_response_20261118.md'
lines = open(R, encoding='utf-8').read().split('\n')
ril, note = [], []
sec = None
for i, l in enumerate(lines, 1):
    if l.startswith('## 1.'): sec = 'ril'
    elif l.startswith('## 2.'): sec = 'note'
    elif l.startswith('## 3.'): sec = None
    if sec and l.startswith('|') and not l.startswith('|---') and not l.startswith('| ID') and not l.startswith('| N.'):
        cells = [c.strip() for c in l.strip('|').split('|')]
        disp = cells[-1]
        m = re.match(r'\*\*(.+?)\*\*', disp)
        head = m.group(1) if m else disp[:40]
        (ril if sec == 'ril' else note).append((i, cells[0][:12], head))
def plain(h): return h.strip() == 'Accolta.'
print('rilievi:', len(ril), ' semplici:', sum(plain(h) for _, _, h in ril), ' qualificati:', [c for _, c, h in ril if not plain(h)])
print('note:', len(note), ' semplici:', sum(plain(h) for _, _, h in note), ' qualificate:', [c for _, c, h in note if not plain(h)])
print('totale disposizioni:', len(ril) + len(note), ' semplici', sum(plain(h) for _, _, h in ril + note), ' qualificate', sum(not plain(h) for _, _, h in ril + note))
for i, c, h in ril + note:
    if not plain(h): print('   R:%d %s -> %s' % (i, c, h))
