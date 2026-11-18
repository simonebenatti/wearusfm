# Per ogni riferimento di riga della bozza (P:n, J:n, R:n, DEC:n, prompt:n) stampa la riga della fonte, per il controllo a vista.
import re
B = '/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-temponet/docs/reviews/claude_review_pose_temponet_revision_20261118.md'
G = '/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2/gpt_3e007c8/'
F = {'P': G + 'docs/pose_temponet_frozen_protocol_20261117.md',
     'J': G + 'results/infra/pose_temponet_proposal_20261117/proposal.json',
     'R': G + 'docs/reviews/pose_temponet_response_20261118.md',
     'DEC': G + 'docs/decisioni.md',
     'prompt': G + 'docs/prompt_claude_pose_temponet_review_20261117.md'}
L = {k: open(v, encoding='utf-8').read().split('\n') for k, v in F.items()}
seen = {}
for i, l in enumerate(open(B, encoding='utf-8'), 1):
    for m in re.finditer(r'\b(P|J|R|DEC|prompt):(\d+)(?:-(\d+))?', l):
        k, a, b = m.group(1), int(m.group(2)), m.group(3)
        key = (k, a, b)
        seen.setdefault(key, []).append(i)
for (k, a, b), rows in sorted(seen.items(), key=lambda t: (t[0][0], t[0][1])):
    rng = f'{a}-{b}' if b else f'{a}'
    tgt = L[k][a - 1][:150].replace('\n', ' ')
    print(f'{k}:{rng}  [bozza {",".join(map(str, rows))}]  -> {tgt}')
