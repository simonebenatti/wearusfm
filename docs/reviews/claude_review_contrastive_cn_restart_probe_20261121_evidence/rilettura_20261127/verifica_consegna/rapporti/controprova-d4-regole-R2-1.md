# Controprova R2 (worktree/ramo per raccolta e registro)

Esito: il rilievo regge, ma la gravità corretta è «minore».

Fatti verificati:
- Sequenza, riga 190: «Percorso proposto per la raccolta, sul ramo `worktree-claude-training-engine`». Il passo 8 (riga 195-196) dice solo «commit e push» e non nomina né ramo né worktree.
- Brief, riga 22: «Voce nel registro docs/claude_GPT_shared, commit e push», senza ramo né worktree. La riga 12 nomina il ramo solo come posizione della sequenza.
- `git -C .../claude-training-engine branch --show-current` dà worktree-claude-training-engine. `status --porcelain` mostra due voci non tracciate: la sequenza e la cartella prova_a_secco_invii sotto docs/reviews/..._evidence/rilettura_20261127/.
- Il canonico /Users/simonebenatti/dev/wearusfm è su main a c47e6557.

Quello che il rilievo esagera: la sequenza nomina il ramo, quindi non tace del tutto. Il rischio riguarda solo commit e push dopo che i job sono finiti. Seguendo i testi alla lettera nessun tentativo si brucia e il lancio non fallisce. Un commit nel posto sbagliato si può ancora correggere, purché non arrivi un push su main. Per questo la gravità è «minore», o al massimo «importante» se si pesa il rischio di un push su main.
