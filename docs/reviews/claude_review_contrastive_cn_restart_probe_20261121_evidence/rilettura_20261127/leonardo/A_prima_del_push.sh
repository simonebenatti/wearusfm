# Blocco A: stato del repository canonico prima del push del riferimento di deploy. Sola lettura.
date "+%d/%m/%Y %H:%M:%S"
C=/leonardo_work/IscrB_WearUsFM/wearusfm
echo "ramo estratto: $(git -C "$C" rev-parse --abbrev-ref HEAD)"
echo "HEAD: $(git -C "$C" rev-parse HEAD)"
echo "voci non pulite: $(git -C "$C" status --porcelain --untracked-files=all | wc -l)"
echo "hook non di esempio: $(ls "$C/.git/hooks" | grep -vc '\.sample$')"
echo "receive.denyCurrentBranch: $(git -C "$C" config --get receive.denyCurrentBranch)"
echo "riferimento deploy/cn-restart545-20261121 gia' presente: $(git -C "$C" for-each-ref refs/heads/deploy/cn-restart545-20261121 | wc -l)"
echo "commit di release gia' presente: $(git -C "$C" cat-file -e b58a66f695c8744dfa5fe633829cafe13a5a1fc4^{commit} 2>/dev/null && echo si || echo no)"
