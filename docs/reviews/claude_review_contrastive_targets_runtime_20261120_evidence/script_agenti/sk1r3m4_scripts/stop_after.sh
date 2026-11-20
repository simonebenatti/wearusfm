#!/bin/zsh
# Stop harness A (parent pid $1) once its TSV ($2) has the line for mutant $3; restore clone $4.
PARENT=$1; TSV=$2; LAST=$3; CLONE=$4
while ! grep -q "^${LAST}	" "$TSV" 2>/dev/null; do
  if ! kill -0 $PARENT 2>/dev/null; then echo "parent gone"; exit 0; fi
  sleep 5
done
CHILDREN=$(pgrep -P $PARENT)
kill -TERM $PARENT 2>/dev/null
for c in $CHILDREN; do pkill -TERM -P $c 2>/dev/null; kill -TERM $c 2>/dev/null; done
sleep 3
git -C "$CLONE" checkout -q -- .
git -C "$CLONE" status --porcelain --untracked-files=all
echo "stopped after $LAST; clone restored; children=$CHILDREN"
