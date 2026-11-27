#!/bin/bash
# ssh finto per la prova a secco: non contatta nessun host. Registra argomenti e stdin, risponde come sbatch --parsable.
n=$(ls "$FINTO_DIR" | grep -c '^chiamata_' || true); n=$((n+1))
printf '%s\n' "$#" > "$FINTO_DIR/chiamata_${n}_numero_argomenti.txt"
for a in "$@"; do printf '%s\n' "$a"; done > "$FINTO_DIR/chiamata_${n}_argomenti.txt"
cat > "$FINTO_DIR/chiamata_${n}_stdin.txt"
echo 99999990$n
