# Launcher CPU posa: ambiente Python prima del guard

## Difetto e correzione

Il guard dei percorsi usa Path.is_relative_to prima che il launcher originale
carichi moduli e venv. lib_code_dir.sh verifica checkout e percorsi, ma non attiva
l'interprete. L'ambiente iniziale può quindi essere incompatibile con quella API.
La correzione sposta soltanto le quattro righe già esistenti module purge/load e
source del venv prima del primo Python. Guardie, risorse, thread e programma
scientifico restano identici; nessun job fallito o rilanciato per questo difetto.

## Verifica indipendente

Regressione shell funzionale: esegue launcher e guard Path/hash reali, con Python
iniziale simulato incompatibile. Il launcher precedente fallisce con exit 73;
quello corretto passa e raggiunge il lettore con l'ambiente e i thread previsti.
43 test passati sia dall'autore (3,17 s) sia dal critico (3,65 s); bash -n e
diff --check corretti. Review indipendente: APPROVATO, nessun blocco.

Il launcher CPU non è nei 110 sorgenti del benchmark/estrazione né nei cinque
sorgenti numerici dei piani posa. Hash ricontrollati: invariati. Benchmark e cache
restano validi; il fix sarà usato soltanto in un nuovo checkout CPU pubblicato.
I tre job GPU attivi continuano sullo SHA c196323.

## Seguito

Handoff in docs/logsquare_cpu_handoff_20261107.md: 16 job CPU scientifici e due
checkout condivisi, massimo 51 core-ore, zero GPU. Pin delle nove cache da
completare dopo le estrazioni. Nessun job CPU sottomesso da questa consegna.
Gli otto job delle sonde restano sul primo checkout; il secondo serve alle
release dei fit posa e ai confronti. Nessun cambio di ricetta o nuovi esperimenti.
