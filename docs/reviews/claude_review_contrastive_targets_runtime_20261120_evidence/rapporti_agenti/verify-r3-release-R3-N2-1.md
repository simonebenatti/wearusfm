# verify-r3-release-R3-N2-1 — scettico n. 1 su R3-N2 (read_text di locale nel validate)

Avvio 2026-11-20 04:00. Clone temporaneo rev_runtime/tmp/vr3n2_1/clone a 0efa4d45 (--shared), poi commit temporaneo 13d4ef5e con una review UTF-8 con accenti (14 byte non ASCII) e blocco valido (hash ricalcolati: source 40d39f56...cde0, plan b03c0bc6...021a, preflight e4b39591...6dd92).

## Riproduzione (CLI bind-review, macOS, Python 3.11.14 del venv)
- LC_ALL=C: utf8_mode=1, rc 0
- LC_ALL=en_US.US-ASCII: utf8_mode=0 pref=US-ASCII, rc 1, UnicodeDecodeError 'ascii' 0xc3; traceback: main:275 -> bind_review:151 -> validate_review_binding:116 (review.read_text()) -> pathlib.read_text
- LANG=en_US.US-ASCII (senza LC_ALL): rc 1, stesso errore
- LC_ALL=en_US.UTF-8: rc 0; LC_ALL=en_US.ISO8859-1 (Latin-1, mojibake ma blocco ASCII): rc 0
- LC_ALL=xx_XX.US-ASCII (nome inesistente, come en_US.US-ASCII su glibc): setlocale fallisce -> C -> utf8_mode=1, rc 0
- LC_ALL=en_US.US-ASCII PYTHONUTF8=1: rc 0
- check-binding (stessa funzione di contrastive_job.py:70 e contrastive_run.py:182): C rc 0, en_US.US-ASCII rc 1 (stesso errore), xx_XX.US-ASCII rc 0

## Verifiche a fbfde797 (= 0efa4d45 per questo file)
- contrastive_release.py:116 `declared = review_payload(review.read_text())` (codifica di locale); :148 `review_payload(original.decode())` (UTF-8 stretto); :151 bind_review richiama validate_review_binding. Chiamanti sul cluster: contrastive_job.py:70 (dispatcher), contrastive_run.py:182 (runtime). Fatto confermato.
- Lato sicuro: ogni discordanza di decodifica e' respinta (payload del binding == payload ricalcolato, pin a byte della copia); nessuna review o evidenza invalida accettata. Solo disponibilita'.
- Gli altri file letti con read_text nel cancello (piani, binding, job, TSV sacct) sono ASCII (plan 0, preflight 0, delivery_pin 0 byte non ASCII; JSON con ensure_ascii): la copia della review e' l'unico file non ASCII.
- La review precedente di Claude (94f50942, docs/reviews/claude_review_contrastive_targets_amendment_20261119.md) ha 87 byte non ASCII: accenti realistici.
- Test: tests/cpu/test_contrastive_release.py 22 passed sia con LC_ALL=C sia con LC_ALL=en_US.US-ASCII: nessun test con review non ASCII (lacuna di copertura coerente col rilievo).
- Nessun PYTHONUTF8/PYTHONIOENCODING/LANG/LC_* nei launcher contrastivi, in lib_code_dir.sh, in submit_job.py (git grep su scripts e src).

## Leonardo (evidenze indirette, nessun accesso)
- Job 62618177 inviato da agente gpt con submit_job.py e --export=ALL (results/infra/coordinamento_claude/ledger_copia.tsv:739), launcher pose_b19618_eval.sbatch con la stessa pila (module purge; profile/deeplrn; cineca-ai/4.3.0; venv_wearusfm; nessun override di codifica): print() di Python (channel_readout.py:555-556) ha scritto "±" come C2 B1 (UTF-8) nel log; sha 8d5bfe40...29ac = pin del percorso Leonardo in results/infra/pilot_posa/b19618_20261029/final/files.tsv:60. Con codifica ASCII print() su file solleverebbe UnicodeEncodeError, con Latin-1 sarebbe un byte B1: in quel job la codifica di Python era UTF-8.
- en_US.US-ASCII e' un locale macOS; su glibc i soli codeset ASCII sono C/POSIX (-> UTF-8 mode di Python 3.11) e un nome inesistente ricade su C (emulato sopra con xx_XX.US-ASCII: rc 0). Il guasto su Leonardo richiederebbe un locale esotico (multibyte non UTF-8) o PYTHONUTF8=0. Non verificato direttamente.
- Sul Mac lo stesso validate gira gia' dentro bind-review: un locale ASCII dell'operatore emergerebbe prima di qualunque commit o job.

## Esito
Rilievo riprodotto (fatto di codice e guasto con locale ASCII su macOS): NON confutato. "Farebbe fallire allo stesso modo ogni job" e' sovrastimato per Leonardo: con la stessa impostazione su glibc non fallisce, e l'unico job GPT con la stessa pila mostra I/O UTF-8. Gravita' reale: NOTA (fail-closed, nessuna misura falsata, guasto non certo). Rimedio: read_bytes().decode("utf-8") o read_text(encoding="utf-8") al prossimo giro (cambia source_digest: nuova review); senza toccare l'inventario, PYTHONUTF8=1 nell'--export dell'invio o review solo ASCII.
Chiusura 04:12: clone temporaneo e basetemp cancellati; conservati vr3n2_1/scripts e vr3n2_1/out; copia di riferimento a 0efa4d45, status pulito.
