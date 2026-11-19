# Verifica scettica n. 1 del rilievo P6-N3 (scheda db66b507: regola generale del profilo mancante)

Avvio: 2026-11-19. Copia: clone @ 438c8d65 (sola lettura).

## Progressi (1)
- Riprodotto il grep: db66b507^ riga 20 «Le coppie che richiedono il profilo mancante restano inconclusive.»;
  db66b507 riga 22 «Se manca soltanto il profilo deterministico, restano utilizzabili i confronti completati fra gli altri tre
  profili che soddisfano gli invarianti.» Testo citato corretto.
- db66b507 e' la punta di origin/codex/tokenizer-logcontrast-20261104 e NON e' antenato di 438c8d65 (merge-base --is-ancestor: no).
- orchestrate() @365cb1e2 (blob identico alla punta): break solo per category "failed" (r.445-446); il profilo deterministico con
  tutti i marker "unsupported" diventa "unsupported_deterministic" senza break; status FAILED_REQUIRES_REVIEW per qualunque
  "failed" o disaccordo di rank (r.487-488); INCONCLUSIVE_UNSUPPORTED_DETERMINISM per il solo deterministico non supportato (r.493-494).

## Progressi (2)
- Blob del protocollo: 6149836e in 365cb1e2/35ea94ba/65bc6022/438c8d65 (regola generale presente); 2dfed33b solo in db66b507.
  Driver CUDA 3cf7cc57 identico in tutti. Nella consegna rivista (438c8d65) la frase generale c'e' ancora: il rilievo riguarda
  la preregistrazione della lettura su un altro ramo (codex/tokenizer-logcontrast-20261104), pertinente al punto 6.
- Origine: N-2 di Claude (0afd0864, review r.138-139) chiedeva solo «se il deterministico si ferma su un operatore senza
  implementazione deterministica, gli altri tre restano validi». db66b507 ha SOSTITUITO la frase generale invece di aggiungerla.
- aggregate() r.371-386 calcola solo le coppie fra profili presenti: le coppie col profilo mancante non esistono (frase vecchia
  quasi tautologica). Il contenuto operativo perso e' l'implicazione «le altre coppie restano leggibili» per un fallimento
  NON deterministico.
- Caso (b) del rilievo: la riga 22 di db66b507 vale per ogni causa («memoria, operatore non supportato o problema distribuito»)
  e dice che, mancando solo il deterministico, i confronti degli altri tre restano utilizzabili: con il deterministico "failed"
  (OOM/NCCL -> FAILED_REQUIRES_REVIEW) la regola scritta c'e'. La scheda r.16 copre solo l'operatore non supportato, ma rimanda
  al protocollo (r.3). Il caso (b) NON e' scoperto.
- Caso (a) (bf16_math o fp32_math "failed" -> break): nessuna frase esplicita sulla leggibilita' dei profili completati.

## Progressi (3) - prova del codice
- Probe tmp/p6n3v1/probe_orchestrate.py (orchestrate() del driver 3cf7cc57 con worker finti al posto di torchrun; uscite
  cancellate dopo, clone pulito):
  S1 bf16_math failed -> tentati [bf16_auto, bf16_math], completati [bf16_auto], cross [], FAILED_REQUIRES_REVIEW rc 2
  S2 fp32_math failed -> completati [bf16_auto, bf16_math], cross [bf16_auto_bf16_math], FAILED_REQUIRES_REVIEW rc 2
  S3 deterministico failed (OOM) -> tre completati, cross 2 coppie, FAILED_REQUIRES_REVIEW rc 2
  S4 deterministico unsupported -> tre completati, INCONCLUSIVE_UNSUPPORTED_DETERMINISM rc 2
  S5 tutto ok -> MEASURED_FOR_REVIEW rc 0
- Altri documenti di db66b507 (protocollo r.13/24/55/61-63, scheda r.7/16, prompt 3-launcher r.21, decisioni, registro):
  nessuna regola sulla leggibilita' dei profili completati quando fallisce un profilo non deterministico. r.61-63 (timeout:
  "raccogliere ... profili completati"; "report mancante = cella incompleta") non dice come leggerli.
- Impatto limitato: l'emendamento della consegna (r.49-51) richiede di leggere la diagnostica e poi registrare la risoluzione o un
  emendamento esplicito; nessun codice legge l'esito della diagnostica. La lacuna non puo' far accettare evidenze in automatico.

## Esito
NON CONFUTATO (nel nucleo), gravita' NOTA. Riprodotti testo e comportamento del codice.
- Confermato il caso (a): fallimento "failed" di bf16_math o fp32_math (break) -> nessuna frase scritta su cosa resti leggibile;
  la frase rimossa delimitava l'inconclusivita' alle coppie col profilo mancante.
- Confutato il sotto-caso (b): la riga 22 di db66b507 vale per ogni causa, e memoria/problema distribuito sono proprio i
  fallimenti "failed" -> FAILED_REQUIRES_REVIEW; mancando solo il deterministico, i tre profili restano utilizzabili per regola scritta.
- Imprecisioni: la frase e' stata tolta dal protocollo, non dalla «scheda» (file nuovo in db66b507, mai contenuta); db66b507 non
  e' nella consegna (438c8d65 conserva il blob 6149836e con la frase generale), ma e' la lettura preregistrata della diagnostica
  richiesta dal punto 6 (prompt 3-launcher r.5: leggere il protocollo «nel commit di consegna di questo prompt»).
- Correzione proposta adeguata: rimettere la frase generale accanto a quella sul deterministico, soli documenti, senza nuovo giro.
