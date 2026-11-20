# Verifica scettica n. 2 — r7-governance G-2 (CLAUDE.md, «questa chat» e precedenza)

Stato: concluso.

## Riproduzione dei fatti (comandi miei)

- `git -C <clone> show fbfde797 -- CLAUDE.md`: il commit aggiunge 4 righe in testa (H1 «# Linea3, incarico del19/11/2026»
  e un paragrafo) sopra «# WearUsFM». Confermato.
- `git -C <clone> show fbfde797:CLAUDE.md | head -3 | xxd`: gli spazi mancano davvero (0x4c696e6561 33 = «Linea3»,
  «del19», «per3-launcher», «Base438c8d65», «review94f50942», «decisionefaa4a8d7», «sacctCOMPLETED0:0», «entro7200s»);
  nessun NBSP o carattere a larghezza zero. Confermato.
- `git -C <clone> show fbfde797:CLAUDE.md | grep -n ...`: «questa chat» a riga 3; le frasi operative non dicono «della
  linea 3»; sotto restano righe 31, 41-50, 60, 62, 88, 251, 288. Confermato.

## Motivi contrari al MINORE (cercati e verificati)

1. **L'ambiguità tolta da d9020c20 riguardava i lanci, non torna.** `git show d9020c20 -- docs/claude_GPT_shared`:
   «La dicitura «questa chat» in CLAUDE.md poteva attribuire i lanci sia a GPT scientifico sia a 3-launcher».
   In fbfde797:CLAUDE.md:3 invii, checkout e controlli Leonardo sono attribuiti per nome ed «esclusivamente a
   **3-launcher**»; «questa chat» resta solo per implementazione e preparazione (non operative), e la costruzione
   «preparazione per 3-launcher» esclude 3-launcher come referente. Stessa attribuzione per nome in:
   prontuario fbfde797:docs/contrastive_targets_execution_20261119.md:7-8 («unico operatore per deployment, invii,
   stato e raccolta su Leonardo»; il prontuario «prevale sui prontuari operativi del 16–18/11»), decisioni.md:8301-8302
   (citazione letterale «Gli invii e i controlli su Leonardo spettano a 3-launcher»), registro («operatore unico
   **3-launcher**») e delivery_pin.json a 0efa4d45 (`"operator": "3-launcher"`).
2. **La precedenza ha un perimetro strutturale.** Il blocco è un H1 a sé, «# Linea3, incarico del19/11/2026», chiuso
   dal successivo H1 «# WearUsFM» (riga 5); «Questo blocco» = righe 1-3, tutte su oggetti della linea 3 (base 438c8d65,
   review 94f50942, decisione faa4a8d7, audit, calibrazione 7200 s). Il rilievo stesso ammette che «il titolo la limita».
   La lettura che darebbe per superate le righe 60/251/288 deve ignorare il titolo: non dimostrata.
3. **Nessun lettore delle linee 1/2 legge oggi questo testo.** Il blocco esiste solo sul ramo
   codex/contrastive-target-windows-20261118 (fbfde797/0efa4d45): assente da origin/main f546c9e4, da
   origin/worktree-claude-training-engine e dal ramo di GPT scientifico (faa4a8d7, db66b507: CLAUDE.md senza il blocco).
   Caveat: merge-base(fbfde797, main) = f546c9e4 e main non ha toccato CLAUDE.md da lì, quindi un'integrazione tal quale
   porterebbe il blocco in testa a main (rischio futuro, documentale).
4. **Nessun effetto sulla catena.** CLAUDE.md non è letto dal codice (git grep in scripts/src/tests: solo commenti) e non
   entra nell'inventario `source_paths()` di fbfde797:scripts/contrastive_release.py:58-64 (src/wearusfm/**/*.py,
   scripts/contrastive_*.py, scripts/slurm/contrastive*.sbatch, train_jepa.py, lib_code_dir.sh): né il binding della
   review né i job dipendono da questo testo; correggerlo non cambia `source_sha256`.
5. **Mandato:** la citazione letterale c'è in decisioni.md:8278-8279 («Ti affido l’implementazione dell’emendamento
   RMS100/fast125 e la preparazione del pacchetto per 3-launcher»); CLAUDE.md ne è una parafrasi fedele.
6. **Spazi:** reali ma tipografici e sistemici nella consegna (anche nel prontuario: «per100/125», «ore08:00»,
   «del17/11 o94f50942», «di121test»).
7. **Precedente del progetto:** discrepanze di CLAUDE.md sull'identità dell'operatore sono state classificate come note:
   2d157cc7 (claude_review_consolidation_20261104.md:410, «N-2 (nota)») e c0a4f58b (§6.4 «Note», N-3: «CLAUDE.md prescrive
   ancora le sottomissioni tramite semplicione ... Non incide sui risultati»), caso più grave di questo.

## Esito

Fatti riprodotti; tesi centrale ridimensionata (l'ambiguità operativa di d9020c20 non torna; la precedenza è delimitata
dal titolo H1). Resta un problema di chiarezza: indicale «questa chat» in un file che d9020c20 considera condiviso fra
chat, mancanza di «della linea 3» nelle frasi, parafrasi senza virgolette, spazi mancanti. Per la scala comune
(«NOTA = chiarezza o documentazione») e per i precedenti N-2/N-3: **NOTA, non MINORE**. refuted=false.
La correzione proposta dal revisore resta utile, specie prima di un'integrazione su main.

Non verificato: se le chat GPT carichino davvero CLAUDE.md come istruzioni (nel repo non c'è AGENTS.md; configurazione
ChatGPT fuori perimetro); se e quando l'integratore porterà il ramo su main tal quale.
