# Prompt per ChatGPT: lettura dell'emendamento 1 a `pilot_posa_v1` come responsabile scientifico

Preparato da Claude il 26/10/2026 su richiesta di Simone («Passa l'emendamento 1 a GPT come responsabile scientifico; firmo dopo la sua
lettura»). Simone lo incolla a ChatGPT e riporta la risposta; Claude risponde nel registro. Il testo da incollare comincia sotto la riga.

---

Sei il responsabile scientifico del progetto WearUsFM dal 26/10/2026 (decisione di Simone; ruoli in CLAUDE.md, «Ruoli dal 26/10/2026»).
Simone ha letto la tua revisione del 26/10 (`proposta_scientifica.md`) e ha chiesto a Claude di chiudere le attivita' avviate, aggiornare la
diagnosi e scrivere un **emendamento esplicito** al protocollo R3 gia' firmato, senza cambiarne le regole. Ora chiede a te di leggerlo: **Simone
firma dopo la tua lettura.** Nessun job e' stato lanciato; nessuno va lanciato con questa lettura.

## Da leggere, nell'ordine (ramo `worktree-claude-training-engine`, commit `ef9aeb0`, su GitHub: `git fetch origin worktree-claude-training-engine`)

1. `docs/emendamento_pilot_posa_v1.md`: l'emendamento (E0-E8). Aggiunge, senza toccare il criterio §5 ne' le soglie firmate: E1 posa fuori
   campione sui 15 utenti dello split `val` di emg2pose contro il predittore costante (regola congelata: rapporto MAE_R3 / MAE_costante sulle
   finestre dinamiche, limite superiore dell'IC95 sui 15 utenti < 0,9 e mediana per utente < 0,9); E2 controllo con EMG disallineato in
   valutazione (stesso utente e fase, >= 8 s; limite inferiore dell'IC95 del rapporto col costante > 0,95, altrimenti (a) non interpretabile);
   E3 sonda ridge lineare per istante dai latenti congelati su R3, R3c, R0; E4 norme e coseno dei gradienti posa/ancore sui moduli condivisi;
   E5 adozione (R3 contro R0, §5) separata dall'attribuzione (R3 contro R3c), con R3c nella conferma a due semi; E6 albero delle decisioni
   congelato; E7 costi (+1,0 / 4,0 GPU-ora; totale ~99 / 136).
2. `docs/diagnosi_aggiornata_20261026.md`: la tabella per ipotesi H1-H9, i tuoi sette rilievi verificati (tutti concordati, con le correzioni di
   formulazione alla diagnosi di Claude), e le precisazioni alla tua revisione (CPEP/RandInit confermato dal fact-checker; il difetto di
   `key_time_valid` che avevi visto e' stato corretto; `PoseHead` = LayerNorm + Linear dichiarato; cautela sulla prova `[pool; dec]`).
3. `docs/claude_GPT_shared`, testata (stato dei sei punti al 26/10) e voci di Claude del 26/10: in particolare la voce «codice (2-4/4) e REVIEW
   DEL CRITICO SUL CODICE» (7 bloccanti e 8 minori, corretti in d8650c6) e le «precisazioni d'implementazione» in `docs/decisioni.md`.
4. `docs/pilot_posa_v1.md` (testata: firmato il 26/10; «Stato del codice» e «Precisazioni» in fondo al §6) per vedere che cosa resta invariato.

## Che cosa ti chiede Simone

**A. Verdetto sull'emendamento, punto per punto (E1-E7):** confermi, modifichi o togli? Per ogni modifica scrivi la regola alternativa come
«proposta» con la ragione (le soglie 0,9 / 0,95 / 1% sono di Claude, dichiarate prima di guardare; se ne preferisci altre, dillo ora: dopo i
risultati non si toccano). In particolare: (i) le finestre «dinamiche» (varianza della posa sopra la mediana di `val`) come insieme della regola
E1 sono la scelta giusta, o vuoi la regola su tutte le finestre? (ii) il controllo E2 appaiato per utente e fase a >= 8 s basta a cambiare il
movimento, o vuoi un appaiamento diverso (altra sessione, altro utente con la stessa fase)? (iii) E3 con 200 registrazioni e 8 istanti per
finestra e' sufficiente? (iv) l'albero E6 copre i casi che ti interessano?

**B. R3 ereditato:** confermi R3 + R3c cosi' come sono firmati (ancore 0,2 + testa di posa lineare a zero, peso 1,0, senza JEPA, 30M, 2 epoche,
controllo ancore-only), o proponi modifiche PRIMA del lancio (anche queste come proposta per la firma di Simone)? Se proponi di accantonarlo,
di' quale esperimento metti al suo posto, con criterio e costo.

**C. Ordine dei passi e via:** l'ordine previsto e' codice dell'emendamento (`scripts/pose_eval.py`, `--pose-grad-every`, ~1 giorno) -> review ->
deploy e checkout -> collaudo dell'ingest su 40 registrazioni (poi la regola di validita' dei campioni da firmare) -> ingest intero -> collaudo dbg
di R3 -> training R3 e R3c -> valutazione. Ogni job col via di Simone. Confermi l'ordine? Vuoi essere tu a chiedere i via a Simone, o lo fa
Claude per la parte tecnica col tuo benestare scientifico registrato?

**D. Integrazione:** `main` e' fermo a 4680617 (22/10). Le consegne di Claude del 23-26/10 (readout per canale, pilot, R3 fino a ef9aeb0) sono
sul ramo, pubblicate e coperte dalla suite (1.430 test prima delle correzioni della posa; suite rilanciata dopo). La review del codice della
posa e l'integrazione su `main` restano tue: dimmi se le fai prima o dopo il lancio (il lancio parte da un checkout di campagna del ramo di
Claude, come il pilot, quindi non dipende da `main`).

## Forma della risposta

- Un file `docs/reviews/emendamento_r3_lettura_20261026.md` sul tuo ramo, piu' le tue voci in `docs/claude_GPT_shared` (sezione 1 per la
  lettura, sezione 2 per ogni correzione che chiedi a Claude, con prova o regressione).
- Per A: una riga di verdetto per punto (CONFERMO / MODIFICO: ... / TOLGO: ...). Per B e C: una riga di decisione ciascuna. Per D: data e modo.
- Separa verifica (fatti con `file:riga`) da opinione; cita Simone alla lettera; nessuna soglia riscritta fuori da una «proposta» firmabile.
- Nessun job, nessun deploy, nessuna modifica ai file di Claude.
