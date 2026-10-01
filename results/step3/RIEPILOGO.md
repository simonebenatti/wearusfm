# Passo 3 — gate di consistenza al ricampionamento D8 (job 59087481, 01/10/2026)

**Questi sono DATI letti contro soglie e definizioni congelate prima del lancio** (`docs/decisioni.md`, D8a: soglie del 29/09, definizioni firmate
il 01/10 «firmo D8: 1a, 2, 3, 4a, 5, 6», filtro a muro firmato il 01/10 «firmo la correzione del filtro, lancia il gate»). Testo delle definizioni:
`docs/proposta_gate_d8.md`.

File: `gate_59087481.json` (esito completo) e `gate_d8_59087481.out` (log), copiati da Leonardo con sha256 uguali
(`835912d5...c040` e `c1e9678b...d9c`).

## Provenienza e costo
- Codice `c730b18`; front-end `src/wearusfm/model/frontend.py` all'inizializzazione (patch 25 ms = default di lavoro di D10, contesto 100 ms per
  lato, 24 + 24 + 16 kernel Fourier/spline/MLP), semi 0, 1, 2.
- Dati: Kaifosh processato (`$SCRATCH/data/processed/kaifosh`, 100 sessioni), 20 finestre da 4 s (+ 0,5 s di margine per lato) per soggetto =
  2000 finestre da 100 soggetti; split della sonda per soggetto 60/20/20, 800 campioni di test (400 finestre x 2 versioni).
- CPU seriale (`lrd_all_serial`, 2 core): 20 min 7 s, ~0,7 ore locali, 0 GPU-ora. Primo lancio (job 59086063) fuori memoria dopo 71 s, prima di
  qualunque misura (causa e correzione in `decisioni.md`).
- **Memoria di picco 12,4 GiB (MaxRSS 12.984.284 K) sui 16 GB richiesti**: molto piu' della stima fatta sul Mac (~5 GB). Causa non indagata; per un
  rilancio serve piu' memoria.

## Risultati

Soglie: (i) errore relativo RMS <= 5% nel totale e in ciascuna famiglia; (ii) sonda A-contro-B <= 0,55 con intervallo al 95% che contiene 0,50.

| Caso | Seme | Errore totale | Fourier | Spline | MLP | Sonda (acc. bil.) | IC 95% | Modello scelto | Esito |
|---|---|---|---|---|---|---|---|---|---|
| 1 kHz (450 Hz) | 0 | 0,001% | 0,001% | 0,000% | 0,000% | 0,500 | 0,500-0,500 | logreg C=0,1 | passa |
| 1 kHz (450 Hz) | 1 | 0,001% | 0,001% | 0,000% | 0,000% | 0,500 | 0,500-0,500 | logreg C=0,1 | passa |
| 1 kHz (450 Hz) | 2 | 0,001% | 0,001% | 0,000% | 0,000% | 0,500 | 0,500-0,500 | logreg C=0,1 | passa |
| 200 Hz (90 Hz) | 0 | 0,41% | 0,43% | 0,42% | 0,30% | 0,503 | 0,451-0,554 | logreg C=0,1 | passa |
| 200 Hz (90 Hz) | 1 | 0,38% | 0,40% | 0,42% | 0,25% | 0,504 | 0,453-0,556 | logreg C=10 | passa |
| 200 Hz (90 Hz) | 2 | 0,40% | 0,43% | 0,43% | 0,28% | 0,505 | 0,473-0,539 | logreg C=10 | passa |

**Gate D8: SUPERATO** (entrambi i casi, tutti e tre i semi, entrambe le misure).

## Note (letture, non decisioni)
1. Nel caso 1 kHz le due versioni danno feature praticamente identiche: la sonda da' la stessa risposta alle due versioni di ogni finestra e
   l'accuratezza e' esattamente 0,500. Nel caso 200 Hz l'errore (~0,4%) e' dello stesso ordine di quello misurato su dati sintetici prima del gate
   (0,35%); la sonda resta vicina al caso.
2. Nel caso 200 Hz l'estremo superiore dell'intervallo supera 0,55 in due semi (0,554 e 0,556). La regola congelata riguarda l'accuratezza (<= 0,55)
   e il fatto che l'intervallo contenga 0,50, non l'estremo superiore: il dato e' riportato perche' l'intervallo e' largo (400 finestre di test).
3. Il gate e' misurato sul front-end **all'inizializzazione**, come firmato (la proprieta' e' dell'architettura: fattore Δt e anti-aliasing per
   famiglia). Vale per la configurazione provata (patch 25 ms, contesto 100 ms): se D10 cambia la lunghezza della patch, se rifare il gate e' una
   decisione di Simone.
4. La sonda ha scelto sempre la regressione logistica sulla validazione (il gradient boosting era fra i candidati).
