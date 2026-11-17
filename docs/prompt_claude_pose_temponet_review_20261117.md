# Prompt per Claude — review della proposta TEMPONet sui latenti

Ramo GPT `codex/pose-temponet-proposal-20261117`, commit che pubblica questo file. Si tratta della risposta al prompt del 16/11 consegnato da Simone, non di un lancio. Stato **REVIEW_REQUIRED**, nessun runtime implementato, nessun job/deploy/merge. Il protocollo A sull'efficienza delle etichette resta accantonato; qui «livello A» significa soltanto lettore su encoder congelato.

Leggi:

- `docs/pose_temponet_frozen_protocol_20261117.md`;
- `results/infra/pose_temponet_proposal_20261117/proposal.json` e `analytic_checks.json`;
- le nuove voci GPT del 17/11 nel registro e in `docs/decisioni.md`.

La proposta usa le cache già conservate del teacher T100-G 30M 49.046: 40 utenti per il fit delle teste, 15 utenti val di sviluppo in una sola lettura finale. TCN compatta ispirata a TEMPONet, non istanza invariata della replica raw né riuso dei suoi pesi: LN affine per patch, 384→64, tre Conv k3 con dilatazioni 1/2/3, 64→32→20, ReLU, dropout 0,1, stride1 senza pooling, RF13 centrato compatibile col crop6. 65.204 parametri. MLP istantanea quasi esattamente a pari capacità (65.300), L13 lineare temporale100.628, PoseHead L0 rifittata8.468; stessa TCN anche su pre, descrittiva. Cinque teste nuove: 20 fit di selezione +15 rifit su tre semi. Output inizialezero, shuffle indipendente dall'inizializzazione, loss MSE media per finestra, LRcostante{1e-4,1e-3},50epochemax. Ridge e testa storiche riusate senza nuovo fitting.

Rilievi del prompt affrontati: la ridge storica pesa gli elementi validi, non le finestre, quindi il confronto controllato primario è TCN-post/L0. RF378 riguarda un elemento convoluzionale della TEMPONet raw, non tutta l'uscita flatten. TCN e backbone restano non causali. Il test ufficiale non è un insieme di selezione. Raw/random e livelloB sono separati; nessun intervento sui digest o sui nove contrasti della linea3.

Chiedo verdetti separati su:

1. **Domanda e architettura:** corretto il livello congelato e questo adattamento rispetto al codice esistente? RF, capacità, normalizzazione e crop mantengono il significato dichiarato? Niente ricerca di teste aggiuntive.
2. **Controlli e attribuzione:** L0 rifittata, MLP a capacità pari, L13 e pre descrittivo bastano per la domanda circoscritta? Conferma che non si pretende un vantaggio del pretraining rispetto a random/raw.
3. **Selezione e supporto:** due fold utenti, MSE per finestra, epoca/LR selezionati sul train, tre rifit con semi fissi, media di prestazioni e non ensemble; indici/mask/statistiche originali. Nessuna apertura val per scegliere varianti.
4. **Criteri:** cinque contrasti E_dyn con IC99% Bonferroni, soglia pratica2%, pre/post descrittivo. Chiusura sia positiva sia negativa/inconclusiva. Una vittoria dinamica non implica MAE/livello migliori o adozione automatica.
5. **Costo e fattibilità:** tetto proposto4GPUh+12corehCPU-only,44coreh totali,8GiBoutput. Non è una previsione misurata. Il benchmark deve provare che l'intera matrice, proiettata con tutte le50epoche e fattore2, rientra prima del primo fit; niente riduzioni silenziose o retry per rientrare. Saldo/quota del prompt sono storici.
6. **Consegna e seguito:** GPT autore del codice, Claude revisore e unico operatore proposti per continuità con la linea2, decisione di Simone prima del piano eseguibile. Nessun merge TEMPONet, edit ai rami di Claude, sorgenti/config della linea3 o pretraining nuovo.

La review locale preliminare ha già fatto correggere shuffle, aggregazione dei secondari e distinzione fra miglioramento dinamico e qualità assoluta della posa. Non sostituisce la tua review. Pubblica verdetti, rilievi e proposte motivate sul tuo ramo con registro, commit/push e SHA remoto verificato. Nessun benchmark, fit o invio prima della decisione sul disegno e sul tetto; niente nuovo consenso per le campagne già chiuse.
