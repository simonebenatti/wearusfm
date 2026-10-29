# Preparazione di B: consegna e verifiche del 28/10/2026

Autore: GPT, su richiesta di Simone «la vuoi preparare tu?». Base tecnica Claude `afd2e4d6aca7a417fef59efcd3df2c048fa23653`.
Codice congelato: **`24139ab901c052357a46a65bc20466f2eb1e8358`**. Ramo di consegna: `codex/r3-long-collaudo-20261028`.
I commit successivi consegnano documentazione, piano e comandi che puntano a questo SHA del codice; non cambiano l'eseguibile.

**Pronto per review reciproca e firma, non lanciato.** Nessun deploy, checkout remoto, job, E3 R0 o J2 bis eseguito in questa preparazione.
Protocollo: [collaudo_r3_b_20261028.md](../collaudo_r3_b_20261028.md). Il lancio resta separato dalla preparazione autorizzata.

## Scelte scientifiche

- B riparte dall'inizializzazione di R3, seme 0; 5.000 aggiornamenti dello schedule 49.046, warmup 1.962. Non riprende il dbg decaduto.
- Pannello bilanciato di 32 utenti, due registrazioni per utente, quattro finestre valide da quattro secondi per registrazione.
  Finestre senza sovrapposizione; campionamento deterministico basato sulla disponibilita' dei dati, mai sulla loss.
- Statistica primaria: media delle otto MSE per utente, poi media dei 32 utenti. Rapporto endpoint/inizio <=0,9 e limite superiore
  dell'IC95 bootstrap appaiato <1 (1.000 repliche, seme 20261027). Un denominatore bootstrap nullo invalida la lettura.
- Discesa nel flusso, MAE, gradienti e varianza temporale sono descrittivi. Il gate non prova da solo apprendimento del movimento o
  generalizzazione; nessuna continuazione automatica di R3. Le regole dei run precedenti restano immutate.
- Proposta di risorse aggiornata e ancora da firmare: 90 min, quattro A100, 32 core = 6 GPU-ore /48 core-ore, piu' checkout CPU
  di due core per dieci minuti. Il margine aggiunto serve alle diagnostiche e ai checkpoint; nessun retry automatico.

## Correzioni tecniche rispetto alla bozza

Selezione solo fra registrazioni attive del training, deduplicate, con supporto valido e quote complete. Il manifest del pannello registra
coordinate, metadati di registrazione, esclusioni, conteggi per giunto, statistiche e hash dei batch. Ogni valutazione porta l'hash canonico
dell'intero manifest. Alla ripresa l'identita' deve coincidere. Errori di preparazione sul rank 0 vengono propagati agli altri rank.

La varianza ora si calcola lungo il tempo per ogni giunto e poi si media sui giunti ammissibili: bias diversi fra giunti non simulano
movimento. Al passo zero il lettore verifica direttamente le medie predette per giunto (tolleranza 1e-6), oltre a MSE e varianza.
Supporti, passi, aggregati e baseline sono verificati prima del gate; duplicati, dati incompleti o provenienza incoerente non passano.
Il launcher ha una guardia facoltativa `REQUIRE_NEW_RUN=1` per rifiutare directory esistenti e riprese esterne prima di iniziare B.

## Verifiche locali

**123 test distinti superati** nelle tre esecuzioni seguenti (sei test del pannello compaiono nei primi due gruppi):

| Gruppo | Test | Esito |
|---|---:|---|
| `test_pose_panel`, `test_pose_training`, `test_pose_chain_oracle`, `test_pretraining_loader` | 28 | passati, 80,88 s |
| `test_pose_panel`, `test_arms_collaudo`, `test_collaudo_b_launcher`, `test_pose_targets`, `test_stability_diagnostics` | 99 | passati, 80,92 s |
| `test_pose_panel_ddp` | 2 | passati, 14,42 s |

I 76 test del lettore includono le regressioni degli altri bracci, corruzioni del pannello, log duplicati, schedule sbagliato,
bootstrap con utenti a errore iniziale nullo e bias non nulli al passo zero. Il confronto pannello acceso/spento controlla pesi,
teacher, optimizer, RNG e metriche del training; in DDP lo fa su entrambi i rank. Il rifiuto di un pannello impossibile viene ricevuto
da tutti i rank senza aggiornamenti o checkpoint.

Un'esecuzione nel sandbox e' stata interrotta durante il test con worker del DataLoader; l'intero secondo gruppo e' poi terminato
fuori sandbox. Anche Gloo richiede comunicazione locale fra processi: i due test DDP sono stati eseguiti fuori sandbox. Non sono
stati usati job o GPU sul cluster. I test sintetici verificano il contratto; non dimostrano che B superera' il gate sui dati reali.

Ulteriori controlli: `bash -n scripts/slurm/train_ddp.sbatch`, `git diff --check`, ricostruzione del config dal parser vero usando
gli argomenti del piano e confronto con il lettore B (nessuna differenza). Somma LR fino a 5.000: 0,40195; e' un integrale del LR,
non un'equivalenza di aggiornamenti Adam. Config completo salvato in `results/infra/pilot_posa/config_b_atteso_20261028.json`.

Review indipendenti: `r3_optimizer` sulla selezione, RNG, supporti, launcher e varianza; `b_panel_review` su materializzazione e DDP;
`r3_science_critic` su gate, bootstrap, limiti e budget. Rilievi risolti: propagazione DDP, supporti/nonoverlap, varianza temporale,
criteri aggiuntivi non giustificati, verifica diretta delle predizioni iniziali e corrispondenza dei descrittivi salvati.

## Consegna a Claude

Leggere [prompt_claude_review_b_20261028.md](../prompt_claude_review_b_20261028.md). La review reciproca prima del deploy resta
quella prevista in `CLAUDE.md`: «Quando un'integrazione cambia il comportamento del codice di Claude, Claude la rivede prima del deploy».
Non e' una richiesta di riscrivere il protocollo o di eseguire altri esperimenti. La firma di Simone riguarda la proposta concreta
nei file `piano_b_20261028.json` e `comandi_b_20261028.txt`; fino ad allora tutti i comandi restano preparati, non eseguiti.
