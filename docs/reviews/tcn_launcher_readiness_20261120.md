# TCN launcher: review acquisita e prerequisito hardware ancora non verificato

20/11/2026, operatore unico «2-TCN head launcher». Nessun job, deploy, prenotazione nel ledger o `release-plan`.
L'autorizzazione di Simone alla catena resta valida; non manca un nuovo consenso.

## Review e integrità

Letta integralmente la review operativa favorevole di Claude, commit
`85f0d62` (SHA completo nel JSON), `docs/reviews/claude_review_pose_temponet_runtime_rereview_20261120.md`.
Copia testuale inclusa nel ramo operativo. Codice rivisto `01252b6c3023c49046b312e003b7915d8f3f6119`,
pacchetto `aa3c8047ab8078d5dcc4e061537c3e5b35189bc8`.
Verificati i 246 file del manifest sorgenti e i tre artefatti pinnati della consegna;
source_sha256 `027bb7876b372ceb537610322c11f1903c8c2224abfca27484c403bbc297d732`.
CLI `inspect` sul piano consegnato: 24 allocazioni, ancora REVIEW_REQUIRED nel piano storico,
nessuna release implicitamente dedotta dai test. Nessun sorgente modificato.

## Controlli aggiornati che passano

Al 20/11 circa10:29CET: nessun TCNH nel ledger o nella coda utente; checkout e radice output TCN ancora assenti.
WORK:1.030.313.960KiB usati su1.073.741.824KiB, **41,416GiB** disponibili sulla quota.
Riserva prudente per scritture pendenti altrui:24GiB, distinta dai12GiB base del gate TCN.
Base della riserva: tre checkpoint300M e copia temporanea circa17,4GiB, arrotondamento20GiB già nel handoff scaling,
più4GiB per metadati e scritture operative; non è una misura futura esatta. In coda soltanto il training scaling39237,
che scrive su SCRATCH; la successiva archiviazione WORK è comunque riservata. Gate12+24GiB superato.
Saldo:655.982ore locali residue. SerialeUP,8CPU/30.800MB/4h; BoosterUP/24h.
Prenotazioni di manutenzione Booster e seriale23/11 08:00–25/11 08:00 confermate.
Queste fotografie vanno aggiornate prima del vero rilascio e dopo i benchmark come previsto.

## Prerequisito non verificato

La review favorevole (§3) e il documento d'esecuzione chiedono di confrontare il **modello CPU di login08 e login13 prima del rilascio**:
il contesto numerico cpu8 include quel valore e se i modelli differiscono serve una regola per classe prima degli invii dipendenti.

- `scontrol show node` mostra stessa topologia per entrambi:2socket,32core/socket,2thread/core,128CPU,514000MB.
- Il file corrente `/var/spool/slurmd/conf-cache/slurm.conf` li mette nello stesso gruppo `NodeName=login[01-16]`.
  Non contiene però il modello CPU: questa uguaglianza non basta a certificare il requisito.
- `ssh login08 lscpu` e `ssh login13 lscpu` dal login pubblico falliscono con `Permission denied`.
  Richiedere agent forwarding non risolve; nessuna credenziale è stata modificata o copiata sul cluster.
- Anche un collegamento dal Mac attraverso il login pubblico non esegue il comando sui nodi seriali: la connessione si chiude
  dopo l'offerta del certificato. Il client inizialmente non conosceva la chiave host; la registrazione temporanea non ha
  risolto l'autenticazione. Nessun parametro di verifica delle chiavi è stato disabilitato.
- `lscpu` eseguito su login07 dà Intel Xeon Platinum8358@2.60GHz. Non lo si spaccia per una lettura di login08/login13.

Quindi **non eseguo release-plan** e non invio il preflight fintanto che manca la prova richiesta.
Non aggiungo allocazioni hardware fuori dal grafo24job, non rimuovo il modello dal pin e non modifico i sorgenti rivisti.
Servono output affidabili già disponibili dei due nodi, oppure una procedura operativa approvata che raccolga la misura
entro le allocazioni esistenti. È un prerequisito tecnico, non una nuova decisione scientifica o autorizzazione ai lanci.

[Fotografia strutturata](../../results/infra/pose_temponet_execution_20261119/launcher_preflight_20261120/readiness.json)
e stdout della verifica nello stesso percorso. Il worktree scientifico originale è intatto; questa consegna sta nel ramo
`codex/pose-temponet-launcher-20261120`.

## Seguito: misura richiesta da Simone

Simone chiede «e come si fa a complertarlo? lo fai tu ?». L'operatore completa il controllo con due inventari hardware
SLURM di un minuto, uno per nodo, via submitter e ledger comuni. Non sono esperimenti scientifici e non modificano
i sorgenti rivisti. Sono due allocazioni diagnostiche aggiuntive fuori dal grafo scientifico24job (totale fisico26secompleto):
zeroGPUh e massimo0,266667coreh includendo180s/job e dueCPUallocate prudenziali. Il costo è detratto dalla riserva
operativaCPUdi3coreh, senza aumentare il tetto80coreh CPU-only o272coreh globale. Numero e costo vanno esplicitati
nel consuntivo, senza contarli come due fit. Nessun retry o risottomissione automatica. Il piano e lo script di inventario
sono nella directory delle prove, distinti dal piano eseguibile immutato della matrice scientifica.

## Chiusura del prerequisito, 20/11 alle11:01CET

Inventari64931087(login08) e64931101(login13) COMPLETED0:0: **entrambi Intel Xeon Platinum8358@2.60GHz**.
Logcopiati e verificati perSHA256, contabilitàsacctarchiviata:2s×2CPU+3s×2CPU=**0,002778coreh**,zeroGPUh.
Il tentativoSSHnon era la procedura necessaria:avrei dovuto passare prima alle misureSLURM, senza insistere sulcertificato.
Controllo hardware ora soddisfatto. Evidenzaoperatore aggiornata11:01:21+01:00,quota41,407GiB e riservaaltrui24GiB;
`release-plan` completato con review85f0d62159d0ac5337857540a14466b3875bebf1,codice01252b6c e sorgenti invariati.
Piano rilasciato `plan_released_20261120.json`,sha256e3f5b119b19af08274d2df0744ef3772323f4325bcb4d50e72804b56235ef261.
Segue pubblicazione dei soli metadati e invio del preflight tramite ledgercomune; nessun ulteriorevia richiesto.

## Esito del primo preflight e consegna della diagnosi

Job64931573 **FAILED1:0** dopo73s su2CPU:0,040556coreh,zeroGPUh. Inclusi i due inventari:treallocazioni,
**0,043333coreh totali**,zeroGPUh. Nessun benchmark o fit neurale inviato. Ricevuta e log raccolti byteperbyte,
SHA256remoti verificati,contabilitàarchiviata. Checkoutcreato e piano materializzato restano intatti,come il ledger.

Errore `cache user population differs from original producer`. Diagnosi indipendente sui soli `meta_json.npy`
delle duecache:train1907finestre/40utenti,val18548/15;uguaglianzadirettafalsa,uguaglianzaaggiungendo ilprefissocartella
`u` esatta suentrambi. Nessun nuovo score o apertura dei latenti/target da questa diagnosi. Ilproduttorestorico
crea lecartelle con `u`+`_SAFE_RE.sub('_',user)` ma conserva l'IDsenza `u` inmeta_json.

Concordanza con notaClaude629fdaed,copiata testualmente nel ramo con loSHAoriginale nel JSONdell'esito.
**Correggere alla fonte `historical_producer`**, non eliminare il confronto di riga439:la lista entrainbenchmark,
calibrazioni,controlli efold. La correzione deve preservareunamappaturaesatta alla cartellaoriginale,con test
su entrambe lericevute reali esulformato vero dei metadati.

Consegna alresponsabile scientifico/codice:nuovaidentitàsorgenti,piano,ID echeckout;diffminimo e riletturabreveClaude.
PerL3-9ilnuovotentativo èunadecisione distinta; nessun rilancioautomatico,nessuna sovrascrittura o uso delvecchiocheckout.
Ildifetto èdi rappresentazioneidentificativi,non prova didaticambiati o popolazionidiverse. La paritàR-1 e gli altri
gate restanoda eseguire nelnuovopreflight.
