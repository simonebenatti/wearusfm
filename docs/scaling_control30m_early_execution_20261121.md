# Anticipo delle tre estrazioni del controllo30M a49.046

21/11/2026. Simone: «prepara il percorso separato per il30M, verificalo e fallo
lanciare al launcher senza chiedermi di nuovo il consenso, così andiamo avanti
mentre il300M è in coda». Questa consegna realizza quell'anticipo. Operatore unico:
**1-30/300 launcher**. Il training300M e il suo checkout87da975 restano invariati.

## Perimetro e provenienza

Si anticipano esattamente le tre estrazioni già previste del30M: posa, P1 e P2.
Identici checkpoint teacher49.046, finestre, utenti, dati, split, guadagno,
algoritmi, ID logici, percorsi di output, allocazioni e log del piano precedente.
Non si riaddestra il30M e non si inventa un pin300M finale. Il piano indipendente
contiene solo il30M e `control_only: true`. L'adapter sostituisce unicamente la
guardia dello schema del piano nel loader, derivandola dal sorgente originale
con hash3720836…; esecuzione, controlli, finalizzatore P1-r2 e ricevute sono quelli
originali. Il loader originale viene ripristinato anche su errore.

Nuovo worktree: `wearusfm-scaling-control30m`, ramo
`codex/scaling-control30m-20261121`. Un checkout separato ospita adapter e piano;
nessuna modifica a codice o cartelle di job già accettati. I file originali,
compresi controller, cinque valutatori, launcher, trainer, src e submitter,
rimangono byte-identici a87da975.

Piano e politica risorse:
`results/infra/scaling_49046_20261118/evaluation/evaluation/control30m_early_20261121/`.
Il pin del checkpoint resta
`bfa6141fc2ed6505d2f582a46f0c04f36fd35f867e8cff2b11555b241947a490`;
la ricevuta del controllo è quella preservata nel checkout storico.

## Risorse e sequenza

| Job | GPU/CPU/RAM | Walltime | Limite |
|---|---|---|---|
| SCALING-49046-control30m-checkout-20261121 |0/2/4G |5min |0,166667coreh;0,266667 con margine |
| SCALING-49046-extract-30m-pose-20261118 |1/8/100G |150min |2,5GPUh |
| SCALING-49046-extract-30m-p1-20261118 |1/8/100G |120min |2GPUh |
| SCALING-49046-extract-30m-p2-20261118 |1/8/100G |150min |2,5GPUh |

Le7GPUh richieste/7,15 con margine sono già comprese nelle14/14,3 della valutazione
completa: nessuna duplicazione né aumentoGPU. Un solo checkout aggiuntivo porta
i checkout complessivi da5 a6, il loro tetto da1,333333 a1,6coreh con margine e
l'inviluppo CPU-only da347,8 a**348,066667/400**. Altri tetti invariati.
Nel consuntivo finale aggiungere il costo effettivo del checkout anticipato:
il controller storico continua a contabilizzare separatamente i suoi cinque.

Il controller nuovo `scripts/scaling_control30m_ops.py` prepara documenti,
legge/verifica gate, emette righe e raccoglie. **Non invia job.** Comandi:

1. `prepare` riproduce piano e risorse; file esistenti accettati solo byte-identici.
2. `gate checkout --sha RELEASE_SHA --reviewed RELEASE_SHA`, poi `lines checkout
   --sha RELEASE_SHA`. Pubblicare il gate e le righe prima di eseguirle; deployref
   nuovo e job di checkout, senza toccare il checkout87da975.
3. A COMPLETED0:0: `collect checkout`. Il gate delle estrazioni verifica inoltre
   HEAD pulito, hash del piano realmente presente e prerequisiti sul cluster.
4. `gate extractions --sha RELEASE_SHA`, poi `lines extractions --sha RELEASE_SHA`.
   Pubblicare e inviare le tre righe indipendenti una alla volta. Si attendono
   il checkout verificato, non il300M. Nessuna dipendenza da un job già purgato.
5. Dopo COMPLETED0:0 di tutti e tre: `collect extractions`. Conservare sacct grezzo,
   accounting, log, ricevute, provenienze e accettazione. La verifica integrale
   dei byte delle cache resta nei lettori congelati prima dei fit.

Per **ogni** riga del submitter usare il preparatore locale richiesto da Simone:
`python3 /Users/simonebenatti/dev/wearusfm/scripts/mac_launcher_command.py WORKTREE SHA -- <riga>`.
Passare i campi `cmd` e `workdir` restituiti direttamente allo strumento di
esecuzione, separando preparazione, invio e raccolta. Verificare ricevuta e
partenza su Leonardo per gli ID già inviati. Il preparatore controlla la parità
del submitter con lo SHA congelato; non estende il permesso a wrapper temporanei.

Gate→lines conserva età massima1800s, snapshot del ledger, documenti, sorgenti e
prove dei prerequisiti. Coda/contabilità illeggibili, quotaWORK<30GiB, output già
presenti o ID ambigui fermano lo stadio. Zero nuovi ID300M, niente retry automatico.
Un vero errore si diagnostica; un'attesa di slot non rinnova il consenso.

## Rientro nel confronto finale, senza nuove estrazioni30M

Il vecchio worktree `wearusfm-scaling-49046-launcher` rimane il percorso del300M.
NON fondere il codice di questo branch nel suo checkout o nei suoi commit
metadata: devono continuare a contenere solo documenti rispetto al rilascio87da.

Quando il300M conclude training e archivio, il controller originale genera il
piano completo con i pin effettivi. Il suo gate `extractions` trova i tre ID30M
già nel ledger, ne richiede COMPLETED0:0 ed emette solo le tre estrazioni300M
mancanti. Se qualche ID30M non è concluso, attende; non lo rilancia.

Dopo `collect extractions` originale sui sei job, sostituire **solo** il comando
di costruzione dei reader plans con il seguente, eseguito dal nuovo worktree:

`scripts/scaling_control30m_ops.py reader-plans --campaign-root /Users/simonebenatti/Documents/ChatGPT/FM\ per\ EMG/wearusfm-scaling-49046-launcher`

Il comando verifica l'accettazione anticipata e l'identità di tutti i file/pin
del30M fra raccolta anticipata e finale. Il bridge richiede stessi input comuni,
stesso pin30M, tutte le sei ricevute e la coppia esatta codice/piano per braccio.
Produce pose_plan, probe_plan e lineage nel worktree storico, insieme a piano,
accettazione e politica risorse dell'anticipo. Non riscrive cache o provenienze.
I lettori87da975 supportano già produttori distinti per cache e rimangono invariati.

Pubblicare quei documenti sul ramo storico e riprendere normalmente checkout-plans,
validazione, benchmark, budget e fit, aggregati e confronto. Nessuna nuova review
fra gli invii né altro consenso. L'anticipo non avvia fit indipendenti del30M:
riduce l'attesa delle feature; lettori e confronto restano la catena comune.

## Verifica e limiti

Test del nuovo adapter/launcher, piani e join, gate/righe/accounting e regressioni
del rilascio precedente. Due review indipendenti: scientifica su adapter/join,
operativa sui cancelli. Corretta prima del rilascio una divergenza fra raccolta
e rilettura del MaxRSS: per questi stadi non alimenta un budget di memoria e può
essere assente; il valore resta registrato quando presente, come nel collector.
Nessuna prova GPU o cache futura viene attestata dai test locali.
