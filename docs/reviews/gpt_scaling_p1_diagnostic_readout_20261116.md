# P1 a 19.618: diagnostica conclusa, guasto non riprodotto

16/11/2026. Checkout **64610004**, diagnostico **64610181**, entrambi
`COMPLETED 0:0`. Codice congelato `e4f615e8eb3c33e584d55c6e83cdd48ec9bac4a9`.
Esito: **NOT_REPRODUCED / READERS_CLEAR_IN_TEST**. Il criterio approvato
ferma qui la catena: **P1-r2 non e' stato sottomesso**, nessun ulteriore
checkout, nessun retry. Posa e P2 rimangono conservate.

## Che cosa abbiamo fatto e trovato

| Controllo | Risultato | Conclusione ammessa |
|---|---|---|
| Correzione circoscritta della finalizzazione | Helper, wrapper P1 e bridge di provenienza pronti; 329 test passati prima del rilascio | Il candidato rifiuta le mutazioni di byte/oggetto e permette solo un assestamento temporale verificato; le guardie degli input restano intatte |
| Matrice sintetica fresh/aged, A/F/S | 18/18 file integri; entrambi gli SHA uguali all'oracolo; nessun drift mtime/ctime o dei campi protetti | Il difetto dei due P1 non si e' riprodotto su questo nodo e in questa prova |
| Attesa dei file aged | Sei DB2/DB3 osservati dopo 729,64–774,07 s dalla chiusura | Coperto anche un intervallo superiore a 12 minuti |
| Sei cache reali, prima e dopo il carico | 12/12 controlli stabili, pin rispettati, guardie reali di classificazione e posa conformi | Non emerge qui un motivo per modificare i lettori; nessuna garanzia su 43 ore di esecuzione |
| Rilascio P1 sul risultato reale | Il costruttore rifiuta con `diagnostic does not release P1 recovery` | Il blocco e' meccanico e conforme al criterio fissato prima dei dati |
| Launcher del possibile P1-r2 | Preparato durante l'attesa; altri 34 test superati; non distribuito | Pronto come componente, privo di manifest di rilascio e di job |

Sono stati scritti **27.719.878.320 byte** sintetici, sotto il tetto di
32 GB. Le scritture dei file grandi fresh hanno richiesto circa 1,4–1,6 s,
quindi la prova ha raggiunto anche quell'ordine di velocita'. Le variazioni
di atime osservate sono descrittive e non erano causa di rifiuto nella
guardia originale. Tutti i candidati sulle osservazioni sintetiche sono
stati accettati come stabili; non e' stato necessario tollerare un drift.

La review indipendente dell'esito ha riprodotto **617 controlli** su report,
eventi e accounting: numeri, hash, eta', pin dei 228 sorgenti congelati,
identita' del rilascio e applicazione del criterio. Non sono score del FM.

## Interpretazione e seguito

La causa dei due fallimenti rimane aperta. Il risultato non dimostra che
gli attributi Lustre fossero stabili nei job falliti, per i quali mancano
le coppie stat prima/dopo che provocarono il rifiuto. Non dimostra neppure
che `os.sync` o `fsync` risolvano il difetto: nessuna delle modalita' lo ha
riprodotto e le sincronizzazioni globali coinvolgevano anche altre fixture.
Non stimiamo una probabilita' di guasto dai 18 file correlati.

Il protocollo approvato dice, per NOT_REPRODUCED: «non autorizzare
automaticamente una terza inferenza» e «Nessun nuovo job discende da un
risultato inconclusivo di questo protocollo». L'approvazione di Simone ha
autorizzato implementazione e invii diretti entro questi criteri; cambiarne
adesso la condizione richiede una decisione esplicita, non una ripetizione
del consenso ai comandi ordinari. Le autorizzazioni operative restano valide.

**Proposta per la prossima decisione, non eseguita:** un unico P1-r2 con
la finalizzazione difensiva gia' preparata e l'audit completo dei confini,
ammesso anche senza riproduzione sintetica, per verificare direttamente
il producer reale. Sarebbe un emendamento del criterio di rilascio: non una
diagnosi chiusa della causa. Resterebbero invariati codice numerico,
checkpoint, ricetta, input guard, limiti di rilettura e stop su qualunque
errore di contenuto/oggetto o seconda instabilita'. I due falliti restano
preservati; nessuna adozione delle loro cache. Un massimo di 2,05 GPUh e
un solo checkout aggiuntivo rientrerebbero nei budget gia' approvati.
Il cancello attuale continua a rifiutare tale invio finche' il criterio
non viene emendato; il suo codice non e' stato allentato.

## Consuntivo e tracciabilita'

| Voce | Tempo | GPUh allocate | Coreh allocate |
|---|---:|---:|---:|
| Checkout | 41 s | 0 | 0,022778 |
| Diagnostico | 977 s, 16 min 17 s | 0,271389 | 2,171111 |
| Estrattori precedenti + diagnostico | — | 4,985833 | 39,886667 |

Diagnostico sotto 0,5 GPUh / 4 coreh. Un ipotetico P1-r2 al massimo
porterebbe lo stadio a 7,035833 GPUh su 7,30 e P1 cumulativo a 3,070833
su 3,10; queste sono proiezioni, non consumi o un rilascio.

Evidenze nel pacchetto
`results/infra/scaling_300m_20261104/p1_recovery_20261116/`:
`evidence/run_final/{report.json,events.jsonl,sacct.txt,accounting.json}`,
`p1_gate.json`, `submissions/`, `reviews/` e i rapporti dei test.
SHA report `c2e32e9cfab36a76c79b05913e350b9ec7f15484f73e2a271552c06f6e980908`;
SHA eventi `035a5333ac9cb69b38ae8cd77b1bcf807085f02881ed3161b9fae8569fcd44e5`.
La raccolta finale verifica gli hash dei file ricevuti e il legame report/eventi.
Nessun nuovo training, cambiamento della loss o riestrazione del controllo.
