# Continuazione B a 19.618 e confronto con R3c — 28/10/2026

**Autorizzata da Simone:** «procedi», poi «quando pronto lanciali tu». Il via comprende preparazione,
sottomissione ed esecuzione della proposta intermedia, anche oltre 5 GPU-ore; nessuna seconda richiesta di consenso.
GPT implementa e lancia, Claude revisore. Metodo e codice pubblicati prima delle rispettive esecuzioni.
Non comprende il tronco completo a 49.046, nuovi semi o retry automatici.

## Scopo e disegno

Il checkpoint B5000 ha appreso una componente di posa fuori dai soggetti train, ma dinamica e beneficio restano modesti.
E3 non ha recuperato altra dinamica mediante il rifit ridge. Questo motiva una misura intermedia dell'evoluzione e del
trasferimento; non prova che altro training risolvera' il problema. 19.618 e' circa il **40%** dei passi del tronco a 49.046,
non un quarto, ed e' un checkpoint conservato di R3c. Le letture sono esplorative, a un seme, su insiemi gia' consultati.
Il test ufficiale resta intatto; nessuna selezione del migliore checkpoint intermedio o modifica dei parametri da val.

Un solo resume da 5.000 a 19.618: stessa ricetta R3, encoder30M, seme0, quattro A100, batch globale32, accumulo1,
6 worker/rank, segmento1200, manifest/scale/durate/pose_stats congelati. JEPA0, ancore0,2, posa1,0, M280,
mano non sollecitata inclusa. WSD total49046, steps_at_d98094, warmup1962, LR1e-4; cambia solo lo stop operativo.
Nessuna modifica a testa, pooling, obiettivo, peso o margini. Medie dei canali nel readout ufficiale conservate.

Ambiente scientifico frozen `24139ab901c052357a46a65bc20466f2eb1e8358` nel checkout esistente
`/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pilot-posa-b-20261028`.
Nuovi wrapper/analisi inviati come payload con SHA; nessuna modifica al checkout remoto.
Origine B: `.../wearusfm_runs/runs/pilot_posa/PILOT-POSA-R3-B-30M-s0-20261028` su scratch;
destinazione nuova: `.../wearusfm_runs/runs/pilot_posa/PILOT-POSA-R3-B19618-30M-s0-20261028`.
R3c: `.../wearusfm_runs/runs/pilot_posa/PILOT-POSA-R3C-ancore-30M-s0/checkpoint_step0019618.pt`.

## Integrita' della ripresa e comparabilita'

Preflight CPU in sola lettura: SHA e contenuto dei checkpoint fidati B5000 e R3c19618; config, step effettivo,
manifest, scale, durate, modello condiviso, mascheramento, ottimizzatore, clipping, EMA, stream e ambiente.
Differenze attese: testa/loss/target di posa e diagnostiche. Non usare il checkpoint R3c postdecay49.046 o ramo24.523.
Il diff del codice R3c c5e42cb → B24139ab non modifica modello/JEPA; loader/pannello hanno test d'invarianza.
Identita' nominale di seme/schedule non basta a dichiarare gli stessi batch: verificare anche firma dati e stream.
Le differenze di firma che riguardano esclusivamente target/pannello vengono elencate; nessuna uguaglianza forzata.
Un disaccordo che altera input, campionamento o ottimizzazione blocca la lettura di attribuzione e va risolto prima del training.

Il launcher crea una cartella nuova esclusivamente, copia byte per byte checkpoint e due manifest diagnostici (non hardlink),
e usa il resume normale. Student, teacher, optimizer, RNG di ogni rank e posizione dello stream sono ripresi integralmente.
Manifest/pannelli/ambiente verificati; B5000 originale resta immutato e le cache precedenti conservano la loro provenienza.
I log della continuazione iniziano a5001. Il receipt finale e' valido solo a19618, stop=max_steps, exit0 e source immutato;
un time limit non autorizza analisi degli endpoint o un retry. La copia `checkpoint_step0019618.pt` e' l'input delle valutazioni.

## A. Evoluzione della posa: student B19618 contro B5000

Riusare cache B5000, estrarre B19618 con frozen pose_eval, --which student, finestre e preprocessing identici.
Guardia esatta su ordine/metadati delle finestre, target, maschere, std e pose_var; nessuna intersezione o esclusione
opportunistica per creare una coppia. Stessi15utenti val,1950registrazioni richieste,10finestre4s, seed20261026.
Le finestre dinamiche sono quelle gia' definite dalla mediana del target della val; non si ridefiniscono dal nuovo errore.

- E1: rapporto MAE in radianti/costante train e differenza appaiata B19618−B5000, pooled sugli elementi validi.
  Salvare somme per utente, conteggi, rapporti e mediana individuale. Il denominatore costante e' comune.
- Dinamica: differenza appaiata dei rapporti MSE centrata/Vy, sullo stesso supporto finestra/giunto con n>=2.
  Riportare medie/dinamica/totale, Vp/Vy, correlazione centrata e per utente; tutte e poco dinamiche secondari.
- E2: stesse coppie deterministiche per utente/fase/lato e supporto comune, come il codice congelato. Riportare
  MAE allineata/disallineata, gap e IC per ciascun checkpoint e delta appaiato del gap, senza obbligo di crescita.
  E2 misura associazione e puo' cambiare anche per la postura media: non e' una scala monotona della qualita'.

Bootstrap appaiato utenti:1000repliche, seme20261009, rapporti di somme (non media dei rapporti utenti).
Ogni replica usa gli stessi utenti per entrambi i checkpoint; denominatore nullo/nonfinito rende il confronto non valutabile,
senza scartare repliche. Per le differenze di errore E1/dinamica: IC interamente sotto0 indica errore ridotto,
sopra0 aumentato, contenente0 inconcludente, non equivalenza. Il segno del delta del gap E2 indica invece
aumento/diminuzione dell'associazione e non e' un criterio di miglioramento.
E1 e dinamica rispondono a domande diverse: disaccordi si riportano, non si seleziona il risultato favorevole.
Nessun pass/fail sulle soglie dell'endpoint49.046, nessuna inferenza di plateau da due endpoint.

## B. Trasferimento: teacher B19618 contro teacher R3c19618

Percorso storico frozen `channel_readout.py`, che carica il teacher per entrambi; sola lettura `pool`, larghezza2d,
nessuna nuova variante local/dec. P1 DB2/DB3/DB6 e P2 EPN612/UCI; split/fold/seed/ricetta e selezione C sul train/val
interni identici al pilot. I checkpoint a19618 sono entrambi nella fase stabile, non postdecay: lettura intermedia dichiarata.
Quattro estrazioni, quattro sonde CPU, confronto --single-seed --models, R3c riferimento e B candidato.
Impronte input, soggetti, split, righe OOF e provenienza devono coincidere dove previsto; fallimento blocca il confronto.

Riportare delta S in punti percentuali e IC95 appaiati per P1/P2 e per dataset, stessi pesi del pilot.
Un solo seme di pretraining: gli IC fra utenti non misurano variabilita' fra inizializzazioni. Nessuna correzione multipla,
quindi nessuna conclusione confermativa dall'esplorazione dei dataset. Eventuali etichette legacy non decidono il seguito.
Mostrare gli intervalli anche rispetto a −1/0/+1 punti (riferimento pratico gia' usato dal pilot), senza inventare un gate nuovo:
non rilevare un calo non prova assenza di danno; IC ampi che includono cali di almeno1punto restano informativi sul rischio.
Il criterio finale di adozione R3 contro R0 resta distinto e invariato.

## C. Accessibilita' della posa: student B19618 contro student R3c19618

Stesse200registrazioni/40utenti train e stesse finestre val, target/statistiche/supporti identici.
Un'unica trasformazione comune prima del fit: media dei K latenti per istante gia' in cache, poi
**LayerNorm stateless FP32, elementwise_affine=False, eps=1e-5**, indipendente dai parametri delle teste.
R3c non ha testa di posa: has_head=false e checkpoint_pose_stats_sha256=null sono attesi, non errori.
Le statistiche dei target della cache devono invece essere le stesse pose_stats congelate.

Ridge float64 con intercetta non penalizzata, maschera per giunto e tutti gli istanti validi; due fold utenti train
ordinati/alternati, statistica di centratura stimata solo nel fit di ciascun fold. Selezione separata di alpha per encoder,
stessi fold, griglia10**[-2..4], estensioni fino a3decadi per lato, parita' alpha minore, bordo irrisolto dichiarato.
Obiettivo interno: MSE standardizzata mascherata pooled sui due fold; penalita' alpha sulla SSE non normalizzata.
Nessun tuning su val o sul suo sottogruppo dinamico. Riciclo delle routine E3 testate, senza cambiare il precedente E3B5000.

Primario diagnostico: delta appaiato rapporto centrato ridgeB−ridgeR3c sulle dinamiche, stesso bootstrap di A.
Secondari: E1, scomposizione media/dinamica/totale, varianza/correlazione, gruppi all/static e per utente.
Controllare M2 sulle cache B con la sua vera testa, stesse tolleranze preventive0,005max/0,0005RMSE per blocco128;
R3c senza testa non ha un controllo M2, ma le stesse guardie dati/latenti/provenienza.
Conclusione limitata all'accessibilita' dopo questa normalizzazione: si eliminano media e norma per istante e non si
misura tutta l'informazione lineare dei latenti grezzi. Nessun confronto numerico con E3B5000 post-LayerNorm appresa.

## Lettura congiunta e seguito

I tre confronti restano distinti. Posa migliore e P1/P2 con IC ampi lasciano il trasferimento non risolto.
Se P1/P2 escludono benefici rilevanti con sufficiente precisione, il quadro e' compatibile con una specializzazione;
la sola mancata significativita' non basta a sostenerla.
Migliore accessibilita' insieme a beneficio P1/P2 rende la supervisione di posa piu' promettente, con i limiti di un seme.
Risultati inconcludenti conservano l'incertezza; un calo P1/P2 va pesato per dimensione e precisione.
E3 non determina una mossa unica; nessun risultato genera automaticamente training a49.046.
Non adottate le soglie proposte in a2753a6: E1>=0,92 non e' plateau, IC separati non sono un test appaiato,
E2 non deve aumentare e mancata significativita' del danno non ne dimostra l'assenza.

## Risorse e contabilita'

| Attivita' | Risorse e massimo | GPU-ore massime | Core-ore massime |
|---|---|---:|---:|
| Preflight checkpoint | 2CPU,16G,10min | 0 | 0,333333 |
| Resume14618passi | 4A100,32CPU,480G,4h (limite interno13800s) | 16 | 128 |
| Posa train+val dei due encoder | 2job, ciascuno1A100,8CPU,60G,1h | 2 | 16 |
| P1/P2 quattro estrazioni | 2P1x90min e2P2x60min,1A100/8CPU/100G | 5 | 40 |
| Quattro sonde pool | 8CPU,30G,4h ciascuna | 0 | 128 |
| Rapporto posa appaiato e ridge | 8CPU,30G,1h | 0 | 8 |
| Confronto P1/P2 | 8CPU,30G,30min | 0 | 4 |
| **Totale** | **14job incluso preflight** | **23** | **324,333333** |

Stima centrale: train~12,4GPU-ore (3,1h dai tempi totali misurati di B), estrazioni P1/P2~1,4 e posa~1,0; totale~14,8GPU-ore.
I massimi CPU sono prudenziali e contabilizzano anche le CPU allocate nei job GPU. La componente dei soli job CPU
e' 140,333333 core-ore, distinta dalla componente184 core-ore dei job GPU. Nessun nuovo checkout necessario.
Voce seconda ondata del pilot; residuo e accounting aggiornati nel piano esecutivo prima dei lanci.
Piano JSON/comandi e SHA dei payload individuano le risorse effettive; non si fraziona per aggirare l'autonomia.
Submitter/ledger comune, dipendenze afterok, output nuovi esterni al codice; nessun retry automatico.

Simone richiede inoltre: **dopo aver lanciato i job, integrare tutte le consegne su main**. Review e merge successivi
non cambiano i checkout congelati dei job; commit originali e registro di entrambi conservati.
