# Linea 3 · Lettura della calibrazione RMS100/fast125

Il job `65015116` ha completato i 512 batch previsti sui dati di training con zero aggiornamenti dell’ottimizzatore. Le impronte iniziale e finale dello studente e del teacher coincidono. Il lettore congelato, rieseguito sui file archiviati, restituisce gli stessi campi e gli stessi pesi del rapporto originale `20cb900522b62b46f17d1b65cbbe1ef579a4cdd5e935a7ab31b7a309ed516532`. Sono verificati i digest dei 14 artefatti raccolti. Il checkpoint resta su Leonardo, con SHA e dimensione nella ricevuta del launcher. [Rapporto originale](../../results/infra/contrastive_targets_20261118/calibration_execution_20261121/calibration_report.json) · [Consuntivo](launcher_targets_calibration_20261121.md)

## Lettura dei gradienti

Dopo 256 batch di riempimento, tutti i 256 batch del pannello offrono supporto e tre norme di gradiente finite e positive. La selezione prescritta prende i primi 32, passi 257–288. Nel pannello è ammissibile il 99,84% delle query InfoNCE, con l’esenzione CapgMyo già registrata. RnC è attivo in 256 batch su 256. Le norme sotto si riferiscono ai parametri condivisi dell’encoder. La mediana della norma dell’obiettivo di base è 25,303 sui 32 batch.

| Mediana sui 32 batch, salvo il peso | InfoNCE | Rank-N-Contrast |
|---|---:|---:|
| Norma del gradiente contrastivo grezzo | 2,558 | 0,914 |
| Rapporto fra norma di base e norma contrastiva, calcolato per batch | 10,044 | 27,879 |
| Peso λ dalla regola firmata | 0,118471 | 0,328853 |
| Norma pesata / norma di base, calcolata per batch | 1,180% | 1,180% |
| Coseno del gradiente contrastivo con quello di base | −0,053 | +0,041 |

La regola usa `0,1 × 0,1179564123141922 × mediana(‖g_base‖ / ‖g_termine‖)`. RnC riceve un coefficiente 2,78 volte maggiore perché produce un gradiente iniziale più piccolo. Ciascun coefficiente appartiene a un braccio distinto. La quota mediana iniziale dell’1,18% discende dalla regola e dal suo fattore di attenuazione storico. Il rapporto effettivo quando la base evolve si misura ai passi 513–576 dei benchmark. I coseni mediani sono prossimi a zero anche su tutto il pannello (−0,057 e +0,043). Al punto iniziale i due termini hanno quindi, ciascuno rispetto alla base, una componente prevalente in una direzione diversa. La misura di utilità per posa e classificazione richiederà gli endpoint a valle. [Protocollo firmato](../../results/infra/contrastive_targets_20261118/plan.json)

## Fattibilità e seguito proposto

Con batch globale 32 e quattro GPU, in bf16 con SDPA math e TF32 spento, il massimo fra i rank della memoria CUDA allocata è 19,27 GiB. Il tempo medio misurato per passo è 0,956 s nel riempimento e 1,987 s nel pannello. SLURM riporta 891 s complessivi. Checkout e calibrazione consumano 0,99 GPU-ore e 7,963333 core-ore. La coorte, incluso l’audit, è a 0,99 GPU-ore e 16,953333 core-ore. I benchmark attivi daranno il riferimento per il preventivo del training.

Accetto la calibrazione come prerequisito e propongo il solo **collaudo tecnico CUDA**. La release dovrà incorporare il rapporto e il checkpoint `d0dd318302f278e08513ec7d5acde3c82b36a523f7c650829167a68864870ebc`. Il collaudo usa due aggiornamenti, batch di 8 e un rank senza posa. Confronta le quattro GPU con il riferimento non distribuito in FP32 e bf16 e verifica gli arresti coordinati. I termini contrastivi attivi hanno peso unitario in questa prova, come previsto per mettere alla prova il calcolo. Il budget della cella è 4 GPU per 30 minuti, 2 GPU-ore e 16,266667 core-ore incluso il checkout. [Cella e risorse](../../results/infra/contrastive_targets_20261118/preflight_plan.json)

Se il CUDA passa, i benchmark confronteranno C0, CN e CR con i pesi misurati. Nel C0 si verificherà la parità a peso zero su batch globale 32 ai passi 128 e 256. In CN e CR si leggeranno ai passi 513–576 la mediana e il 95º percentile del rapporto fra norme pesate e norma di base, seguiti dai confronti delle riprese dal checkpoint 544. La scelta fra gli obiettivi dipenderà poi dalla posa dinamica, da P1 e da EPN612. Questa lettura conclude la calibrazione e propone la fase successiva.
