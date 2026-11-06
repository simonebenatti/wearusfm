# PSD-3: tabelle descrittive

Shard: 20/20 (completo); ricetta unica: True; thread (BLAS, torch): ['[{"MKL_NUM_THREADS": "4", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}, 4]'].

## BAcc OOF per dataset e cella (punto ricalcolato dalle predizioni; MLP = media dei due semi, fra parentesi i semi)

| dataset | cella | logistica | MLP |
|---|---|---:|---:|
| ninapro_db2 | B/pool | 17.37 | 16.74 (16.74, 16.74) |
| ninapro_db2 | B/dec | 18.30 | 17.44 (17.55, 17.32) |
| ninapro_db2 | R3c/pool | 18.27 | 17.68 (17.48, 17.89) |
| ninapro_db2 | R3c/dec | 18.43 | 17.65 (17.61, 17.70) |
| ninapro_db3 | B/pool | 2.69 | 2.52 (2.47, 2.57) |
| ninapro_db3 | B/dec | 2.64 | 3.26 (3.10, 3.43) |
| ninapro_db3 | R3c/pool | 2.43 | 2.67 (2.23, 3.11) |
| ninapro_db3 | R3c/dec | 3.06 | 2.94 (3.22, 2.66) |
| ninapro_db6 | B/pool | 33.73 | 32.75 (32.54, 32.97) |
| ninapro_db6 | B/dec | 30.57 | 29.73 (29.92, 29.55) |
| ninapro_db6 | R3c/pool | 34.38 | 33.16 (33.34, 32.99) |
| ninapro_db6 | R3c/dec | 31.82 | 30.26 (30.13, 30.38) |
| epn612 | B/pool | 76.36 | 77.25 (77.38, 77.12) |
| epn612 | B/dec | 75.95 | 76.77 (76.76, 76.77) |
| epn612 | R3c/pool | 80.35 | 80.70 (80.72, 80.68) |
| epn612 | R3c/dec | 81.40 | 81.38 (81.44, 81.32) |
| uci_emg | B/pool | 77.15 | 77.65 (78.42, 76.88) |
| uci_emg | B/dec | 78.77 | 76.15 (76.90, 75.39) |
| uci_emg | R3c/pool | 77.82 | 79.08 (79.97, 78.19) |
| uci_emg | R3c/dec | 75.44 | 76.77 (78.11, 75.43) |

## Convergenza e selezione

- Fit candidati della logistica: 480; al limite di 12.000: 2 (selezionati al limite: 0, riaddestramenti al limite: 0).
- Iterazioni massime per C: {'0.001': 1053, '0.01': 1574, '0.1': 2826, '1': 3773, '10': 6398, '100': 12000}.
- C scelti (istogramma sui fold): {'0.01': 19, '0.1': 9, '1': 12, '10': 8, '0.001': 28, '100': 4}; lambda MLP scelte: {'0.01': 49, '0.001': 20, '0.0': 11}.
- MLP: nessuna pretesa di convergenza (budget fisso di 60 epoche); divario medio train − val (BAcc): 28.31 punti.
- Celle con fit al limite: {'B/ninapro_db3/dec': 2}.

## Dettaglio per cella e fold: C scelto (iterazioni), lambda scelta, train/val MLP per seme, varianza PCA

| cella | fold | C (iter.) | al limite | lambda | train/val seme 0 | train/val seme 1 | PCA128 var. |
|---|---:|---|---:|---|---|---|---:|
| B/epn612/dec | 0 | 0.01 (1568) | 0 | 0.01 | 87.98/77.31 | 88.05/77.44 | 99.41 |
| B/epn612/dec | 1 | 0.01 (1262) | 0 | 0.01 | 88.61/73.64 | 88.05/75.64 | 99.41 |
| B/epn612/dec | 2 | 0.1 (1678) | 0 | 0.01 | 87.56/78.14 | 87.64/77.56 | 99.41 |
| B/epn612/dec | 3 | 0.01 (1345) | 0 | 0.001 | 91.55/75.28 | 91.72/74.11 | 99.42 |
| B/epn612/pool | 0 | 1 (1672) | 0 | 0.001 | 88.62/76.97 | 89.04/77.69 | 99.90 |
| B/epn612/pool | 1 | 10 (1858) | 0 | 0.001 | 88.93/75.72 | 88.95/75.17 | 99.89 |
| B/epn612/pool | 2 | 1 (1309) | 0 | 0.0 | 89.53/78.08 | 89.71/77.89 | 99.89 |
| B/epn612/pool | 3 | 1 (1117) | 0 | 0.001 | 88.61/75.39 | 88.82/75.36 | 99.90 |
| B/ninapro_db2/dec | 0 | 0.001 (458) | 0 | 0.01 | 45.04/20.38 | 44.56/19.94 | 97.33 |
| B/ninapro_db2/dec | 1 | 0.001 (435) | 0 | 0.01 | 43.52/20.72 | 43.37/20.45 | 97.34 |
| B/ninapro_db2/dec | 2 | 0.001 (454) | 0 | 0.01 | 46.41/17.73 | 46.13/17.06 | 97.42 |
| B/ninapro_db2/dec | 3 | 0.001 (422) | 0 | 0.01 | 45.26/20.84 | 45.61/19.19 | 97.37 |
| B/ninapro_db2/pool | 0 | 0.01 (215) | 0 | 0.001 | 39.95/21.82 | 40.27/20.27 | 99.55 |
| B/ninapro_db2/pool | 1 | 0.01 (218) | 0 | 0.01 | 31.32/20.94 | 31.84/22.96 | 99.54 |
| B/ninapro_db2/pool | 2 | 0.01 (245) | 0 | 0.01 | 34.21/17.52 | 33.55/18.25 | 99.56 |
| B/ninapro_db2/pool | 3 | 100 (579) | 0 | 0.0 | 42.24/17.81 | 41.17/17.79 | 99.56 |
| B/ninapro_db3/dec | 0 | 0.001 (291) | 1 | 0.01 | 58.86/4.90 | 58.66/3.86 | 98.39 |
| B/ninapro_db3/dec | 1 | 1 (2036) | 0 | 0.001 | 56.41/4.63 | 57.23/4.17 | 98.12 |
| B/ninapro_db3/dec | 2 | 0.001 (336) | 1 | 0.01 | 53.72/3.59 | 53.75/0.34 | 98.50 |
| B/ninapro_db3/dec | 3 | 10 (4429) | 0 | 0.0 | 55.91/2.04 | 55.06/1.64 | 98.24 |
| B/ninapro_db3/pool | 0 | 10 (1384) | 0 | 0.0 | 48.63/2.43 | 47.77/2.94 | 99.70 |
| B/ninapro_db3/pool | 1 | 100 (1642) | 0 | 0.01 | 35.06/3.10 | 35.98/2.50 | 99.72 |
| B/ninapro_db3/pool | 2 | 1 (834) | 0 | 0.01 | 38.68/1.25 | 38.68/2.07 | 99.71 |
| B/ninapro_db3/pool | 3 | 0.001 (77) | 0 | 0.01 | 34.17/3.44 | 34.46/3.54 | 99.69 |
| B/ninapro_db6/dec | 0 | 0.01 (502) | 0 | 0.01 | 74.96/29.86 | 74.67/32.49 | 98.40 |
| B/ninapro_db6/dec | 1 | 0.001 (428) | 0 | 0.001 | 80.17/31.42 | 79.90/32.80 | 98.33 |
| B/ninapro_db6/dec | 2 | 1 (698) | 0 | 0.01 | 77.41/26.54 | 77.54/32.06 | 98.40 |
| B/ninapro_db6/dec | 3 | 10 (571) | 0 | 0.0 | 81.87/35.80 | 81.60/34.93 | 98.50 |
| B/ninapro_db6/pool | 0 | 0.001 (118) | 0 | 0.01 | 68.10/37.50 | 68.49/38.02 | 99.55 |
| B/ninapro_db6/pool | 1 | 0.001 (119) | 0 | 0.01 | 70.98/34.49 | 70.75/33.97 | 99.58 |
| B/ninapro_db6/pool | 2 | 0.01 (256) | 0 | 0.01 | 69.61/30.76 | 70.03/33.37 | 99.62 |
| B/ninapro_db6/pool | 3 | 0.01 (209) | 0 | 0.0 | 78.06/37.27 | 77.81/37.58 | 99.62 |
| B/uci_emg/dec | 0 | 0.001 (101) | 0 | 0.01 | 99.28/92.46 | 99.28/79.76 | 99.56 |
| B/uci_emg/dec | 1 | 0.1 (108) | 0 | 0.001 | 99.55/87.30 | 100.00/92.46 | 99.59 |
| B/uci_emg/dec | 2 | 0.001 (68) | 0 | 0.01 | 99.19/70.83 | 100.00/73.61 | 99.50 |
| B/uci_emg/dec | 3 | 0.001 (94) | 0 | 0.01 | 99.23/81.03 | 99.58/77.70 | 99.49 |
| B/uci_emg/pool | 0 | 0.001 (56) | 0 | 0.01 | 92.26/74.21 | 93.49/88.39 | 99.92 |
| B/uci_emg/pool | 1 | 100 (43) | 0 | 0.01 | 91.61/78.97 | 92.23/76.19 | 99.92 |
| B/uci_emg/pool | 2 | 0.001 (43) | 0 | 0.01 | 97.14/77.78 | 97.15/70.83 | 99.90 |
| B/uci_emg/pool | 3 | 0.001 (56) | 0 | 0.01 | 91.11/81.59 | 93.49/74.92 | 99.90 |
| R3c/epn612/dec | 0 | 100 (6222) | 0 | 0.01 | 91.63/80.44 | 91.85/79.58 | 99.42 |
| R3c/epn612/dec | 1 | 0.01 (1288) | 0 | 0.01 | 91.85/79.94 | 91.72/79.67 | 99.42 |
| R3c/epn612/dec | 2 | 0.1 (1610) | 0 | 0.01 | 91.53/83.28 | 91.70/82.44 | 99.43 |
| R3c/epn612/dec | 3 | 0.01 (1109) | 0 | 0.01 | 91.94/78.92 | 91.89/79.81 | 99.43 |
| R3c/epn612/pool | 0 | 0.1 (876) | 0 | 0.0 | 93.60/79.83 | 93.57/79.64 | 99.89 |
| R3c/epn612/pool | 1 | 10 (2031) | 0 | 0.001 | 92.03/79.44 | 91.95/77.67 | 99.89 |
| R3c/epn612/pool | 2 | 1 (1154) | 0 | 0.001 | 91.83/82.39 | 91.60/83.44 | 99.89 |
| R3c/epn612/pool | 3 | 0.1 (722) | 0 | 0.0 | 93.47/78.89 | 93.23/78.78 | 99.89 |
| R3c/ninapro_db2/dec | 0 | 0.001 (452) | 0 | 0.01 | 49.47/19.96 | 48.71/20.37 | 97.64 |
| R3c/ninapro_db2/dec | 1 | 0.001 (410) | 0 | 0.01 | 47.28/22.76 | 46.71/20.97 | 97.64 |
| R3c/ninapro_db2/dec | 2 | 0.01 (897) | 0 | 0.01 | 50.61/17.30 | 50.44/17.82 | 97.71 |
| R3c/ninapro_db2/dec | 3 | 0.001 (443) | 0 | 0.01 | 49.07/18.58 | 49.97/18.95 | 97.67 |
| R3c/ninapro_db2/pool | 0 | 0.01 (227) | 0 | 0.01 | 34.34/20.08 | 34.30/21.09 | 99.59 |
| R3c/ninapro_db2/pool | 1 | 0.1 (351) | 0 | 0.001 | 42.64/23.38 | 42.43/21.77 | 99.56 |
| R3c/ninapro_db2/pool | 2 | 0.01 (233) | 0 | 0.01 | 36.11/17.11 | 36.29/17.78 | 99.58 |
| R3c/ninapro_db2/pool | 3 | 0.1 (373) | 0 | 0.001 | 43.92/19.08 | 43.84/19.96 | 99.59 |
| R3c/ninapro_db3/dec | 0 | 0.001 (235) | 0 | 0.0 | 66.38/3.31 | 65.85/1.86 | 98.55 |
| R3c/ninapro_db3/dec | 1 | 0.001 (302) | 0 | 0.0 | 62.21/1.93 | 62.46/2.28 | 98.36 |
| R3c/ninapro_db3/dec | 2 | 10 (6398) | 0 | 0.01 | 57.56/1.01 | 58.34/1.06 | 98.66 |
| R3c/ninapro_db3/dec | 3 | 1 (2410) | 0 | 0.001 | 59.92/2.29 | 59.15/2.66 | 98.40 |
| R3c/ninapro_db3/pool | 0 | 0.01 (245) | 0 | 0.001 | 50.50/3.29 | 49.43/1.76 | 99.75 |
| R3c/ninapro_db3/pool | 1 | 0.01 (213) | 0 | 0.01 | 35.66/4.40 | 36.01/3.35 | 99.74 |
| R3c/ninapro_db3/pool | 2 | 1 (783) | 0 | 0.01 | 37.49/0.76 | 38.95/2.52 | 99.76 |
| R3c/ninapro_db3/pool | 3 | 1 (574) | 0 | 0.01 | 34.57/3.33 | 35.12/3.30 | 99.74 |
| R3c/ninapro_db6/dec | 0 | 0.001 (433) | 0 | 0.01 | 77.66/31.53 | 77.90/31.39 | 98.59 |
| R3c/ninapro_db6/dec | 1 | 0.001 (416) | 0 | 0.0 | 82.35/35.33 | 82.37/31.81 | 98.53 |
| R3c/ninapro_db6/dec | 2 | 10 (803) | 0 | 0.01 | 79.88/31.91 | 79.77/30.53 | 98.57 |
| R3c/ninapro_db6/dec | 3 | 0.01 (540) | 0 | 0.01 | 81.02/37.98 | 80.95/35.08 | 98.64 |
| R3c/ninapro_db6/pool | 0 | 0.001 (125) | 0 | 0.01 | 70.62/36.34 | 70.70/35.91 | 99.60 |
| R3c/ninapro_db6/pool | 1 | 0.001 (127) | 0 | 0.001 | 79.38/35.95 | 79.35/34.82 | 99.61 |
| R3c/ninapro_db6/pool | 2 | 0.1 (433) | 0 | 0.01 | 72.86/33.30 | 72.52/30.99 | 99.64 |
| R3c/ninapro_db6/pool | 3 | 0.01 (223) | 0 | 0.001 | 80.46/40.21 | 80.08/40.76 | 99.66 |
| R3c/uci_emg/dec | 0 | 1 (136) | 0 | 0.001 | 100.00/84.92 | 100.00/92.86 | 99.57 |
| R3c/uci_emg/dec | 1 | 1 (64) | 0 | 0.001 | 99.17/86.90 | 99.12/75.40 | 99.61 |
| R3c/uci_emg/dec | 2 | 0.001 (95) | 0 | 0.001 | 99.62/70.83 | 100.00/75.00 | 99.53 |
| R3c/uci_emg/dec | 3 | 0.01 (198) | 0 | 0.01 | 98.48/83.81 | 100.00/77.14 | 99.51 |
| R3c/uci_emg/pool | 0 | 10 (92) | 0 | 0.01 | 94.95/79.76 | 96.75/88.10 | 99.92 |
| R3c/uci_emg/pool | 1 | 0.1 (177) | 0 | 0.01 | 96.48/86.90 | 95.68/71.03 | 99.92 |
| R3c/uci_emg/pool | 2 | 0.001 (73) | 0 | 0.01 | 95.86/66.67 | 96.57/69.44 | 99.91 |
| R3c/uci_emg/pool | 3 | 0.001 (51) | 0 | 0.001 | 93.54/77.30 | 94.23/78.25 | 99.91 |

## Durate per cella (secondi di fit sommati sui quattro fold)

| cella | logistica | MLP | PCA |
|---|---:|---:|---:|
| B/epn612/dec | 974 | 2492 | 54 |
| B/epn612/pool | 546 | 2372 | 21 |
| B/ninapro_db2/dec | 1134 | 4436 | 62 |
| B/ninapro_db2/pool | 550 | 4670 | 19 |
| B/ninapro_db3/dec | 1022 | 1127 | 21 |
| B/ninapro_db3/pool | 299 | 1180 | 7 |
| B/ninapro_db6/dec | 172 | 1710 | 72 |
| B/ninapro_db6/pool | 95 | 1755 | 21 |
| B/uci_emg/dec | 1 | 40 | 1 |
| B/uci_emg/pool | 1 | 40 | 1 |
| R3c/epn612/dec | 1014 | 2522 | 55 |
| R3c/epn612/pool | 508 | 2520 | 23 |
| R3c/ninapro_db2/dec | 1311 | 4697 | 63 |
| R3c/ninapro_db2/pool | 568 | 4574 | 19 |
| R3c/ninapro_db3/dec | 1061 | 1199 | 24 |
| R3c/ninapro_db3/pool | 294 | 1213 | 6 |
| R3c/ninapro_db6/dec | 163 | 1736 | 75 |
| R3c/ninapro_db6/pool | 109 | 1616 | 18 |
| R3c/uci_emg/dec | 1 | 39 | 3 |
| R3c/uci_emg/pool | 1 | 39 | 1 |

## Confronto appaiato (bootstrap per soggetto, 2000 repliche, seme 20261029; scarto massimo fra punto ricalcolato e `bacc` del confronto: 0.00e+00)

### Aggregato P1 (pesi {'ninapro_db2': 0.6153846153846154, 'ninapro_db3': 0.23076923076923078, 'ninapro_db6': 0.15384615384615385})

| cella | BAcc | IC95 |
|---|---:|---|
| B/dec/logistic | 16.58 | [15.18; 17.95] |
| B/dec/mlp | 16.06 | [14.75; 17.31] |
| B/pool/logistic | 16.50 | [15.24; 17.82] |
| B/pool/mlp | 15.92 | [14.74; 17.17] |
| R3c/dec/logistic | 16.94 | [15.43; 18.46] |
| R3c/dec/mlp | 16.20 | [14.80; 17.59] |
| R3c/pool/logistic | 17.09 | [15.72; 18.47] |
| R3c/pool/mlp | 16.60 | [15.25; 17.97] |

| contrasto | delta (punti) | IC95 |
|---|---:|---|
| B_minus_R3c/dec/logistic | -0.37 | [-0.95; +0.16] |
| B_minus_R3c/dec/mlp | -0.14 | [-0.63; +0.35] |
| B_minus_R3c/pool/logistic | -0.60 | [-1.10; -0.10] |
| B_minus_R3c/pool/mlp | -0.68 | [-1.14; -0.18] |
| dec_minus_pool/B/logistic | +0.08 | [-0.55; +0.67] |
| dec_minus_pool/B/mlp | +0.14 | [-0.45; +0.69] |
| dec_minus_pool/R3c/logistic | -0.15 | [-0.84; +0.58] |
| dec_minus_pool/R3c/mlp | -0.41 | [-0.93; +0.14] |
| interaction/B | +0.06 | [-0.42; +0.52] |
| interaction/R3c | -0.25 | [-0.82; +0.31] |
| mlp_minus_logistic/B/dec | -0.52 | [-0.89; -0.16] |
| mlp_minus_logistic/B/pool | -0.58 | [-1.03; -0.11] |
| mlp_minus_logistic/R3c/dec | -0.75 | [-1.24; -0.27] |
| mlp_minus_logistic/R3c/pool | -0.49 | [-0.95; -0.02] |

### Aggregato P2 (pesi {'epn612': 0.5, 'uci_emg': 0.5})

| cella | BAcc | IC95 |
|---|---:|---|
| B/dec/logistic | 77.36 | [72.74; 81.21] |
| B/dec/mlp | 76.46 | [72.39; 80.09] |
| B/pool/logistic | 76.75 | [72.89; 80.54] |
| B/pool/mlp | 77.45 | [73.79; 80.82] |
| R3c/dec/logistic | 78.42 | [73.29; 82.47] |
| R3c/dec/mlp | 79.07 | [74.64; 82.84] |
| R3c/pool/logistic | 79.09 | [74.78; 82.85] |
| R3c/pool/mlp | 79.89 | [76.12; 83.43] |

| contrasto | delta (punti) | IC95 |
|---|---:|---|
| B_minus_R3c/dec/logistic | -1.05 | [-2.78; +0.84] |
| B_minus_R3c/dec/mlp | -2.62 | [-4.37; -0.78] |
| B_minus_R3c/pool/logistic | -2.33 | [-4.00; -0.72] |
| B_minus_R3c/pool/mlp | -2.44 | [-3.65; -1.27] |
| dec_minus_pool/B/logistic | +0.61 | [-1.25; +2.41] |
| dec_minus_pool/B/mlp | -0.99 | [-2.40; +0.30] |
| dec_minus_pool/R3c/logistic | -0.67 | [-3.08; +1.67] |
| dec_minus_pool/R3c/mlp | -0.82 | [-3.53; +1.69] |
| interaction/B | -1.60 | [-3.17; +0.05] |
| interaction/R3c | -0.15 | [-2.20; +2.05] |
| mlp_minus_logistic/B/dec | -0.91 | [-2.42; +0.76] |
| mlp_minus_logistic/B/pool | +0.70 | [-0.64; +1.97] |
| mlp_minus_logistic/R3c/dec | +0.66 | [-0.90; +2.18] |
| mlp_minus_logistic/R3c/pool | +0.80 | [-1.15; +2.66] |

### Contrasti per dataset (delta in punti, IC95)

| dataset | contrasto | delta | IC95 |
|---|---|---:|---|
| epn612 | B_minus_R3c/dec/logistic | -5.45 | [-6.05; -4.85] |
| epn612 | B_minus_R3c/dec/mlp | -4.61 | [-5.14; -4.03] |
| epn612 | B_minus_R3c/pool/logistic | -3.99 | [-4.69; -3.30] |
| epn612 | B_minus_R3c/pool/mlp | -3.45 | [-4.05; -2.88] |
| epn612 | dec_minus_pool/B/logistic | -0.40 | [-1.11; +0.27] |
| epn612 | dec_minus_pool/B/mlp | -0.48 | [-1.06; +0.09] |
| epn612 | dec_minus_pool/R3c/logistic | +1.05 | [+0.42; +1.63] |
| epn612 | dec_minus_pool/R3c/mlp | +0.68 | [+0.13; +1.22] |
| epn612 | interaction/B | -0.08 | [-0.75; +0.55] |
| epn612 | interaction/R3c | -0.37 | [-0.99; +0.24] |
| epn612 | mlp_minus_logistic/B/dec | +0.81 | [+0.32; +1.33] |
| epn612 | mlp_minus_logistic/B/pool | +0.89 | [+0.41; +1.38] |
| epn612 | mlp_minus_logistic/R3c/dec | -0.02 | [-0.51; +0.47] |
| epn612 | mlp_minus_logistic/R3c/pool | +0.35 | [-0.13; +0.80] |
| ninapro_db2 | B_minus_R3c/dec/logistic | -0.12 | [-0.91; +0.61] |
| ninapro_db2 | B_minus_R3c/dec/mlp | -0.22 | [-0.82; +0.40] |
| ninapro_db2 | B_minus_R3c/pool/logistic | -0.91 | [-1.51; -0.27] |
| ninapro_db2 | B_minus_R3c/pool/mlp | -0.95 | [-1.57; -0.27] |
| ninapro_db2 | dec_minus_pool/B/logistic | +0.94 | [+0.13; +1.78] |
| ninapro_db2 | dec_minus_pool/B/mlp | +0.70 | [-0.19; +1.53] |
| ninapro_db2 | dec_minus_pool/R3c/logistic | +0.15 | [-0.89; +1.25] |
| ninapro_db2 | dec_minus_pool/R3c/mlp | -0.03 | [-0.72; +0.70] |
| ninapro_db2 | interaction/B | -0.24 | [-0.80; +0.30] |
| ninapro_db2 | interaction/R3c | -0.19 | [-0.97; +0.61] |
| ninapro_db2 | mlp_minus_logistic/B/dec | -0.87 | [-1.31; -0.41] |
| ninapro_db2 | mlp_minus_logistic/B/pool | -0.63 | [-1.14; -0.10] |
| ninapro_db2 | mlp_minus_logistic/R3c/dec | -0.77 | [-1.37; -0.18] |
| ninapro_db2 | mlp_minus_logistic/R3c/pool | -0.59 | [-1.09; -0.04] |
| ninapro_db3 | B_minus_R3c/dec/logistic | -0.42 | [-1.03; +0.24] |
| ninapro_db3 | B_minus_R3c/dec/mlp | +0.32 | [-0.22; +0.70] |
| ninapro_db3 | B_minus_R3c/pool/logistic | +0.27 | [-0.89; +1.58] |
| ninapro_db3 | B_minus_R3c/pool/mlp | -0.15 | [-0.88; +0.68] |
| ninapro_db3 | dec_minus_pool/B/logistic | -0.05 | [-1.45; +1.13] |
| ninapro_db3 | dec_minus_pool/B/mlp | +0.75 | [+0.01; +1.40] |
| ninapro_db3 | dec_minus_pool/R3c/logistic | +0.64 | [+0.29; +0.93] |
| ninapro_db3 | dec_minus_pool/R3c/mlp | +0.27 | [-0.36; +1.02] |
| ninapro_db3 | interaction/B | +0.80 | [-0.37; +1.90] |
| ninapro_db3 | interaction/R3c | -0.36 | [-0.86; +0.37] |
| ninapro_db3 | mlp_minus_logistic/B/dec | +0.62 | [-0.17; +1.21] |
| ninapro_db3 | mlp_minus_logistic/B/pool | -0.18 | [-1.03; +0.60] |
| ninapro_db3 | mlp_minus_logistic/R3c/dec | -0.12 | [-0.46; +0.31] |
| ninapro_db3 | mlp_minus_logistic/R3c/pool | +0.24 | [-0.20; +0.60] |
| ninapro_db6 | B_minus_R3c/dec/logistic | -1.25 | [-2.98; +0.50] |
| ninapro_db6 | B_minus_R3c/dec/mlp | -0.52 | [-2.40; +1.48] |
| ninapro_db6 | B_minus_R3c/pool/logistic | -0.65 | [-1.89; +0.46] |
| ninapro_db6 | B_minus_R3c/pool/mlp | -0.41 | [-1.74; +1.03] |
| ninapro_db6 | dec_minus_pool/B/logistic | -3.16 | [-4.44; -1.98] |
| ninapro_db6 | dec_minus_pool/B/mlp | -3.02 | [-3.95; -2.01] |
| ninapro_db6 | dec_minus_pool/R3c/logistic | -2.56 | [-3.99; -0.95] |
| ninapro_db6 | dec_minus_pool/R3c/mlp | -2.91 | [-4.41; -1.05] |
| ninapro_db6 | interaction/B | +0.15 | [-1.20; +1.52] |
| ninapro_db6 | interaction/R3c | -0.35 | [-1.95; +1.04] |
| ninapro_db6 | mlp_minus_logistic/B/dec | -0.84 | [-1.93; +0.16] |
| ninapro_db6 | mlp_minus_logistic/B/pool | -0.98 | [-2.97; +0.91] |
| ninapro_db6 | mlp_minus_logistic/R3c/dec | -1.56 | [-3.54; +0.11] |
| ninapro_db6 | mlp_minus_logistic/R3c/pool | -1.22 | [-3.15; +0.83] |
| uci_emg | B_minus_R3c/dec/logistic | +3.34 | [-0.15; +7.08] |
| uci_emg | B_minus_R3c/dec/mlp | -0.62 | [-4.17; +2.98] |
| uci_emg | B_minus_R3c/pool/logistic | -0.67 | [-3.85; +2.37] |
| uci_emg | B_minus_R3c/pool/mlp | -1.43 | [-3.71; +0.83] |
| uci_emg | dec_minus_pool/B/logistic | +1.62 | [-2.06; +5.06] |
| uci_emg | dec_minus_pool/B/mlp | -1.50 | [-4.27; +0.96] |
| uci_emg | dec_minus_pool/R3c/logistic | -2.39 | [-7.18; +2.26] |
| uci_emg | dec_minus_pool/R3c/mlp | -2.31 | [-7.52; +2.74] |
| uci_emg | interaction/B | -3.13 | [-6.23; +0.08] |
| uci_emg | interaction/R3c | +0.07 | [-4.01; +4.46] |
| uci_emg | mlp_minus_logistic/B/dec | -2.63 | [-5.59; +0.69] |
| uci_emg | mlp_minus_logistic/B/pool | +0.50 | [-2.13; +2.94] |
| uci_emg | mlp_minus_logistic/R3c/dec | +1.33 | [-1.86; +4.40] |
| uci_emg | mlp_minus_logistic/R3c/pool | +1.26 | [-2.58; +4.93] |
