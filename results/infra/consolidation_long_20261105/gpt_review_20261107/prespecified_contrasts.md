# Contrasti prespecificati — valutazione lunga

Fonte immutabile: `c109bfef5a3f80cc4b2334e3554c2171db938467`. IC95% appaiati per soggetto, 2.000 repliche, seed 20261102. Nessun aggiustamento per confronti multipli.

## Classificazione

| Contrasto | P1, pp [IC95%] | P2, pp [IC95%] | V-R1 congiunta |
|---|---:|---:|---|
| 19618:post-pre | +1.70 [+1.07; +2.38] | +2.59 [+0.62; +4.79] | practical_benefit |
| 39237:post-pre | +2.12 [+1.51; +2.75] | +2.16 [+0.35; +4.07] | practical_benefit |
| 49046:post-pre | +1.58 [+0.84; +2.30] | +3.91 [+1.74; +5.86] | practical_benefit |
| pool:39237-19618 | +0.33 [-0.15; +0.80] | +0.24 [-1.33; +1.63] | small_or_inconclusive |
| pool:49046-19618 | +0.35 [-0.22; +0.91] | +1.35 [-0.13; +2.71] | small_or_inconclusive |
| pool:49046-39237 | +0.02 [-0.46; +0.48] | +1.12 [-0.11; +2.41] | small_or_inconclusive |
| post:39237-19618 | +0.47 [+0.01; +1.00] | +0.05 [-1.13; +1.26] | small_or_inconclusive |
| post:49046-19618 | +0.52 [-0.09; +1.15] | +1.52 [+0.27; +2.89] | practical_benefit |
| post:49046-39237 | +0.05 [-0.34; +0.41] | +1.47 [+0.17; +2.85] | practical_benefit |
| pre:39237-19618 | +0.06 [-0.39; +0.57] | +0.48 [-0.78; +1.76] | small_or_inconclusive |
| pre:49046-19618 | +0.64 [+0.13; +1.18] | +0.20 [-2.15; +2.80] | small_or_inconclusive |
| pre:49046-39237 | +0.58 [+0.19; +0.96] | -0.27 [-2.65; +2.23] | small_or_inconclusive |

## Posa

E_dyn: riduzione relativa positiva = miglioramento; delta E2 positivo = maggior vantaggio dell'allineamento. Gruppo dynamic per E_dyn; E2 sul proprio supporto comune crop6. I due contrasti primari sono post_ridge e trained_head 49.046−19.618.

| Contrasto | Riduzione E_dyn, % [IC95%] | Delta gap E2, rad [IC95%] | Lettura E_dyn |
|---|---:|---:|---|
| 19618/post_ridge_minus_19618/pre_ridge | +18.97 [+17.00; +21.28] | +0.05351 [+0.04937; +0.05803] | practical_benefit |
| 39237/post_ridge_minus_19618/post_ridge | +3.48 [+3.03; +3.93] | +0.01010 [+0.00805; +0.01201] | practical_benefit |
| 39237/post_ridge_minus_39237/pre_ridge | +21.63 [+19.58; +23.95] | +0.06330 [+0.05850; +0.06786] | practical_benefit |
| 39237/pre_ridge_minus_19618/pre_ridge | +0.21 [+0.06; +0.37] | +0.00031 [+0.00001; +0.00065] | small |
| 39237/trained_head_minus_19618/trained_head | +3.36 [+2.75; +3.89] | +0.01128 [+0.00887; +0.01343] | practical_benefit |
| 49046/post_ridge_minus_19618/post_ridge | +6.32 [+5.87; +6.83] | +0.01871 [+0.01620; +0.02088] | practical_benefit |
| 49046/post_ridge_minus_39237/post_ridge | +2.94 [+2.62; +3.30] | +0.00861 [+0.00740; +0.00971] | practical_benefit |
| 49046/post_ridge_minus_49046/pre_ridge | +23.86 [+21.83; +26.17] | +0.07159 [+0.06641; +0.07644] | practical_benefit |
| 49046/pre_ridge_minus_19618/pre_ridge | +0.31 [+0.16; +0.48] | +0.00063 [+0.00033; +0.00099] | small |
| 49046/pre_ridge_minus_39237/pre_ridge | +0.10 [+0.01; +0.19] | +0.00032 [+0.00019; +0.00044] | small |
| 49046/trained_head_minus_19618/trained_head | +6.52 [+5.92; +7.15] | +0.02040 [+0.01759; +0.02310] | practical_benefit |
| 49046/trained_head_minus_39237/trained_head | +3.28 [+2.84; +3.70] | +0.00913 [+0.00770; +0.01067] | practical_benefit |

## Altri gruppi della posa

| Contrasto | Riduzione E_dyn all, % [IC95%] | Riduzione E_dyn static, % [IC95%] |
|---|---:|---:|
| 19618/post_ridge_minus_19618/pre_ridge | +19.10 [+17.39; +21.12] | +19.64 [+18.17; +20.92] |
| 39237/post_ridge_minus_19618/post_ridge | +3.31 [+2.91; +3.64] | +2.60 [+1.92; +3.33] |
| 39237/post_ridge_minus_39237/pre_ridge | +21.57 [+19.86; +23.52] | +21.34 [+20.02; +22.51] |
| 39237/pre_ridge_minus_19618/pre_ridge | +0.26 [+0.11; +0.44] | +0.50 [+0.25; +0.76] |
| 39237/trained_head_minus_19618/trained_head | +3.20 [+2.61; +3.66] | +2.57 [+1.65; +3.36] |
| 49046/post_ridge_minus_19618/post_ridge | +5.94 [+5.54; +6.36] | +4.35 [+3.66; +5.05] |
| 49046/post_ridge_minus_39237/post_ridge | +2.72 [+2.41; +3.04] | +1.79 [+1.39; +2.23] |
| 49046/post_ridge_minus_49046/pre_ridge | +23.66 [+21.95; +25.62] | +22.85 [+21.32; +24.26] |
| 49046/pre_ridge_minus_19618/pre_ridge | +0.32 [+0.17; +0.49] | +0.37 [+0.13; +0.62] |
| 49046/pre_ridge_minus_39237/pre_ridge | +0.06 [-0.04; +0.15] | -0.13 [-0.34; +0.08] |
| 49046/trained_head_minus_19618/trained_head | +6.21 [+5.61; +6.76] | +4.88 [+3.98; +5.67] |
| 49046/trained_head_minus_39237/trained_head | +3.10 [+2.67; +3.52] | +2.38 [+1.81; +2.95] |
