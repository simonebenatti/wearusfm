# Linea 1 — Recupero v3 della continuazione 300M 5.000 → 19.618

Data: 09/11/2026. Autore del codice e operatore: Claude. Simone, 09/11 alle 11:46: «allora puoi lanciare, no need per la
review di gpt, che ha finito i token»; alle 12:06, alla domanda «Preparo il v3 col cancello corretto e un nuovo pre-controllo
(solo CPU, +1,8 core-ore)?»: «Sì, prepara il v3».

## Perché c'è un v3

Il pacchetto v2 (`../continuation19618_20261109_v2/`) è arrivato fino al pre-controllo. Il job 63974196 ha trovato i tre file
d'identità uguali a quelli del run a 5.000, con `identical: true`. La sua ricevuta però portava `torch_before_numpy: false`, e il
training v2 esige `true`: il training non è stato inviato. La misura era sbagliata. `sys.modules` registra l'ordine di
completamento: Python sposta un modulo in fondo quando ne termina l'esecuzione. Torch importa numpy mentre carica `torch._C`,
quindi risulta sempre dopo numpy. Il v3 legge `"numpy" in sys.modules` subito prima di `import torch` e mette questa misura
nel cancello, accanto a `identical` e ai pin. Prove del v2 in `../continuation19618_20261106/evidence/checkout-training-v2/` e
`evidence/identity-precheck/`.

## Contenuto

- `training_plan.json`, SHA-256 `f450ec5d08baaac7d8e65f3e3eb4edd3c6559b91863e5df56e76ecf86798dbbf`. Uguale al v2 salvo: pin del wrapper (`c8ad4d241b4ed490ee444b3434f556ab341986f533e01ccaa6d68f50a6a60a33`) e del launcher del pre-controllo;
  protocollo del pre-controllo `-v2`; cartelle del run, delle ricevute e del pre-controllo `-v3`; blocco `superseded_package`
  sul v2. Ricetta, dose, allocazione, tetto, run a 5.000, controllo e orizzonte: invariati.
- `make_training_plan.py` e le letture del 09/11 alle 11:43 (`remote_listing.txt`, `remote_sha256.txt`, copia filtrata di
  `../continuation19618_20261106/evidence/recovery-reread-20261109/`), `remote_verification.json`.
- `remote_read.sh`: rilettura in sola lettura prima del deploy, con il `sacct -P` degli otto job già eseguiti.
- `make_envelope.py` e `budget_envelope.json` (SHA-256 `7ecc130e46c541f75f2343c87952013ef1feb24c535f0e53fdd915910aabceae`): consuntivo dalle prove attestate e inviluppo emendato.
- `training_submit_lines.txt`: le quattro righe d'invio, da `scripts/scaling_plans19618.py recovery-lines`.

## Inviluppo emendato

| Voce | Freeze v2 | Firmato v2 | v3 |
|---|---:|---:|---:|
| GPU-ore con margine | 66,35 | 66,731111 | 66,731111 |
| Core-ore CPU senza GPU, con margine (tetto 350) | 345,166667 | 347,584444 | 349,384444 |
| Fase checkout | 0,8 | 1,066667 | 1,333333 |
| Pre-controllo dell'identità | nessuna | 1,533333 | 3,066667 |
| Core-ore nei job GPU, con margine | 530,8 | 533,848889 | 533,848889 |

Il checkout e il pre-controllo del v2 contano alla loro allocazione con margine, come i job pianificati. Hanno usato 0,098889
core-ore reali. Nessuna GPU è stata usata dal v2.

## Sequenza

Ogni passo parte solo se il precedente è finito come atteso ed è stato letto. Stop al primo esito diverso, senza retry.
1. Rilettura con `remote_read.sh` e `make_training_plan.py <listing> <sha256> --check`, con esito `passed: true`.
2. Riferimento `deploy/scaling_300m_continuation19618_20261109_v3`, mai `main`.
3. Checkout congelato, `COMPLETED 0:0`, poi `collect checkout-training` (prove in `evidence/checkout-training-v3/`).
4. Pre-controllo, `COMPLETED 0:0`, poi `collect identity-precheck` (prove in `evidence/identity-precheck-v3/`): nella ricevuta
   `identical: true` e `numpy_loaded_before_torch: false`.
5. Training, solo con la suite completa `tests/cpu` verde sul commit di deploy. Poi `collect training` e lo stadio B.
