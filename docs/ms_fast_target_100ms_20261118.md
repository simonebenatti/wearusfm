# Sei bande del target rapido a100ms — 18/11/2026

**Superato il 18/11 sera:** la finestra del target rapido e' **125 ms**, versione `step1_psd_power_shape_energy_fast125_v3`.
A 100 ms la banda 20-33,6 Hz aveva due bin FFT a 1000 e 2000 Hz ma uno solo a 1926 e 2048 Hz, quindi lo stesso segnale dava un
target diverso secondo il dataset. La versione `..._fast100_v2` e' ritirata prima di ogni uso. Contratto aggiornato in
`docs/target_windows_fix_20261118.md`; il resto di questa pagina descrive la consegna intermedia a 100 ms.

Simone: «no dobbiamo almeno raddoppiarla, almeno 100ms anche lei». Seguito della consegna RMS100 `5333c5c`.

## Ricetta nuova

| Bersaglio | Supporto | Passo |
|---|---:|---:|
| LogRMS |100ms|25ms|
| Log-potenze rapide, sei bande (`ms_fast`) |100ms, prima50ms|25ms|
| Forma spettrale, cinque bande |200ms|25ms|
| Log-inviluppo |500ms|25ms|
| Log-potenze lente,24bande |500ms|100ms, assegnate alle quattro patch vicine|

Il nuovo default e' `loader.ms_fast_window_ms=100.0`, serializzato nella ricetta e nella firma dati.
Sei bande log-spaziate20–450Hz, stessa Hann simmetrica, normalizzazione della densita' monolatera e integrazione
per banda; non cambiano numero delle uscite, dimensione delle teste o pesi della loss. Le bande sopra il limite
effettivo del canale o senza bin FFT restano escluse. La posa resta invariata.

A200Hz si passa da10a20campioni: la griglia FFT passa da20a10Hz, senza creare frequenze oltre Nyquist.
L'avanzamento resta5campioni (25ms), overlap15campioni (75ms). L'arrotondamento dei mezzi campioni segue quello
del nuovo RMS100; a2048Hz il supporto e'205campioni. Supporti incompleti ai bordi sono mascherati; nessun padding
entra come osservazione valida. Le guardie dei transitori e le maschere estratte conservano la semantica esistente.

## Identita' e compatibilita'

- Versione nuova: `step1_psd_power_shape_energy_fast100_v2`.
- Versione storica a50ms: `step1_psd_power_shape_energy_v1`.
- CLI di training e calibrazione/estrazione: `--ms-fast-window-ms 100` (default), oppure50 esplicito per la ricetta
  storica. Il parametro sceglie anche la versione corretta; il trainer rifiuta supporto e versione incoerenti.
- I checkpoint senza il campo spettrale sono interpretati a50ms. Riprese e continuazioni non possono passare
  silenziosamente a100ms: la configurazione e la firma dati vengono confrontate prima del training.
- Un vecchio run completo richiede anche `--rms-window-ms 25`; la consegna intermedia RMS100/spettro50 resta
  riproducibile con RMS100 e `--ms-fast-window-ms 50`.
- Le calibrazioni per i nuovi target vanno rigenerate sul solo train. Il lettore delle cache spettrali controlla
  la coerenza fra versione e supporto registrato; una cache50 non viene dichiarata100 o viceversa.

Questo intervento riguarda i bersagli del pretraining. Il frontend dei precedenti rami PSD/wavelet rimane quello
della loro architettura storica50/500ms, e non e' attivo nella ricetta corrente senza rami. La baseline `stats/mask`
delle sonde viene esplicitamente fissata a50/500ms per evitare di cambiarne le cache indirettamente. Cambiare quegli
ingressi sarebbe un intervento distinto. Il decoder continua a predire i nuovi target100ms quando il loader usa
la nuova ricetta. Nessun piano congelato, checkpoint, cache esistente o job viene modificato.

## Verifiche

Oracolo indipendente su indici grezzi e FFT numpy in float64 a200/1000/2000/2048Hz: valori per banda, disponibilita',
bordi, frequenza delle uscite, overlap. Verificato che gli altri target restano identici cambiando solo la finestra
rapida. Test del loader reale, CLI, firma dati, rifiuto di versioni incoerenti, cache e riprese.

Confrontati tutti i campi dei target con il codice originale di `5333c5c` (RMS100/spettro50) e `e114a78`
(RMS25/spettro50):16casi per ciascuno, tutti identici bit per bit, incluse maschere e scala lenta.
Esiti della suite e consuntivo nel registro e in `results/infra/ms_fast_target_20261118/verification.json`.
Nessuna misura di efficacia scientifica e nessun lancio cluster in questa consegna.
