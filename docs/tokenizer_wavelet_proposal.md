# Tokenizer wavelet — proposta separata, non default

> Archivio del 21/10/2026: proposta GPT del 12/10 recuperata fra i file non tracciati.
> E' un'ipotesi storica sul tokenizer. I successivi rami PSD/wavelet e il loro pilot
> hanno protocolli separati; questo testo non descrive l'implementazione corrente.

12/10/2026, GPT. **Solo disegno sperimentale: nessuna modifica a frontend.py, FMConfig o checkpoint;
nessun training/nuovo job autorizzato.** Prima chiudere random-baseline e obiettivi della finestra1.

## Ipotesi e controllo minimo

Il banco attuale produce64 feature: Fourier24, spline24, MLP16 su patch25ms, griglia riferimento16kHz,
integrale con Δt, anti-alias alla Nyquist nativa e contesto100ms. Il revisore segnala spline poco utili
sopra~350Hz/ridondanti sotto: **ipotesi da verificare sul banco reale**, non proprieta' dimostrata qui.

Prima misurare risposta in frequenza per famiglia, Gram/rango e ridondanza, energia delle attivazioni,
gradienti ed effetto del dropout di famiglia sulle sonde. Distinguere analisi di un checkpoint
da ablation riaddestrata: azzerare feature in inferenza crea distribution shift.

Un solo candidato iniziale: sostituire le24 spline con24 kernel oscillanti localizzati tipo
Gabor/Morlet, mantenendo Fourier24+MLP16, dimensione64, proiezione e resto del modello invariati.
Niente nuova loss spettrale insieme al cambio front-end. Stesso numero di uscite **non** garantisce
stesso numero di parametri/FLOP: misurarlo e dichiararlo; controllo a capacita' confrontabile.

La motivazione e' localizzare contenuti nel tempo, non aggiungere semplicemente altre bande.
Morlet e' una sinusoide modulata da una gaussiana; ammissibilita'/localizzazione e rapporto
scala-frequenza vanno trattati esplicitamente ([Torrence & Compo, 1998](https://paos.colorado.edu/research/wavelets/bams_79_01_0061.pdf)).
Un banco finito di kernel su patch non e' automaticamente una trasformata wavelet continua.

## Vincoli tecnici prima del codice

- Stessa griglia fisica e integrazioneΔt del banco corrente; niente normalizzazione che dipenda
  arbitrariamente dal numero di campioni/fs e fornisca un identificatore di dataset.
- Kernel centrati/localizzati, media zero e norma definite sulla griglia fisica di riferimento.
  Finestratura/troncamento possono reintrodurre DC: verificare dopo finestratura e anti-aliasing,
  non assumere che la formula Morlet teorica basti. Norme/durata non cambiate per ogni dataset.
- Centri, frequenze, larghezze, parti reali/immaginarie, apprendibilita' e numero di parametri
  vanno congelati prima del confronto. Non rivendicare invarianza di fase dalle sole coppie quadratura.
- A20–40Hz una patch25ms non contiene molti cicli: non promettere risoluzione delle basse frequenze
  senza un supporto piu' lungo. Allungare il supporto sarebbe un secondo asse, non questa ablation.
- Preservare trattamento bordi e fs native; nessun ricampionamento nascosto. Nuovo front-end
  richiede nuovo identificatore/config e training da zero, non resume strict=False di un checkpoint.
- Contratto con Claude prima di nuovi campi di FMConfig/launcher; sviluppo GPT in proprio worktree,
  tokenizer corrente e checkout dei job invariati.

## Verifiche e criteri

Test locali sintetici: sinusoidi/bursts/chirp e ritardi, DC, Nyquist e fs native200/1000/2000Hz;
gradienti finiti, alias/bordi, shape/budget, ricampionamento band-limited identico in tempo fisico.
Poi **riaprire il gateD8** per la nuova famiglia: Kaifosh,450Hz2k→1k e90Hz2k→200,
3semi, errore relativoRMS<=5% nel totale e in ogni famiglia; probeA/B<=55% eIC95contiene50%,
stesse procedure/split firmati. Estendere il nome "spline" alla nuova famiglia richiede firma,
non un PASS nominale del vecchio banco.

Confronto eventuale 30M: banco attuale vs candidato, stessi seed/dati/masking/loss/endpoint.
P1/P2 con baselinecasuale per ogni architettura, costo memoria/tempo, sonde temporali e fisiche
e dataset-ID finali. L'R² spettrale da solo non decide l'adozione. Soglie/regola selezione e budget
si congelano PRIMA dei nuovi risultati; qui non vengono inventate dopo l'A/B.

Ordine proposto: diagnostica delle famiglie → design preciso e testCPU → gateD8 autorizzato →
eventuale ablationtraining autorizzata. Non aspettare inattivi il collaudoGPU, ma non inserire
le wavelet nel cammino critico della finestra1 prima di chiudere i debiti piu' urgenti.
