# Punto 6 — review prima del preflight, 02/11/2026

**Favorevole: nessun difetto aperto che blocchi il preflight.** Implementazione di estrattore
e lettori affidata a due agenti con file distinti; review scientifica e del codice indipendente
del critico, più controllo separato del launcher e delle risorse. Nessun job eseguito.

## Verifiche

- Quattro lettori M384/M48/S384/S720; il controllo M48 evita di confondere il contributo
  dei residui con quello della compressione. S720 conserva la media completa.
- Contrasti fissi indipendenti dai target, FP32, normalizzazione comune alla media storica;
  padding da lunghezze, nessuna maschera posa nelle feature.
- Estrazione con indici originali, scala e finestre storiche; identità esatta di target,
  supporto e metadati; parità delle medie verificata dopo FP16 e LayerNorm.
- CV e selezione solo sugli utenti train, intercetta non penalizzata, supporti comuni,
  otto celle e nove contrasti con gli stessi bootstrap degli utenti.
- Preflight obbligatorio prima di creare output/caricare il modello per l'estrazione completa.
  Ricevute parent/split complete, hash coerenti e metriche di parità verificate dai reader.
- Tutti i115 pin del piano coincidono; matrici congelate verificate indipendentemente.
  Il runtime non rigenera QR. Storage nuovo:8.797.286.400 byte per i payload dei due modelli.
- Tetto2,25 GPU-ore e28,333 core-ore, dei quali10,333 per i soli job CPU. Tempi da misurare
  prima delle due estrazioni complete e del report. I job B/R3c possono procedere in parallelo.

## Rilievi risolti prima della pubblicazione

1. Aggiunto M48: il solo confronto S384−M384 non isolerebbe il contributo dei contrasti.
2. Gate numerico fissato sulle feature normalizzate e sul denominatore; differenze grezze/ULP
   restano diagnostiche, senza ampliare soglie dopo il preflight.
3. Una ricevuta di split non basta se il job fallisce la guardia globale: ora è richiesta anche
   la ricevuta finale del modello. Test di mutazione aggiornano gli hash per raggiungere
   effettivamente le guardie scientifiche, anziché fermarsi solo al primo checksum.
4. Collegato il preflight alla CLI completa, con verifica ripetuta e attestazione nella ricevuta.

## Validazione locale e freeze

**54 test superati in8,17s** (`test_pose_slot_extract.py`, `test_pose_slot_readout.py`),
`bash -n scripts/slurm/pose_slots.sbatch` e `git diff --check` puliti.
Nessun modulo scientifico storico modificato; test mirati al nuovo percorso.

- Piano SHA256:`0dafa6d4a17cbe196b2508ec6cd4a31e32e6e117fd36d2c720ad96caf0145526`.
- Proiezioni SHA256:`d1e8897ad4f80f7bee788d3cc22808483e74215777ef2f65041a7987a77075c8`.
- SHA dei coefficienti slot:`7e7684abbd36c6e4375e62037c2fe362080406f14f6d3e9f9160fcfb13e69674`.
- SHA della proiezione canali:`e9a4932e1b39c2a3b0a8d9cbcd3896eb9944ca98948e518d0b21fb2b27640097`.

Restano i limiti del [protocollo](../pose_slots_protocol_20261102.md): sette contrasti e una sola
proiezione, capacità aggiuntiva nei confronti incrementali, val già osservata e IC esplorativi.
Controllare M384 contro il precedente istantaneo prima di interpretare i risultati.
Nessuna promozione automatica del FM o lancio di nuove loss.
