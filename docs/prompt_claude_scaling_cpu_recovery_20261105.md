# Review richiesta: recupero CPU scaling30M/300M dopo TIMEOUT

Claude, Simone chiede log e review del recupero; ripartiamo dopo convergenza.
**Solo review: nessun job, modifica di dati o rilancio da parte tua.**

Ramo GPT su origin: `codex/scaling-300m-20261104`. Leggi il commit che pubblica
questo prompt e i suoi artefatti; base precedente `76a3126`. Codice numerico
invariato `aef665ba7ea66fc80291846487907bb3ef9107df`.

## Evidenza e stato

- `docs/reviews/scaling_cpu_timeout_20261105.md`: diagnosi, contabilità, proposta.
- `results/infra/scaling_300m_20261104/cpu_readers_20261105/probe_benchmark/`:
 30 file originali con manifest/hash;13benchmark,14audit,3log; raw sacct e
 accounting per cella; `audit_collected.py` e `diagnosis.json` riproducibile.
- `docs/scaling_300m_cpu_execution_20261105.md`, `status.json` della stessa
 directory CPU, piani effettivi e ricevute submitter già pubblicati.
- Tua review `claude_review_scaling_300m_completion_extractions_20261105.md`,
 soprattutto O-6/O-7/O-8; launcher `scripts/slurm/scaling_readers.sbatch` e
 `scripts/scaling_probe.py::benchmark/validate_resources`.

63593262(30M) e63593593(300M matched) COMPLETED0:0. **63593508(300M naturale)
TIMEOUT**,5.584s su limite5.400s, batchCANCELLED0:15, picco3,839GiB su30GiB.
Tre NinaPro completi; EPN612 audit con cinque C conclusi fino a10; mancaC100.
UCI non iniziato. Tutti83fit registrati convergenti, nessuna metrica downstream.
Il codice non salva uno stato del solver per riprendere il C interrotto.

**Secondo blocco:** la formula congelata applicata ai13report completi dà già
148,134869core-ore, senza overhead120s/shard né i due benchmark mancanti.
Il tetto136core-ore dei fit non è fattibile neppure spezzando diversamente.
Consuntivo CPU-only attuale22,027778core-ore, incluso il TIMEOUT; zeroGPU.
Nessun fit posa/classificazione, retry o nuovo job è stato inviato.

## Proposta operativa da rivedere

1. Conservare training,14cache e13report completi. Riusare i tre report NinaPro
   del parent TIMEOUT solo dopo validazione esplicita, senza promuovere il padre
   a successo; conservare tutti i path/hash/stati/costi originali.
2. Recuperare soltanto i benchmark EPN612 e UCI300M naturale, con nuovi output
   e ID. Proposta di cap: EPN8CPU/30GiB/120min, UCI8CPU/30GiB/10min;
   17,333333core-ore, zeroGPU. EPN rifà tutti e sei C. Non sono stime garantite;
   un nuovo errore/TIMEOUT ferma la catena.
3. Preparare un dispatcher operativo separato, per-dataset, mantenendo il
   checkout numerico aef665ba, piano/cache/ricetta/runtime/login08 identici.
   Non cambiare tolleranza, griglia, max_iter, fold, scaler o larghezza primaria.
   Il dispatcher non è ancora implementato: questa è review del piano, non
   attestazione di codice già pronto.
4. Quando disponibili15report completi, stesso calcolo delle risorse e copertura
   esatta, nuovo budget fit e cumulativo pubblicato prima dei fit. Verificare
   se15shard bastano con2h/shard; nessun aumento nascosto di questi limiti.
   Non inventare adesso un totale misurato dai due benchmark ancora mancanti.
5. Conservare STOP fino a convergenza e verifica dei nuovi launcher/pin. Nessun
   cambio scientifico, nessun training o estrazione da rifare; la continuazione
  19.618 resta separata. Il budget S8 già congelato va trattato esplicitamente
   alla luce della nuova evidenza tecnica, senza cambiare criteri dopo gli score.

## Domande della review

1. Riproduci hash,13report,83fit, runtime e accounting; concordi con diagnosi
   TIMEOUT e con i limiti della verifica offline? Parent ExitCode0:0 non vale
   come successo; raw sacct resta accanto all'accounting.
2. Riproduci indipendentemente il limite inferiore148,134869core-ore e dimostra
   che il tetto136 è insufficiente indipendentemente dal numero di shard.
3. È corretto riusare i tre report completi del job fallito mantenendo traccia
   del parentTIMEOUT? Quali guardie concrete aggiungeresti al rilascio?
4. Sono adeguati il recupero limitato ai due dataset e i cap120/10min proposti,
   distinguendo tetti da stime? Proponi correzioni motivate prima di implementare.
5. Concordi con la sequenza recupero benchmark → budget completo → fit e con
   il mantenimento della formula scientifica? Segnala gli eventuali effetti
   sul freeze condizionale19.618, senza autorizzare quella tranche.

Rispondi con APPROVATO/DA RIVEDERE e motivazioni, distinguendo blocchi e facoltativi.
Registra la review nelle due sezioni, commit/push sul tuo ramo e SHA verificato.
Non sottomettere esperimenti.
