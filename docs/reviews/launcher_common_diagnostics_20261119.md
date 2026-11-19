# Invii operativi dei diagnostici da stato comune

19/11/2026, 3-launcher. Review Claude 0afd0864, consegna db66b507,
piano 35ea94ba, codice congelato 365cb1e29bc410013d4ee0494b9c2e92835bc7f1.

Prima di ciascuno dei quattro invii: piano rigenerato identico byte per byte
a quello approvato; sei pin e submitter verificati; ID assente dal ledger;
quota debug ricontrollata e spazio disponibile circa 46 GiB (riserva 14 GiB).
Ref deploy pubblicato e verificato. Invii esclusivamente via submitter comune.

| Cella | ID SLURM | Primo stato raccolto |
|---|---:|---|
| checkout | 64835236 | COMPLETED 0:0, 35 s, 4 CPU |
| CUDA | 64835286 | PENDING Dependency |
| ripresa CN | 64835297 | PENDING Dependency |
| ripresa CR | 64835304 | PENDING Dependency |

Checkout HEAD verificato sul cluster: 365cb1e29bc410013d4ee0494b9c2e92835bc7f1.
Consuntivo checkout: 0 GPU-ore / 0,038889 core-ore.
Tetto GPU residuo prenotato: 6 GPU-ore. Nessun retry o duplicato.
Rapporti attesi in `$WORK/wearusfm_runs/results/contrastive_common_diagnostics_20261118/`
nelle sottocartelle cuda, resume_CN, resume_CR. Raccolta anche degli artefatti
negativi o parziali, con hash e accounting alla conclusione.
Lettura scientifica e decisioni su training/RMS100/fast125 restano alla chat
3-log quadratico e contrastive loss.

Snapshot successivo: CUDA e CN RUNNING (26 s ciascuno); CR PENDING
QOSMaxJobsPerUserLimit. Due sole celle debug in esecuzione; la terza è già in
coda e partirà automaticamente secondo quota, senza un nuovo invio.
