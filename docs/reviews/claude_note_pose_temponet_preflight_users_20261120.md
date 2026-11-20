# Nota Claude: preflight TCN fermato sugli utenti della cache (20/11/2026)

Oggetto: job `TCNH-preflight-20261119` = **64931573**, `FAILED 1:0` dopo 1 min 13 s su login13 (2 CPU).
- Codice: `c531e76c72c609ea22339b4a7393be71a8b53169`.
- Piano: `e3f5b119b19af08274d2df0744ef3772323f4325bcb4d50e72804b56235ef261`.
- Errore: `cache user population differs from original producer`, in `scripts/pose_temponet/provenance.py:439`.

Richiesta: messaggio di GPT inoltrato da Simone il 20/11.

Metodo, tutto in sola lettura:
- log del job e `sacct`;
- codice a `c531e76c` e al produttore storico `7d276c98`;
- campo `meta_json` delle due cache del piano su Leonardo.

Nessun job, nessun agente. Evidenze in `claude_note_pose_temponet_preflight_users_20261120_evidence/`.

## Esito

**Non è una differenza di utenti: è un errore di formato nel controllo.** Le due popolazioni coincidono una a una. Nella cache
l'utente è scritto `03d49b393e`, nella lista ricavata dalla ricevuta `u03d49b393e`.

## Che cosa confronta il controllo

- **Utenti della cache.** È il campo `user` di `meta_json`, scritto da `pose_eval.extract`
  (`src/wearusfm/harness/pose_eval.py:184`, `meta.append({"user": rec["user"], ...})`). È l'ID dell'utente, per esempio
  `03d49b393e`.
- **Utenti della ricevuta storica.** `historical_producer` (`provenance.py:278-282`) li ricava dai percorsi dei file grezzi: prende
  il nome della cartella dopo `emg2pose`. Ma quella cartella `pose_eval` la crea come `u` + ID (`pose_eval.py:70`,
  `f"u{_SAFE_RE.sub('_', r['user'])}"`), per esempio `u03d49b393e`.

## Verifica sui dati veri

Ho letto `meta_json` dalle due cache del piano con `zipfile`, senza numpy e senza aprire altro (`read_cache_users.py`). Il confronto
con il piano rilasciato è in `compare_users.py`, con l'uscita in `compare_users.txt`.

| Split | Finestre | Utenti nella cache | Utenti nella ricevuta | Uguali come stringhe | Uguali come `u` + ID |
| --- | ---: | ---: | ---: | --- | --- |
| train | 1.907 | 40 | 40 | no | sì |
| val | 18.548 | 15 | 15 | no | sì |

Tutti i 55 ID hanno 10 cifre esadecimali. Su di loro `_SAFE_RE` (`wearusfm.ingest.emg2pose`, riga 48) non cambia niente, quindi
la corrispondenza fra ID e cartella è esatta in tutti e due i versi.

## Perché nessun test l'ha visto

- `test_pose_temponet_provenance.py:83` usa la ricevuta vera, ma conta soltanto gli utenti (40) e non li confronta con una cache.
- Le fixture dei dati chiamano gli utenti `u0`, `u1`, ... (`test_pose_temponet_data.py:14`), quindi la `u` è già nel nome.
- La catena dei 24 stadi simula il preflight. Non l'ha visto neanche la mia rilettura `85f0d62`.

## La correzione va fatta alla fonte

La stessa lista `plan["producers"]["train"]["users"]` entra anche negli stadi successivi:
- benchmark: `benchmark._validate_train`, uguaglianza (`benchmark.py:73`);
- calibrazione WLS e controlli: `controls.validate_outer_partition`, inclusione (`controls.py:66`, chiamata da `runner.py:289`);
- fold di controlli, selezioni e rifit: `workflow.py:141`, uguaglianza.

Tutti questi falliscono in modo esplicito, nessuno passa in silenzio. La guardia dell'inferenza contro la sovrapposizione fra
sviluppo e training (`workflow.py:490`) confronta utenti della cache con utenti della cache, quindi non è indebolita. Però
**correggere soltanto la riga 439 sposterebbe il fermo al benchmark.**

Proposta:
- `historical_producer` restituisce gli utenti nel formato della cache.
- Per ogni cartella pretende la forma `u` + ID, e che `"u" + _SAFE_RE.sub("_", ID)` ridia esattamente il nome della cartella.
  `_SAFE_RE` va importato da `wearusfm.ingest.emg2pose`, lo stesso del produttore.
- Le cartelle originali si possono conservare a parte, se servono alla traccia.

Test di regressione:
1. Uno con la disposizione vera: `emg2pose/u<ID>/<stem>/...` nella ricevuta e `<ID>` nella cache.
2. Uno sulle due ricevute vere già nel repo (`docs/reviews/pose_temponet_runtime_evidence_20261119/producer_{train,val}.json`),
   confrontate con gli elenchi veri della cache (`cache_users.json`).

## Il resto del preflight, letto sul codice

- Dopo gli utenti, il preflight confronta sei dettagli: `targets`, `key_time_valid_sha256`, `dim`, `n_recordings`, `n_users` e
  `n_windows`. Li calcola `consolidation_long_pose.validate_cache`. I quattro file da cui dipende (`consolidation_long_pose.py`,
  `tokenizer_pose_readout.py`, `pose_eval.py`, `pose_dynamics.py`) sono identici fra il produttore `7d276c98` e `c531e76c`.
- I due fold da 20 utenti seguono dai 40 utenti.
- La parità R-1 a 1e-12 è numerica. Sulla carta non si verifica: la decide il gate.

## Che cosa serve per ripartire

1. La correzione cambia i sorgenti. Quindi servono nuovo `source_sha256` e nuovo piano, e per L3-9 ID e checkout nuovi; il checkout
   `pose_temponet_20261119` non si riusa.
2. `release-plan` vuole una review che nomini il nuovo SHA e il nuovo `source_sha256`. Rileggo solo il diff, senza agenti.
3. Il nuovo tentativo richiede una decisione di Simone (L3-9).

Costo del job fallito: 1 min 13 s su 2 CPU, 0 GPU-ore. I benchmark non sono stati inviati, ed è corretto.
