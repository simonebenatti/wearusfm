# Linea 2 — consegna operativa a Claude,05/11 sera

**Owner operativo: Claude. Codice e responsabilità scientifica: GPT consolidamento.**
Simone: «da ora in poi questa parte la gestisci tu», «sarai tu a fare il
deployment e a gestirlo». Vale per EPN15 e valutazione lunga, non per le altre
linee. Fonte Claude3566c8d, conferma diretta di Simone in questa chat; regola
canonica già aggiornata dal coordinamento in mainfe66e2d. GPT non sottomette,
cancella, rilancia, pubblica deploy o effettua monitoraggi operativi di queste
due campagne. Eventuali verifiche successive avvengono dopo, senza bloccare
le operazioni già approvate e gestite da Claude.

La review24860d converge sul pacchetto51b4574. Questa unica consegna cambia
la destinazione del lungo suSCRATCH e il proprietario operativo, correggeN-4
nel solo test e resta **awaiting_claude_rereview**, `review.commit:null`.
La quarta review deve esaminare il solo diff da51b4574. Dopo convergenza,
**Claude esegue i pin** sul proprio ramo figlio del commit consegnato e usa
solo quel ramo per la catena; GPT non li anticipa. Sorgenti fissi fra le fasi.

## Istruzioni e riferimenti

- Valutazione lunga: comandi esatti e rilascio delle quattro fasi, nella sezione completa qui sotto.
- EPN15: job esistenti, controlli, accounting e analisi finale, nella seconda sezione completa qui sotto.
- [Protocollo esecutivo aggiornato: storage, retention, gate e budget](consolidation_long_execution_20261105.md).
- Piani del lungo: `results/infra/consolidation_long_20261105/{plan,launch_plan}.json`.
- Verifiche della consegna: `results/infra/consolidation_long_20261105/handoff_validation.json`.

Le note contengono comandi **da eseguire da Claude**, non azioni eseguite daGPT.
Restano82jobcalcolo+4checkout, tetti9GPUh/1711,666667coreh, ricetta/endpoint
immutati. Tutti i nuovi output scientifici del lungo sono suSCRATCH grande;
metadata esportati e committati a fine fase, payload grandi fuoriGit.
I checkpoint origine, i checkout e piccoli log restanoWORK; FT15 in corso
resta sul suo checkout1359b924 e sui suoi outputWORK.

## Stato della consegna e nessuna operazione sospesa

Ledger Mac controllato in sola lettura:27statiFT15 tutti `sottomesso`, con
ID SLURM (24fit+benchmark+2checkout). Nessuna prenotazione/in_sottomissione
o invio ambiguo aperto. Tutti86IDLONG assenti, nessun checkout o deployLONG
pubblicato daGPT. Nessuna operazione da completare o prenotazione da chiudere.

Ultimo stato operativo: quello letto daClaude alle21:05CET e trasmesso daSimone,
8frozen completati,8C0FT/scratch in esecuzione,8C1 conafterok in attesa.
Nessuna differenza nota dal precedente inventario locale/remoto diGPT;
nessuna nuova interrogazione remota dopo il passaggio. Le note EPN15 separano
il consuntivo archiviato prima dei fit dal consumo corrente ancora da raccogliere.

C-1 richiede controlli freschi diSCRATCH eWORK daClaude prima degli invii;
la vecchia insufficienzaWORK non è prova né di successo né di fallimento dei
nuovi gate. C-2 e nessunretrycieco restano invariati. Nessun esperimento nuovo.

---

# Valutazione lunga — consegna operativa a Claude

I comandi sotto sono istruzioni per Claude, non operazioni eseguite da GPT.
Alla consegna nessuno degli86ID LONG ha una directory o un lock nel ledger
locale. GPT non ha creato checkout, pubblicato deploy o sottomesso job LONG.

Claude possiede deployment, sottomissioni, stato/log e raccolta; GPT conserva
codice e responsabilità scientifica. La quarta review riguarda l'emendamento
SCRATCH/N-4/passaggio operativo rispetto a51b4574 e questo prontuario, incluso
il wrapper della sintesi. La review24860d non è il pin della quarta review.

## Radici e invarianti

- Piano: results/infra/consolidation_long_20261105/plan.json.
- Matrice: results/infra/consolidation_long_20261105/launch_plan.json.
- Output nuovi: /leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/consolidation_long_20261105.
- Checkpoint sorgenti: copie persistenti WORK già pinnate in plan.endpoints.
- Checkouts: /leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/.
- Log calcolo: /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/consolidation_long_<SLURM-ID>.out.
  Log checkout: stesso parent, campaign_checkout_<SLURM-ID>.out.
- 12estrazioni +16benchmark +48sonde +3posa +2confronti +1sintesi=82job;
  4checkout, totale86; tetto9GPUh/1711,666667coreh.

Source e recipe identici fra fasi. Cambiare solo pin e stati, documenti e
risultati. Nessuna modifica a src/**/*.py, scripts/*.py, launcher del lungo o
lib_code_dir.sh dopo il primo checkout. R-2(i) resta valido: un cambio coperto
invalida le estrazioni, non autorizza una ripetizione automatica.

| Fase | ID checkout | Nome checkout / suffisso del ref deploy/ |
|---|---|---|
|extract|LONG-checkout-extract-20261105|consolidation-long-extract-20261105|
|benchmark|LONG-checkout-benchmark-20261105|consolidation-long-benchmark-20261105|
|fit|LONG-checkout-fit-20261105|consolidation-long-fit-20261105|
|aggregate|LONG-checkout-aggregate-20261105|consolidation-long-aggregate-20261105|

## 1. Review, commit di fase e checkout

Claude crea un proprio ramo/worktree figlio dello SHA della consegna indicato
da GPT; non lavora nel worktree GPT. Sul Mac, dentro quel worktree:

~~~bash
LONG_PY=/Users/simonebenatti/.venvs/wearusfm-tok/bin/python
LONG_SUBMIT='/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py'
LONG_PLAN=results/infra/consolidation_long_20261105/plan.json
LONG_LAUNCH=results/infra/consolidation_long_20261105/launch_plan.json
LONG_ROOT=/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/consolidation_long_20261105
LONG_LOGS=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs
~~~

Confrontare byte per byte LONG_SUBMIT e scripts/submit_job.py prima dell'uso:
è il submitter comune già usato per FT15, ledger ~/wearusfm_local/jobs.
Non creare un ledger parallelo. Usare --agente claude.

Dopo la quarta review, impostare LONG_REVIEW_COMMIT al suo SHA completo reale,
verificato su origin; non usare24860d al posto della review dello SCRATCH.

~~~bash
: "${LONG_REVIEW_COMMIT:?SHA completo della quarta review convergente}"
export LONG_REVIEW_COMMIT
"$LONG_PY" - <<'PY'
import json,os,re
from pathlib import Path
root=Path("results/infra/consolidation_long_20261105")
p=root/"plan.json";l=root/"launch_plan.json"
plan=json.loads(p.read_text());launch=json.loads(l.read_text())
review=os.environ["LONG_REVIEW_COMMIT"];assert re.fullmatch("[0-9a-f]{40}",review)
plan["review"].update(status="claude_review_converged",commit=review)
plan["status"]=launch["status"]="claude_review_converged"
assert launch["owner"]=="claude"
p.write_text(json.dumps(plan,indent=2)+"\n");l.write_text(json.dumps(launch,indent=2)+"\n")
PY
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts "$LONG_PY" - <<'PY'
import json
from pathlib import Path
import consolidation_long_common as C
p=json.loads(Path("results/infra/consolidation_long_20261105/plan.json").read_text())
C.validate_plan(p)
assert p["source_sha256"]==C.source_identity()
assert p["recipe_sha256"]==C.recipe_identity(p)
PY
git diff --check
git add results/infra/consolidation_long_20261105 docs/claude_GPT_shared docs/decisioni.md
git commit -m 'Release reviewed long extraction phase'
LONG_PHASE_SHA=$(git rev-parse HEAD)
git push origin HEAD
git ls-remote origin "$(git symbolic-ref HEAD)"
~~~

Controllare SHA remoto=LONG_PHASE_SHA. Ripetere il processo per le fasi
successive dopo aver compilato i loro manifest reali. Non inserire lo SHA
del commit nel piano contenuto nello stesso commit: sarebbe autoreferenziale.
Prima si fissa il commit di fase, poi si usano SHA e CODE_DIR negli invii.
Un successivo commit documentale registra gli SHA reali in
launch_plan.phase_checkouts[].code_sha; si continua a eseguire il precedente
SHA fissato, non il HEAD documentale più recente.

Per extract (ripetere solo per le altre tre righe della tabella):

~~~bash
LONG_PHASE=extract
LONG_CAMPAIGN=consolidation-long-extract-20261105
LONG_CHECKOUT_ID=LONG-checkout-extract-20261105
LONG_DEPLOY_REF=refs/heads/deploy/consolidation-long-extract-20261105
LONG_CODE=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/consolidation-long-extract-20261105
git ls-remote leonardo "$LONG_DEPLOY_REF"
~~~

Al primo uso la risposta dev'essere vuota. Se il ref esiste, riconciliare
con ledger e checkout senza sovrascrivere una campagna attiva. Dopo i
controlli del remoto/hook, spazio e assenza di output/preparazioni:

~~~bash
git push leonardo "$LONG_PHASE_SHA:$LONG_DEPLOY_REF"
git ls-remote leonardo "$LONG_DEPLOY_REF"
"$LONG_PY" "$LONG_SUBMIT" --id "$LONG_CHECKOUT_ID" --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0 --timeout 600 \
 --script-stdin scripts/slurm/campaign_checkout.sbatch -- \
 --partition=lrd_all_serial --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G \
 --time=00:10:00 --no-requeue \
 "--export=ALL,CAMPAIGN=$LONG_CAMPAIGN,CAMPAIGN_SHA=$LONG_PHASE_SHA,EXPECTED_SHA=$LONG_PHASE_SHA"
~~~

Ogni checkout:0GPUh,0,333333coreh massime prudenziali
(2CPU contabilizzate×10min, pur chiedendo1CPU). Leggere sacct COMPLETED0:0,
log “checkout di campagna pronto”, HEAD, git status e sola lettura.
LONG_LOGS deve esistere: è il parent storico, non è stata introdotta una
directory log SCRATCH. I job creano soltanto output nuovi sotto LONG_ROOT.

Le righe successive assumono checkout completato e verificato, quindi non
usano --dopo verso un ID già purgato dalla memoria SLURM (MinJobAge30s).
Se accodate prima del completamento, aggiungere --dopo "$LONG_CHECKOUT_ID"
quando ancora valido; la dipendenza non sostituisce i controlli delle ricevute.

## 2. Comandi esatti per tutti i tipi

Esempi di righe reali della matrice; non eseguire l'intero blocco in un batch
incontrollato. CPU a ondate, massimo8job outstanding dell'intero utente e
limite seriale10 submission. La partizione osservata concede8CPU complessive
all'utente: i job da8CPU non promettono parallelismo fra loro.

### Extract: classificazione P1/P2 e posa train/val

~~~bash
"$LONG_PY" "$LONG_SUBMIT" --id LONG-G19618-p1-20261105 --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0.75 --timeout 600 -- \
 --partition=boost_usr_prod --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=100G \
 --gres=gpu:1 --time=00:45:00 --no-requeue \
 "--export=ALL,CODE_DIR=$LONG_CODE,EXPECTED_SHA=$LONG_PHASE_SHA" \
 "$LONG_CODE/scripts/slurm/consolidation_long.sbatch" --plan "$LONG_CODE/$LONG_PLAN" \
 --expected-code "$LONG_PHASE_SHA" --stage extract-classification \
 --out "$LONG_ROOT/19618/p1" --step 19618 --task p1

"$LONG_PY" "$LONG_SUBMIT" --id LONG-G19618-p2-20261105 --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0.75 --timeout 600 -- \
 --partition=boost_usr_prod --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=100G \
 --gres=gpu:1 --time=00:45:00 --no-requeue \
 "--export=ALL,CODE_DIR=$LONG_CODE,EXPECTED_SHA=$LONG_PHASE_SHA" \
 "$LONG_CODE/scripts/slurm/consolidation_long.sbatch" --plan "$LONG_CODE/$LONG_PLAN" \
 --expected-code "$LONG_PHASE_SHA" --stage extract-classification \
 --out "$LONG_ROOT/19618/p2" --step 19618 --task p2

"$LONG_PY" "$LONG_SUBMIT" --id LONG-G19618-pose-train-20261105 --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0.75 --timeout 600 -- \
 --partition=boost_usr_prod --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=100G \
 --gres=gpu:1 --time=00:45:00 --no-requeue \
 "--export=ALL,CODE_DIR=$LONG_CODE,EXPECTED_SHA=$LONG_PHASE_SHA" \
 "$LONG_CODE/scripts/slurm/consolidation_long.sbatch" --plan "$LONG_CODE/$LONG_PLAN" \
 --expected-code "$LONG_PHASE_SHA" --stage extract-pose-train \
 --out "$LONG_ROOT/19618/pose/train" --step 19618

"$LONG_PY" "$LONG_SUBMIT" --id LONG-G19618-pose-val-20261105 --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0.75 --timeout 600 -- \
 --partition=boost_usr_prod --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=100G \
 --gres=gpu:1 --time=00:45:00 --no-requeue \
 "--export=ALL,CODE_DIR=$LONG_CODE,EXPECTED_SHA=$LONG_PHASE_SHA" \
 "$LONG_CODE/scripts/slurm/consolidation_long.sbatch" --plan "$LONG_CODE/$LONG_PLAN" \
 --expected-code "$LONG_PHASE_SHA" --stage extract-pose-val \
 --out "$LONG_ROOT/19618/pose/val" --step 19618
~~~

Le stesse quattro righe alle dosi39237/49046 hanno gli ID e gli args già
registrati nel piano:12totali. Il gruppo19k permette il controllo di parità;
non scegliere cosa proseguire in base a nuovi score.

### Benchmark:15 classificazione e1 posa

Ora LONG_CODE e LONG_PHASE_SHA sono del checkout benchmark, con cache_manifest
completo, committato e pinnato.

~~~bash
"$LONG_PY" "$LONG_SUBMIT" --id LONG-BENCH-ninapro_db2-pre-20261105 --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0 --timeout 600 -- \
 --partition=lrd_all_serial --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
 --time=00:30:00 --no-requeue \
 "--export=ALL,CODE_DIR=$LONG_CODE,EXPECTED_SHA=$LONG_PHASE_SHA" \
 "$LONG_CODE/scripts/slurm/consolidation_long.sbatch" --plan "$LONG_CODE/$LONG_PLAN" \
 --expected-code "$LONG_PHASE_SHA" --stage benchmark-classification \
 --out "$LONG_ROOT/benchmarks/classification/ninapro_db2/pre" \
 --step 19618 --dataset ninapro_db2 --view pre

"$LONG_PY" "$LONG_SUBMIT" --id LONG-BENCH-pose-20261105 --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0 --timeout 600 -- \
 --partition=lrd_all_serial --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
 --time=00:15:00 --no-requeue \
 "--export=ALL,CODE_DIR=$LONG_CODE,EXPECTED_SHA=$LONG_PHASE_SHA" \
 "$LONG_CODE/scripts/slurm/consolidation_long.sbatch" --plan "$LONG_CODE/$LONG_PLAN" \
 --expected-code "$LONG_PHASE_SHA" --stage benchmark-pose \
 --out "$LONG_ROOT/benchmarks/pose"
~~~

Classificazione5dataset×3viste; tutti15prima del gate globale48shard.
Posa solo train19k. Nessuna selezione di risorse su held/validation.

### Fit:48 classificazione e3 posa

Release/checkout fit con ricevute benchmark, preventivo e ammissione globale.

~~~bash
"$LONG_PY" "$LONG_SUBMIT" --id LONG-FIT-G19618-pre-ninapro_db2-F01-20261105 --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0 --timeout 600 -- \
 --partition=lrd_all_serial --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
 --time=04:00:00 --no-requeue \
 "--export=ALL,CODE_DIR=$LONG_CODE,EXPECTED_SHA=$LONG_PHASE_SHA" \
 "$LONG_CODE/scripts/slurm/consolidation_long.sbatch" --plan "$LONG_CODE/$LONG_PLAN" \
 --expected-code "$LONG_PHASE_SHA" --stage fit-classification \
 --out "$LONG_ROOT/fits/classification/19618/pre/ninapro_db2/F01" \
 --step 19618 --dataset ninapro_db2 --view pre --folds 0 1

"$LONG_PY" "$LONG_SUBMIT" --id LONG-FIT-G19618-pre-epn612-F0123-20261105 --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0 --timeout 600 -- \
 --partition=lrd_all_serial --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
 --time=04:00:00 --no-requeue \
 "--export=ALL,CODE_DIR=$LONG_CODE,EXPECTED_SHA=$LONG_PHASE_SHA" \
 "$LONG_CODE/scripts/slurm/consolidation_long.sbatch" --plan "$LONG_CODE/$LONG_PLAN" \
 --expected-code "$LONG_PHASE_SHA" --stage fit-classification \
 --out "$LONG_ROOT/fits/classification/19618/pre/epn612/F0123" \
 --step 19618 --dataset epn612 --view pre --folds 0 1 2 3

"$LONG_PY" "$LONG_SUBMIT" --id LONG-FIT-G19618-pose-20261105 --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0 --timeout 600 -- \
 --partition=lrd_all_serial --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
 --time=01:00:00 --no-requeue \
 "--export=ALL,CODE_DIR=$LONG_CODE,EXPECTED_SHA=$LONG_PHASE_SHA" \
 "$LONG_CODE/scripts/slurm/consolidation_long.sbatch" --plan "$LONG_CODE/$LONG_PLAN" \
 --expected-code "$LONG_PHASE_SHA" --stage fit-pose \
 --out "$LONG_ROOT/fits/pose/19618" --step 19618
~~~

DB2 usa sempre coppie0,1 e2,3; gli altri dataset tutti4fold. Nessun nuovo
fit19k/pool. Le tre dosi posa sono quelle del piano. Gate negativo=stop,
non shard o tempi cambiati dopo la misura.

### Aggregate:due confronti

Release/checkout aggregate con48report classificazione e3posa pinnati.

~~~bash
"$LONG_PY" "$LONG_SUBMIT" --id LONG-COMPARE-classification-20261105 --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0 --timeout 600 -- \
 --partition=lrd_all_serial --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
 --time=01:00:00 --no-requeue \
 "--export=ALL,CODE_DIR=$LONG_CODE,EXPECTED_SHA=$LONG_PHASE_SHA" \
 "$LONG_CODE/scripts/slurm/consolidation_long.sbatch" --plan "$LONG_CODE/$LONG_PLAN" \
 --expected-code "$LONG_PHASE_SHA" --stage compare-classification \
 --out "$LONG_ROOT/comparison/classification"

"$LONG_PY" "$LONG_SUBMIT" --id LONG-COMPARE-pose-20261105 --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0 --timeout 600 -- \
 --partition=lrd_all_serial --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
 --time=01:00:00 --no-requeue \
 "--export=ALL,CODE_DIR=$LONG_CODE,EXPECTED_SHA=$LONG_PHASE_SHA" \
 "$LONG_CODE/scripts/slurm/consolidation_long.sbatch" --plan "$LONG_CODE/$LONG_PLAN" \
 --expected-code "$LONG_PHASE_SHA" --stage compare-pose \
 --out "$LONG_ROOT/comparison/pose"
~~~

La sintesi richiede entrambi i report reali. Non creare un quinto checkout:
il piano finale esterno è materializzato dall'unico LONG-SUMMARY già previsto,
come descritto nella sezione5.

## 3. Ricevute, manifest e preventivo misurato

A fine fase copiare sul Mac report.json di ogni job, log, ledger e sacct.
Mapping obbligatorio: execution/receipts/<ID>.json ← <out>/report.json.
Confrontare sha256sum remoto e hash locale prima di usare la copia.

~~~bash
mkdir -p results/infra/consolidation_long_20261105/execution/receipts
scp leonardo:/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/consolidation_long_20261105/19618/p1/report.json \
 results/infra/consolidation_long_20261105/execution/receipts/LONG-G19618-p1-20261105.json
~~~

Questo snippet locale rigenera i manifest da tutte le ricevute raccolte,
senza alcun submit. LONG_PIN_PHASE è benchmark dopo le estrazioni, fit dopo
i benchmark, aggregate dopo i fit, summary dopo i confronti. Non riusare il
nome di un manifest già consumato: open("x") rifiuta la sovrascrittura.

~~~bash
export LONG_PIN_PHASE=benchmark
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts "$LONG_PY" - <<'PY'
import hashlib,json,os
from pathlib import Path
import consolidation_long_common as C
import consolidation_long_classification_runtime as CR
import consolidation_long_pose_runtime as PR
root=Path("results/infra/consolidation_long_20261105")
p=root/"plan.json";l=root/"launch_plan.json"
plan=json.loads(p.read_text());launch=json.loads(l.read_text());rows=launch["jobs"]
copies=root/"execution/receipts";dest=root/"execution/manifests"
dest.mkdir(parents=True,exist_ok=True)
def pin(path,remote=None):
    f=Path(path)
    return {"path":remote or str(f),"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),
            "bytes":f.stat().st_size}
def collected(row):
    f=copies/(row["id"]+".json");r=json.loads(f.read_text())
    assert r["complete"] is True and r["protocol"]==C.PROTOCOL and r["kind"]==row["stage"]
    assert r["source_sha256"]==plan["source_sha256"] and r["recipe_sha256"]==plan["recipe_sha256"]
    # Owner has separately verified sacct0:0, phasecode/plan and remotehash.
    return r,pin(f,row["out"]+"/report.json")
def write(name,value):
    f=dest/name
    with f.open("x") as stream:json.dump(value,stream,indent=2);stream.write("\n")
    return pin(f)
phase=os.environ["LONG_PIN_PHASE"]
if phase=="benchmark":
    m={"source_sha256":plan["source_sha256"],"recipe_sha256":plan["recipe_sha256"],
       "classification":{str(s):{} for s in C.DOSES},"pose":{str(s):{} for s in C.DOSES}}
    for row in rows:
        if row["phase"]!="extract":continue
        r,rp=collected(row);step=str(r["checkpoint_step"])
        assert r["which"]=="teacher" and r["checkpoint_unchanged"] is True
        assert r["frozen_teacher_unchanged"] is True
        assert r["checkpoint_sha256"]==plan["endpoints"][step]["checkpoint"]["sha256"]
        if r["kind"]=="extract-classification":
            for ds,cache in r["artifacts"].items():m["classification"][step][ds]={"cache":cache,"receipt":rp}
        else:m["pose"][step][r["split"]]={"cache":r["artifacts"]["cache"],"receipt":rp}
    assert all(len(v)==5 for v in m["classification"].values())
    assert all(set(v)=={"train","val"} for v in m["pose"].values())
    plan["cache_manifest"]=write("cache_manifest.json",m)
elif phase=="fit":
    reports={};pins={}
    for row in rows:
        if row["stage"]=="benchmark-classification":
            r,rp=collected(row);ds=r["dataset"];v=r["view"]
            reports.setdefault(ds,{})[v]=r;pins.setdefault(ds,{})[v]=rp
    gate=CR.resource_gate(reports)
    assert gate["all_passed"] is True and gate["physical_fit_jobs"]==48
    pose,rp=collected(next(r for r in rows if r["stage"]=="benchmark-pose"))
    projection=PR.project_benchmark(pose["timings_seconds"],pose["peak_rss_bytes"])
    assert projection==pose["projection"] and projection["admitted"] is True
    assert pose["held_payload_opened"] is False and pose["held_scored"] is False
    write("measured_resource_gates.json",{"classification":gate,"pose":projection})
    plan["benchmark_receipts"]={"classification":write("classification_benchmark_receipts.json",{"reports":pins}),
                               "pose":rp}
elif phase=="aggregate":
    reports=[];pose={}
    for row in rows:
        if row["stage"]=="fit-classification":
            r,rp=collected(row);assert r["global_resource_gate_passed"] is True;reports.append(rp)
        elif row["stage"]=="fit-pose":
            r,rp=collected(row);assert r["held_payload_opened_after_both_ridge_fits"] is True
            pose[str(r["checkpoint_step"])]={"report":rp}
    assert len(reports)==48 and set(pose)==set(map(str,C.DOSES))
    plan["classification_fit_manifest"]=write("classification_fit_manifest.json",{"reports":reports})
    plan["pose_fit_manifest"]=write("pose_fit_manifest.json",{"fits":pose})
elif phase=="summary":
    for kind in ("classification","pose"):
        _,rp=collected(next(r for r in rows if r["stage"]=="compare-"+kind))
        plan["final_reports"][kind]=rp
else:raise ValueError("unknown pin phase")
assert plan["source_sha256"]==C.source_identity()==launch["source_sha256"]
assert plan["recipe_sha256"]==C.recipe_identity(plan)==launch["recipe_sha256"]
C.validate_plan(plan)
if phase=="summary":
    with (root/"execution/summary_plan.json").open("x") as stream:
        json.dump(plan,stream,indent=2);stream.write("\n")
else:p.write_text(json.dumps(plan,indent=2)+"\n")
l.write_text(json.dumps(launch,indent=2)+"\n")
PY
~~~

Ogni manifest è un file del repo, con pin relativo al checkout; i pin delle
ricevute/cache sono invece i percorsi assoluti SCRATCH originali. Il runtime
ricontrolla tutti i pin quando li usa. Non convertire i vecchi piani storici
in path nuovi: il loro plan_content_sha256 è parte della provenienza.

Controlli dell'owner prima di ciascuna fase, oltre allo snippet:

1. Tutti gli ID del gruppo COMPLETED0:0, un solo tentativo per ID. Sacct
   JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End archiviato.
   GPUh=ElapsedRaw×GPUallocate/3600; coreh=ElapsedRaw×AllocCPUS/3600, solo
   righe dei job, senza doppio conteggio .batch/.extern.
2. Hash e byte di ricevute e artefatti sul cluster uguali ai pin; code_sha e
   plan_sha256 della fase eseguita, non del successivo commit documentale.
   Nessun STOP, symlink, sovrascrittura o checkpoint modificato.
3. Estrazioni: teacher/checkpoint immutati, dosi corrette; posa1907train/
   18548val,40/15utenti,200/1950registrazioni, bersagli/maschere/supporti
   verificati dal runtime. Per classificazione controllare anche i campi
   scalar nelle NPZ19k: historical_pool_parity=true, reference_cache_sha256,
   historical_pool_tolerance, historical_pool_exact_equal e
   historical_pool_max_abs_error. La tolleranza non prova OOF identiche.
4. Benchmark:15celle con6C convergenti sul solo train, hash nuovo unico e
   regolaB verso storico;48shard ammessi globalmente. Posa solo train, policy
   e ambiente coerenti, proiezione e RSS sotto soglia. Pubblicare il preventivo
   misurato e consuntivi prima dei fit.
5. Fit:48shard/copertura40celle e tutti4fold, nessun19k/pool nuovo; audit/OOF
   integri.3posa con tutti3lettori, CV train prima del held e alpha non
   occultato al bordo. Coefficienti/predizioni pinnati.
6. Confronti: tutti i pannelli previsti, due primari posa affiancati, IC
   appaiati, replica storica descrittiva. Non scegliere il pannello favorevole
   né pubblicare un verdetto su una matrice incompleta.

## 4. Registrare SHA di fase e ID effettivi

Dopo la release e gli invii, un commit documentale successivo registra gli
SHA/ID senza cambiare il runtime già congelato:

~~~bash
export LONG_PHASE LONG_PHASE_SHA
"$LONG_PY" - <<'PY'
import json,os
from pathlib import Path
p=Path("results/infra/consolidation_long_20261105/launch_plan.json")
v=json.loads(p.read_text());phase=os.environ["LONG_PHASE"];sha=os.environ["LONG_PHASE_SHA"]
entry=next(r for r in v["phase_checkouts"] if r["name"]=="consolidation-long-"+phase+"-20261105")
assert entry["code_sha"] in (None,sha);entry["code_sha"]=sha
ledger=Path.home()/"wearusfm_local/jobs"
for row in v["jobs"]:
    if row["phase"]!=phase:continue
    f=ledger/row["id"]/"state.json"
    if f.exists():
        state=json.loads(f.read_text())
        assert state["stato"]=="sottomesso" and state["codice_sha"]==sha
        row["slurm_job_id"]=state["job_slurm"]
p.write_text(json.dumps(v,indent=2)+"\n")
PY
~~~

Archiviare anche state.json del checkout e codice/planhash di ciascun job.
Per summary: code_sha del quarto checkout, plan_sha256 del piano esterno
finale; entrambi devono apparire nella consegna, senza confonderli con lo
SHA che archivia il wrapper.

## 5. Sintesi finale senza quinto checkout

I pin dei confronti non esistono prima del quarto checkout. Dopo la raccolta,
phase=summary crea il piano finale sul Mac; lo si committa/pusha insieme a
un wrapper operativo. L'unico LONG-SUMMARY previsto materializza quei byte
sotto SCRATCH, fuori dal codice, ne verifica hash/size/percorso fresco, poi
invoca il launcher del quarto checkout. Nessun sorgente o checkout cambia.
Questa procedura fa parte della quarta review, non è un job aggiuntivo.

Con LONG_CODE e LONG_PHASE_SHA ancora fissati ad aggregate:

~~~bash
export LONG_CODE LONG_PHASE_SHA LONG_ROOT
"$LONG_PY" - <<'PY'
import base64,hashlib,json,os,shlex
from pathlib import Path
root=Path("results/infra/consolidation_long_20261105")
p=root/"execution/summary_plan.json";raw=p.read_bytes();plan=json.loads(raw)
sha=hashlib.sha256(raw).hexdigest();assert all(plan["final_reports"].values())
target=os.environ["LONG_ROOT"]+"/manifests/summary_plan_"+sha+".json"
code=os.environ["LONG_CODE"];expected=os.environ["LONG_PHASE_SHA"]
args=["bash",code+"/scripts/slurm/consolidation_long.sbatch","--plan",target,
      "--expected-code",expected,"--stage","summary","--out",os.environ["LONG_ROOT"]+"/summary"]
program=r'''
import base64,hashlib,os
from pathlib import Path
raw=base64.b64decode(os.environ["LONG_PLAN_B64"],validate=True)
target=Path(os.environ["LONG_SUMMARY_PLAN"])
assert hashlib.sha256(raw).hexdigest()==os.environ["LONG_PLAN_SHA256"]
assert len(raw)==int(os.environ["LONG_PLAN_BYTES"])
root=Path(os.environ["LONG_APPROVED_OUTPUT_ROOT"])
assert target.is_absolute() and root in target.parents
assert not target.exists() and not target.is_symlink()
assert not any(p.is_symlink() for p in target.parents)
target.parent.mkdir(parents=True,exist_ok=True)
with target.open("xb") as stream:stream.write(raw)
target.chmod(0o444)
assert hashlib.sha256(target.read_bytes()).hexdigest()==os.environ["LONG_PLAN_SHA256"]
'''
env={"LONG_PLAN_B64":base64.b64encode(raw).decode(),"LONG_SUMMARY_PLAN":target,
     "LONG_PLAN_SHA256":sha,"LONG_PLAN_BYTES":str(len(raw)),
     "LONG_APPROVED_OUTPUT_ROOT":os.environ["LONG_ROOT"]}
lines=["#!/bin/bash","set -euo pipefail",
       'test "${EXPECTED_SHA:?}" = '+shlex.quote(expected),
       'test "${CODE_DIR:?}" = '+shlex.quote(code)]
lines+=["export "+k+"="+shlex.quote(v) for k,v in env.items()]
lines+=["python3 - <<'PY_PLAN'",program,"PY_PLAN","exec "+shlex.join(args)]
dest=root/"execution/summary_materialize.sbatch"
with dest.open("x") as stream:stream.write("\n".join(lines)+"\n")
print(json.dumps({"plan_sha256":sha,"plan_bytes":len(raw),"remote_plan":target,
                  "runtime_code_sha":expected,"wrapper":str(dest)},indent=2))
PY
bash -n results/infra/consolidation_long_20261105/execution/summary_materialize.sbatch
git add results/infra/consolidation_long_20261105/execution docs/claude_GPT_shared docs/decisioni.md
git commit -m 'Pin final long reports and immutable external summary plan'
git push origin HEAD
git ls-remote origin "$(git symbolic-ref HEAD)"
~~~

Dopo verificaSHA remoto del commit appena pubblicato:

~~~bash
"$LONG_PY" "$LONG_SUBMIT" --id LONG-SUMMARY-20261105 --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0 --timeout 600 \
 --script-stdin results/infra/consolidation_long_20261105/execution/summary_materialize.sbatch -- \
 --account=IscrB_WearUsFM --partition=lrd_all_serial --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=2G \
 --time=00:10:00 --no-requeue \
 "--output=$LONG_LOGS/consolidation_long_%j.out" \
 "--error=$LONG_LOGS/consolidation_long_%j.out" \
 "--export=ALL,CODE_DIR=$LONG_CODE,EXPECTED_SHA=$LONG_PHASE_SHA"
~~~

Il ledger conserva script_stdin_sha256 del wrapper. La materializzazione è
sul compute node, non sul login. Il piano esterno è immutabile e il runtime
ne controlla l'hash anche a fine esecuzione. Se il job fallisce dopo averlo
scritto, non sovrascriverlo né ritentare automaticamente l'ID.

## 6. Retention, anomalie e chiusura

Prima di ogni invio Claude verifica spazio/scrivibilità dello SCRATCH grande
e quota WORK per checkout/log e scritture già ammesse di altre campagne.
FT15 e checkpoint sorgenti restano nelle loro posizioni. Cache/OOF/array nuovi
restano su SCRATCH per meno40giorni secondo Simone; sono rigenerabili da
checkpoint/input/ricetta ma la rigenerazione non è un lancio autorizzato.

A fine di ogni fase copiare subito ricevute, report e manifest sul Mac,
registrare hash/consuntivi, committare e pushare con SHA verificato. Dati,
cache/OOF/NPZ non vanno in Git. Nessun touch automatico o promessa di proroga
della purge; se serviranno oltre40giorni si definisce prima una destinazione
esterna durevole.

Ambiguità del ledger: --riconcilia, mai un secondo ID per lo stesso tentativo.
Un esplicito rifiuto SLURM segue il meccanismo --dopo-rifiuto dopo diagnosi;
non è un retry automatico. TIMEOUT/OOM/nonfinite/errore/gate negativo:
conservare log e parziali, fermare la fase, nessun requeue, overwrite, fold
scartato o criterio cambiato. Non cancellare risultati di altre campagne.

Chiusura: report e contrasti previsti, consuntivo delle86voci contando ogni
checkout una volta, lettura del critico, tabella/decisioni e registro in
entrambe le sezioni, commit/push verificato. GPT verifica dopo, senza
operare sui job Claude. Nessun monitor GPT.


---

# Passaggio operativo EPN15 a Claude — 05/11/2026 sera

Da questo passaggio Claude gestisce deployment, sottomissioni, stato, log e
raccolta EPN15; GPT mantiene codice e responsabilità scientifica. Simone:
«da ora in poi questa parte la gestisci tu» e «sarai tu a fare il deployment
e a gestirlo». Questa nota contiene istruzioni per Claude, non comandi eseguiti
da GPT dopo il passaggio. I 24 fit sono già sottomessi: non rilanciarli.

## Stato e operazioni aperte

Fonte più recente: lettura di Claude delle **21:05 CET**, trasmessa da Simone:
8 frozen completati, 8 C0 FT/scratch RUNNING da circa8ore su16, 8 C1 PENDING
con afterok. GPT non ha effettuato un nuovo controllo remoto dopo il passaggio.
L'inventario stat di GPT delle20:22 mostrava8checkpoint frozen già scritti e
gli altri16assenti: compatibile con questa lettura, non prova di completamento
scientifico dei frozen. Nessuna differenza nota da dichiarare.

Il ledger locale è stato letto durante questa consegna: **27 state.json FT15,
tutti `sottomesso`, tutti con ID SLURM** (24fit, benchmark,2checkout). Nessuno
`prenotato`, `in_sottomissione`, ambiguo o da riconciliare. Nessuna riga LONG.
Nessuna prenotazione o operazione di invio rimasta a metà da parte di GPT.
`sottomesso` nel ledger è stato dell'invio, non stato live del job.

## Riferimenti immutabili

| Oggetto | Riferimento |
|---|---|
|Piano scientifico eseguito|`results/infra/finetune15_20261105/plan.json` al commit1359b924|
|Piano dei24invii|`results/infra/finetune15_20261105/launch_plan.json` corrente; tutte le righe hanno ID reale|
|Protocollo|`docs/finetune15_protocol_20261105.md`|
|Codice fit|`1359b924d25601f92e9035b8bd621c2c6a93f947`|
|CODE_DIR fit|`/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/finetune15-fits-20261105`|
|Piano SHA256|`8a07e975f1df709729222a3a3c79f5f28713b5613584b87b3428213d896054d2`|
|Sorgenti FT15 SHA256|`13de2f317997e497f6d40a53d04dbc6b2061a6d2244c0a809898968362091469`|
|Ricetta SHA256|`db975bd1c46f6fbf3259c0ebd73edbdd68b541bf1cc452e4ad6084f6e4a7b4d1`|
|Review fit|`ceaf95b66519b93e4f2bcfdda2d3cafd2f9f5e24`|
|Teacher19k SHA256|`a31c895061c1f273eaac2d796fc00cf76cffa4318b521a20be4d6ba65b63a056`|
|Input EPN SHA256|`ef0090cb748e9eb4b9e9f474f084b8dbbbb6a9ca5f6594d91ceaf216204ee0c0`|
|Benchmark report SHA256|`772bc48777d7e898b6a7db38b297ec502db339ee47287ea3690798912e308b55`|
|Benchmark checkout|`/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/finetune15-bench-20261105` @ `02f3de8547e0e50ee8d66f6ea43a356518c18e70`|

Il piano fit corrente è byte-identico a quello di1359b924. Per l'analisi
estrarre comunque il blob da quel commit: non cambiare il piano eseguito per
aggiornare l'owner. Il cambio operativo non modifica checkout, record storici
del submitter, identità del run o output EPN15. La nuova autorizzazione SCRATCH
riguarda la valutazione lunga; **EPN15 resta suWORK** perché è già in corso.

## Mappa dei job

Per ogni riga il logical ID è `FT15-EPN-REGIME-F<FOLD>-20261105-C<N>`.
La dipendenza di C1 è `afterok:<ID C0 della stessa riga>`.

| Regime | Fold | C0 | C1 | Risorse per candidato |
|---|---:|---:|---:|---|
|frozen|0|63581664|63581670|1GPU,8CPU,100GiB,6h|
|frozen|1|63581672|63581673|1GPU,8CPU,100GiB,6h|
|frozen|2|63581674|63581677|1GPU,8CPU,100GiB,6h|
|frozen|3|63581682|63581713|1GPU,8CPU,100GiB,6h|
|finetune|0|63581718|63581725|1GPU,8CPU,100GiB,16h|
|finetune|1|63581735|63581741|1GPU,8CPU,100GiB,16h|
|finetune|2|63581742|63581745|1GPU,8CPU,100GiB,16h|
|finetune|3|63581750|63581751|1GPU,8CPU,100GiB,16h|
|scratch|0|63581768|63581837|1GPU,8CPU,100GiB,16h|
|scratch|1|63581887|63582180|1GPU,8CPU,100GiB,16h|
|scratch|2|63582244|63582253|1GPU,8CPU,100GiB,16h|
|scratch|3|63582258|63582269|1GPU,8CPU,100GiB,16h|

Altri tre ID da includere nel consuntivo: checkout benchmark63578395,
benchmark63579657, checkout fit63581502. Ricevute degli invii in
`results/infra/finetune15_20261105/submission/`; originali canonici sotto
`/Users/simonebenatti/wearusfm_local/jobs/<logical-id>/state.json`.

Percorso risultati per candidato:
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/finetune15_20261105/epn/fold<FOLD>/<REGIME>/candidate<N>/`.
Log SLURM:
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/consolidation_ft15_<ID>.out`.
Nei candidati: `metrics.jsonl`, `candidate_<N>.json`, eventuale
`candidate_<N>.pt`, `report.json`; solo C1 può scrivere `held.npz`.

Il launcher è `scripts/slurm/consolidation_finetune15.sbatch`. La riga già
eseguita per C0 fold0 FT, conservata nel ledger, termina con:

```text
--export=ALL,CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/finetune15-fits-20261105,EXPECTED_SHA=1359b924d25601f92e9035b8bd621c2c6a93f947
--qos=normal --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=100G --time=16:00:00
/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/finetune15-fits-20261105/scripts/slurm/consolidation_finetune15.sbatch
fit /leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/finetune15_20261105/epn/fold0/finetune/candidate0 0 finetune 0
```

È una traccia di provenienza, non un invito a sottomettere ancora C0.

## C0/C1, selezione e criteri congelati

C0 esegue il primo LR dalla inizializzazione originale; registra il candidato
e **non legge il held**. C1 esegue il secondo LR dalla stessa inizializzazione
originale, legge la ricevuta/checkpoint C0, sceglie LR ed epoca sulla sola
validation e predice il held una volta per la cella fold/regime. C1 non è una
ripresa di C0 e non prosegue le sue epoche.

- Quattro fold identici a5epoche,314soggetti di sviluppo/47.100finestre;
  seed0, batch8,15epoche complete, nuovo coseno su15 e warm-up invariato.
- Frozen: testa LR{0,001;0,01}; FT: encoder{1e−5;1e−4}, testa×100;
  scratch: encoder{1e−4;1e−3}, testa×10. Stessa griglia della testa nei tre regimi.
- Primari FT−scratch e FT−frozen a15; secondari ciascun regime15−5 e
  `(FT−scratch)15−(FT−scratch)5`, bootstrap appaiato per soggetto2000repliche,
  seed20261102, stessi campioni fra le sei celle.
- Validation-tail: almeno3fold con migliore epoca15, oppure mediana del
  miglioramento15−14≥0,005 di balanced accuracy; descrive la coda del coseno,
  non prova convergenza se falso e non autorizza una terza dose.
- Secondo sguardo sui held di sviluppo dichiarato; nessun test ufficiale,
  nessun verdetto generale FM/P1/P2/regressione o applicazione di V-R1.
- Frozen vs logistica storica82,577% resta descrittivo; nessun «recuperato»
  dedotto da una soglia inventata. Griglia a due punti e seme unico restano limiti.

## Fallimenti e C1 dipendente

TIMEOUT/NODE_FAIL/OOM/errore software: conservare output parziali, log,
ricevuta e accounting; diagnosticare. `--no-requeue`, restart_count0;
nessuna ripresa da `.pt` candidato (pesi migliori, non optimizer/RNG completi).
Se C0 fallisce, **C1 resta non eseguibile**: non rimuovere afterok, non cambiare
dipendenza, non inviarlo con lo stesso output o come candidato alternativo.
Se SLURM lo annulla o lo lascia DependencyNeverSatisfied, registrare tale
stato; eventuale cancellazione amministrativa di una dipendenza impossibile
è un'azione del nuovo owner, non una sostituzione/ripetizione scientifica.
Non eseguire cancellazioni o retry automaticamente sulla base di questa nota.

La divergenza numerica è un esito registrato dal runtime, distinto dal guasto
operativo. Se tutti i candidati divergono o il selezionato è non finito sul
held, nessun fallback scelto sul held. L'aggregatore produce l'audit ma non
pubblica alcun contrasto finale con una cella scientificamente fallita.
Una ripetizione richiede diagnosi e proposta concreta revisionata, nuovi ID e
output legati al tentativo fallito, stessa inizializzazione originale e dose
completa; è fuori dal tetto approvato e non è autorizzata automaticamente.

## Budget e consuntivo

Simone,05/11 alle11:59CET, alla domanda «Confermi il tetto di304,5GPU-ore per
il confronto EPN a15epoche?»: **«Confermo il tetto»**. Fonte addendum Claude
ceaf95b trasmesso da Simone; registrata in plan.budget_authorization.
Tetto **304,5GPUh/2.436,666667coreh**:24fit(304/2432), benchmark(0,5/4),
due checkout(0/0,666667), senza tentativi aggiuntivi.
Preventivo misurato dei fit265,064444GPUh/2.120,515556coreh; atteso indicativo
dal consuntivo5epoche204,815GPUh. Questi sono preventivi, non consumi.

Consuntivo già verificato e archiviato: benchmark453s,1GPU8CPU,
0,125833GPUh/1,006667coreh; checkout31s e26s,2CPU ciascuno,
0,031667coreh. Totale **0,125833GPUh/1,038333coreh** prima dei fit.
Gli8frozen conclusi alle21:05 non hanno ancora un consuntivo SLURM finale
raccolto da GPT in questo ramo: questo totale **non è il consumo corrente
della campagna**. La prima raccolta di Claude deve aggiungere8frozen e tempi
accumulati degli8C0 attivi; poi chiudere tutti27ID a fine matrice. Non stimare
la spesa finale dai tempi del log o dai walltime prenotati.

Comando per Claude, con una sola riga per allocazione (niente doppio conteggio
di `.batch`/`.extern`):

```bash
ssh leonardo 'sacct -X -j 63578395,63579657,63581502,63581664,63581670,63581672,63581673,63581674,63581677,63581682,63581713,63581718,63581725,63581735,63581741,63581742,63581745,63581750,63581751,63581768,63581837,63581887,63582180,63582244,63582253,63582258,63582269 --starttime=2026-11-05 -n -P -o JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End,Restarts'
```

GPUh=ElapsedRaw×numeroGPUallocate/3600, coreh=ElapsedRaw×AllocCPUS/3600;
usare `AllocTRES`, non billing e non4GPUfisiche del nodo per i fit a1GPU.
Separare completati, consumato finora dei RUNNING e massimi residui; gli
stati PENDING non sono spesa allocata. Registrare anche tentativi falliti,
se emergono, senza sostituire il conteggio con i soli successi.

## Raccolta finale e aggregazione: procedura concreta per Claude

Prima di leggere i contrasti: verificare24COMPLETED0:0, restart0,24report
complete con source/recipe/code/plan/input/teacher corretti e dose15. Un
COMPLETEDSLURM non basta se il report dichiara un esito scientifico fallito.
Controllare che C0 non abbia held e che ciascun C1 usi il proprio C0; hash dei
checkpoint selezionati e dei24candidati finiti riletti sul cluster. Niente
nuova inferenza e nessun caricamento di `.pt` per ricreare le predizioni.

I comandi seguenti sono da eseguire sul Mac nel ramo di esecuzione di Claude,
figlio della consegna; i due aggregatori sono già presenti e invariati dal
checkout1359b924. Non cambiare sorgenti della campagna lunga in corso per fare
la raccolta FT15. I file grandi restano fuoriGit. Destinazioni nuove:

```bash
FT15_COLLECTION=/private/tmp/wearusfm-ft15-claude-final-20261105
mkdir "$FT15_COLLECTION"
mkdir "$FT15_COLLECTION/epn15" "$FT15_COLLECTION/epn5"
rsync -a --exclude='*.pt' leonardo:/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/finetune15_20261105/epn/ "$FT15_COLLECTION/epn15/"
rsync -a --exclude='*.pt' leonardo:/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_20261104/finetuning/epn/ "$FT15_COLLECTION/epn5/"
git show 1359b924d25601f92e9035b8bd621c2c6a93f947:results/infra/finetune15_20261105/plan.json > "$FT15_COLLECTION/executed_plan.json"
```

Generare il manifest15 sul Mac usando hash remoti reali, senza scrivere su
Leonardo. Questo blocco non modifica i checkpoint, il ledger o gli output:

```bash
python3 - <<'PY'
import hashlib, json, subprocess
from pathlib import Path
local = Path('/private/tmp/wearusfm-ft15-claude-final-20261105')
assert local.is_dir() and not (local / 'manifest.json').exists()
remote = r'''
import datetime, hashlib, json
from pathlib import Path
root = Path('/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/finetune15_20261105/epn')
assert root.is_dir() and not root.is_symlink()
out = {'files': [], 'checkpoints': [], 'observed_at_utc': datetime.datetime.utcnow().isoformat()+'Z'}
for p in sorted(root.rglob('*')):
    assert not p.is_symlink(), str(p)
    if not p.is_file():
        continue
    before = p.stat()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    after = p.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    entry = {'relative_path': str(p.relative_to(root)), 'remote_path': str(p),
             'sha256': h.hexdigest(), 'bytes': after.st_size}
    out['checkpoints' if p.suffix == '.pt' else 'files'].append(entry)
print(json.dumps(out, sort_keys=True))
'''
captured = subprocess.check_output(['ssh', 'leonardo', 'python3', '-'], input=remote, text=True)
inventory = json.loads(captured)
prior = json.loads(Path('results/infra/consolidation_20261104/final_20261105/manifest.json').read_text())
plan = json.loads((local/'executed_plan.json').read_text())
plan_sha = hashlib.sha256((local/'executed_plan.json').read_bytes()).hexdigest()
assert plan_sha == '8a07e975f1df709729222a3a3c79f5f28713b5613584b87b3428213d896054d2'
manifest = {k: prior[k] for k in ('historical_reference', 'expected_subjects', 'expected_n_rows', 'expected_roles')}
manifest.update(protocol='consolidation-ft15-results-manifest-20261105-v1',
    run_identity={'code_sha': '1359b924d25601f92e9035b8bd621c2c6a93f947',
        'source_sha256': '13de2f317997e497f6d40a53d04dbc6b2061a6d2244c0a809898968362091469',
        'plan_sha256': plan_sha, 'recipe_sha256': plan['recipe_sha256']},
    input_root_remote=plan['fit_output_root'], files=inventory['files'],
    checkpoints=inventory['checkpoints'],
    prior_five={k: plan['prior_five_results'][k] for k in ('report', 'manifest')},
    collection=inventory['observed_at_utc']+'; local non-PT copies; PT payloads hashed remotely only')
for entry in manifest['files']:
    raw = (local/'epn15'/entry['relative_path']).read_bytes()
    assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256']
with (local/'remote_inventory.json').open('x') as f:
    json.dump(inventory, f, indent=2)
with (local/'manifest.json').open('x') as f:
    json.dump(manifest, f, indent=2)
PY
```

L'aggregatore verifica inventario completo e ogni hash (compresi i pin remoti
dei candidati), i blob Git dello snapshot realmente eseguito, selezione sulla
validation, dose e pairing. Ricalcola anche il risultato5epoche dalle OOF:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python scripts/consolidation_ft15_report.py \
  --input-root /private/tmp/wearusfm-ft15-claude-final-20261105/epn15 \
  --input-root-five /private/tmp/wearusfm-ft15-claude-final-20261105/epn5 \
  --plan /private/tmp/wearusfm-ft15-claude-final-20261105/executed_plan.json \
  --manifest /private/tmp/wearusfm-ft15-claude-final-20261105/manifest.json \
  --out /private/tmp/wearusfm-ft15-claude-final-20261105/report.json
```

`--out` deve essere nuovo e fuori dalle due radici input. Se manca una cella
o fallisce un pin non si elimina un file dal manifest per farlo passare:
conservare l'errore e diagnosticare. I file/report delle5epoche restano
immutati. Non scegliere dose/LR dopo aver visto il nuovo held.

Pubblicare nel ramo di esecuzione una nuova directory
`results/infra/finetune15_20261105/final/` con manifest, report, accounting,
ricevute/curve JSON e log pertinenti (senza certificati SSH); le OOF rimangono
nel deposito locale esternoGit, i `.pt` suWORK. Prima del commit verificare
che non entrino NPZ, dati wearable o checkpoint. Scrivere la lettura nei
limiti congelati, registrare nelle due sezioni e nelle decisioni, aggiornare
tabella e consuntivo; commit/push con SHA remoto verificato. GPT verifica
successivamente la raccolta e la lettura, senza bloccare la gestione operativa.

## Anomalie e limiti aperti al passaggio

- Nessun errore attivo comunicato da Claude alle21:05; non equivale a una
  verifica nuova di GPT dei16fit non conclusi.
- QuotaWORK condivisa:16checkpointFT15 ancora da scrivere,1,692672GiB più
  report/OOF. La nuova collocazione del lungo suSCRATCH non sposta i fit FT15.
- Consuntivo finale degli8frozen non ancora raccolto da GPT; usare il sacct
  sopra, senza presentare0,125833GPUh come consumo totale corrente.
- Le interruzioni dei tool durante gli invii sono già risolte: nessun ID
  rimasto prenotato/ambiguo e nessun duplicato. Lo storico è in
  docs/finetune15_execution_20261105.md; non tentare di correggere quei record.
- Nessun monitor automatico attivo di GPT da trasferire e nessun nuovo
  esperimento implicito nel passaggio operativo.
