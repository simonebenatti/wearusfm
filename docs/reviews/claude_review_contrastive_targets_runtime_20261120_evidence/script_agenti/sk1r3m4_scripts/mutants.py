"""Independent mutation set (skeptic 1, R3-M4). Each entry: id, label, file, old, new.

Written from a fresh reading of fbfde797 (scripts identical at 0efa4d45). Where possible the
mutation removes only the raise (or only the guarded condition) and keeps the pin checks
around it, so a survivor means precisely that the specific guard is unobserved by tests.
"""
REL = "scripts/contrastive_release.py"
JOB = "scripts/contrastive_job.py"
RUN = "scripts/contrastive_run.py"
RD = "scripts/contrastive_benchmark_readout.py"

MUTANTS = [
    # ---- review_payload (block parser)
    ("MU1", "piu' blocchi ammessi, usa il primo", REL,
     "    if len(blocks) != 1:\n", "    if not blocks:\n"),
    ("MU2", "status APPROVED non richiesto", REL,
     ' or payload["status"] != "APPROVED":', ":"),
    ("MU3", "insieme esatto delle chiavi non richiesto", REL,
     'if set(payload) != keys or payload["schema"]', 'if payload["schema"]'),
    # ---- validate_review_binding
    ("MU4", "verifica Git del commit rivisto saltata", REL,
     "    if verify_git:\n", "    if False:\n"),
    ("MU5", "byte dei piani al commit rivisto non confrontati", REL,
     '                raise ValueError("reviewed plan bytes differ from executable package")', "                pass"),
    ("MU6", "review_payload/original_review_sha256 del binding non confrontati", REL,
     '        raise ValueError("review copy differs from authenticated original")', "        pass"),
    ("MU7", "bind senza identita' byte della copia", REL,
     '        raise ValueError("review copy is not byte-identical to the original review commit")', "        pass"),
    ("MU8", "binding verso piani di un'altra coorte", REL,
     '        raise ValueError("review binding points to another cohort")', "        pass"),
    ("MU9", "legame interno preflight->source/science plan non controllato", REL,
     '        raise ValueError("preflight source/science-plan binding differs")', "        pass"),
    ("MU10", "hash dichiarati non confrontati con sorgenti/piani correnti", REL,
     '        raise ValueError("review does not bind the current sources and both plans")', "        pass"),
    # ---- validate_audit_accounting
    ("MU11", "durata ammessa fino a 7380 s", REL, "not 0 <= elapsed <= 7200", "not 0 <= elapsed <= 7380"),
    ("MU12", "durata 7200 esclusa", REL, "not 0 <= elapsed <= 7200", "not 0 <= elapsed < 7200"),
    ("MU13", "TIMEOUT ammesso", REL, 'row["State"] != "COMPLETED"', 'row["State"] not in ("COMPLETED", "TIMEOUT")'),
    ("MU14", "AllocCPUS non controllato", REL, "or cpus != 8 or minutes != 120", "or minutes != 120"),
    ("MU15", "TimelimitRaw non controllato", REL, "or cpus != 8 or minutes != 120", "or cpus != 8"),
    ("MU16", "ExitCode non controllato", REL, ' or row["ExitCode"] != "0:0"', ""),
    ("MU17", "piu' righe di allocazione ammesse", REL, "    if len(allocation) != 1:\n", "    if not allocation:\n"),
    ("MU18", "ID logico non controllato", REL,
     '        raise ValueError("audit accounting logical ID differs from the new cohort")', "        pass"),
    ("MU19", "source_sha256 del report audit non controllata", REL,
     ' or audit.get("source_sha256") != source)', ")"),
    ("MU20", "audit_sha256 dell'accounting non controllato", REL,
     ' or accounting.get("audit_sha256") != file_sha(audit_path)', ""),
    ("MU33", "intestazione sacct non richiesta", REL,
     "if not records or not required <= set(records[0]):", "if not records:"),
    ("MU37", "job_id non numerico ammesso", REL,
     "if not isinstance(job_id, str) or not job_id.isdigit() or (", "if ("),
    # ---- make_job
    ("MU21", "make_job: gate accounting saltato per le celle non audit", REL,
     "        validate_audit_accounting(audit, job[\"audit_sacct\"])\n", "        (audit, job[\"audit_sacct\"])\n"),
    ("MU38", "make_job: evidenze possono sovrascrivere la cella", REL,
     '        raise ValueError("evidence cannot override the fixed released stage")', "        pass"),
    # ---- checked_pin / git_bytes / verify_reviewed_sources
    ("MU32", "review fuori dal checkout ammessa", REL,
     'review = checked_pin(binding["review"], repository=True)', 'review = checked_pin(binding["review"])'),
    ("MU35", "digest dei sorgenti rivisti non confrontato", REL,
     '        raise ValueError("reviewed source inventory differs from execution digest")', "        pass"),
    ("MU36", "git_bytes senza contenimento del percorso", REL,
     'if not _full_sha(commit) or not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:',
     "if not _full_sha(commit):"),
    ("MU39", "sorgenti non confrontati byte a byte col commit rivisto", REL,
     '            raise ValueError("execution sources differ from reviewed commit: " + relative)', "            pass"),
    # ---- dispatcher (contrastive_job.build_command)
    ("MU22", "dispatcher senza validate_review_binding", JOB,
     "    validate_review_binding(job.get('package_review'))\n", "    job.get('package_review')\n"),
    ("MU23", "dispatcher: review del job non legata al binding", JOB,
     "        raise ValueError('job review differs from its bound package review')", "        pass"),
    ("MU24", "dispatcher senza controllo della cella del preflight", JOB,
     "        raise ValueError('job differs from the reviewed fixed preflight cell')", "        pass"),
    ("MU25", "dispatcher senza gate accounting (pin mantenuti)", JOB,
     "        validate_audit_accounting(checked_file(job['audit']), checked_file(job['audit_sacct']))",
     "        (checked_file(job['audit']), checked_file(job['audit_sacct']))"),
    ("MU26", "dispatcher: out fuori da fresh_root ammesso", JOB,
     "not out.is_relative_to(cohort_root) or out == cohort_root or ", ""),
    ("MU27", "dispatcher: release con altro binding/accounting", JOB,
     "        raise ValueError('job review/audit accounting differs from execution release')", "        pass"),
    # ---- runtime (contrastive_run.validate_release)
    ("MU28", "runtime senza validate_review_binding", RUN,
     '    validate_review_binding(release.get("package_review"))\n', '    release.get("package_review")\n'),
    ("MU29", "runtime: review_pin non legato al binding", RUN,
     '        raise ValueError("release review commit differs from its bound package review")', "        pass"),
    ("MU30", "runtime senza gate accounting", RUN,
     '    validate_audit_accounting(audit_path, release["audit_sacct"])\n', '    (audit_path, release["audit_sacct"])\n'),
    ("MU34", "runtime benchmark senza prerequisito CUDA", RUN,
     '        validate_cuda_prerequisite(release.get("cuda_report"), cohort, pins, calibration_run_dir)',
     '        (release.get("cuda_report"), cohort, pins, calibration_run_dir)'),
    # ---- final readout
    ("MU31", "lettura finale senza gate accounting", RD,
     "    accounting = validate_audit_accounting(cpu_audit_path, cpu_audit_sacct_path, expected_source=cohort['source_sha256'])",
     "    accounting = {}"),
]
