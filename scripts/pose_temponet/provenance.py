"""Immutable inputs, explicit review releases and stage receipts for pose readers.

Historical producer identities are never recomputed from the current reader.
This module does not submit jobs, amend plans or promote review states.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import tarfile
import uuid

from . import data as D
# After data: it already loads the historical producer modules, so the order in
# which the numerical libraries are imported stays exactly as before.
from wearusfm.harness.pose_eval import _SAFE_RE as _STORAGE_UNSAFE

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "pose-temponet-frozen-encoder-readout-20261117-v3"
OUTPUT_ROOT = "/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_20261117"
HISTORICAL_COMMIT = "2930d7d6feb6a054ae0ce9fded98ac5ac1b12abe"
DOCUMENTARY_REVIEW = {
    "status":"claude_review_converged","commit":"ad9f5141711ba60ea1014bd9ba842af168ff0a37",
    "proposal_sha256":"171b52785f37f80e0a9f1b6194e90e29bd7620c4b7106c117267465bfc6c2906",
    "artifact":{"path":"docs/reviews/claude_review_pose_temponet_v3_20261119.md",
                "sha256":"25a8d513eb115ab8156dafdf1005a500c6f7e43c609045d2dea4a4c04b81cb48","bytes":13178,
                "local_copy":"docs/reviews/claude_review_pose_temponet_v3_20261119.md"},
}
HISTORICAL_RECEIPTS = {
    "train": {"path":"results/infra/consolidation_long_20261105/execution/receipts/LONG-G49046-pose-train-20261105.json",
              "sha256":"27297241e4bf0f12a61edb2e5cf5d8cef28262f6f26caee44f98af263b3a1f36","bytes":342030,
              "local_copy":"docs/reviews/pose_temponet_runtime_evidence_20261119/producer_train.json"},
    "val": {"path":"results/infra/consolidation_long_20261105/execution/receipts/LONG-G49046-pose-val-20261105.json",
            "sha256":"1321038a8e0f50da56e57543c9b5a662c391af523d7094c6984cad1a396a755c","bytes":3258392,
            "local_copy":"docs/reviews/pose_temponet_runtime_evidence_20261119/producer_val.json"},
}
TRAIN_INPUTS = ("cache_train","target_reference_train","pose_stats")
ALL_INPUTS = (*TRAIN_INPUTS,"cache_val","target_reference_val","post_ridge_coefficients",
              "post_ridge_predictions","trained_head_predictions")
TRAIN_STAGES = ("cpu_benchmark","gpu_benchmark","wls_calibration","controls","selection","refit")
STAGES = ("preflight",*TRAIN_STAGES,"locked_inference","metrics","collection")
NUMERIC_GROUPS = {stage:("gpu8" if stage in ("gpu_benchmark","controls","selection","refit","locked_inference")
                        else "cpu8") for stage in STAGES if stage not in ("preflight","collection")}
RECIPE_KEYS = ("protocol","checkpoint_sha256","checkpoint_step","which","model","support","fit","heads",
               "new_head_common","positive_controls","linear_diagnostics","evaluation")
REQUIRED_GATES = {
    "preflight": (), "cpu_benchmark": ("integrity","historical_R1"),
    "gpu_benchmark": ("integrity","historical_R1"),
    "wls_calibration": ("integrity","historical_R1","resources"),
    "controls": ("integrity","historical_R1","resources","closed_references","calibration"),
    "selection": ("integrity","historical_R1","resources","positive_controls"),
    "refit": ("integrity","historical_R1","resources","positive_controls","selection_locked"),
    "locked_inference": ("integrity","historical_R1","resources","positive_controls","selection_locked","refits_locked"),
    "metrics": ("integrity","historical_R1","resources","positive_controls","selection_locked","refits_locked","inference_locked"),
    "collection": (),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _hex(value,length=64):
    return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{%d}" % length,value) is not None


def safe_path(path, *, root=None):
    """Reject links in every existing component, including dangling links."""
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "path must be absolute without parent traversal")
    if root is not None:
        root = safe_path(root)
        require(path == root or root in path.parents, "output path escapes approved root")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(mode), "symlink component refused: "+str(current))
        if current != path:
            require(stat.S_ISDIR(mode), "non-directory path component")
    return path


def safe_output_root(path=OUTPUT_ROOT):
    path = safe_path(path)
    require(str(path) == OUTPUT_ROOT, "unregistered output root")
    return path


@contextmanager
def _parent_fd(path, *, create=False):
    """Directory descriptors prevent a checked parent being replaced by a link."""
    path = safe_path(path)
    fd = os.open(path.anchor,os.O_RDONLY|os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            if create:
                try:
                    os.mkdir(component,mode=0o755,dir_fd=fd)
                except FileExistsError:
                    pass
            newer = os.open(component,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd)
            fd = newer
        yield fd,path.name
    finally:
        os.close(fd)


def _stat_record(value):
    require(stat.S_ISREG(value.st_mode), "expected regular file")
    return {"bytes":value.st_size,"device":value.st_dev,"inode":value.st_ino,
            "mtime_ns":value.st_mtime_ns,"ctime_ns":value.st_ctime_ns}


def file_state(path):
    with _parent_fd(path) as (directory,name):
        return _stat_record(os.stat(name,dir_fd=directory,follow_symlinks=False))


def file_pin(path):
    path = safe_path(path)
    with _parent_fd(path) as (directory,name):
        fd = os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
        with os.fdopen(fd,"rb") as stream:
            before = _stat_record(os.fstat(stream.fileno()))
            value = hashlib.sha256()
            for block in iter(lambda: stream.read(8 << 20),b""):
                value.update(block)
            require(_stat_record(os.fstat(stream.fileno())) == before, "input changed while hashing")
        require(_stat_record(os.stat(name,dir_fd=directory,follow_symlinks=False)) == before,
                "input replaced while hashing")
    return {"path":str(path),"sha256":value.hexdigest(),"bytes":before["bytes"]}


def _pin_path(pin, source_root=ROOT):
    require(isinstance(pin,dict) and _hex(pin.get("sha256")), "invalid artifact pin")
    path = Path(pin["path"])
    if not path.is_absolute():
        require(".." not in path.parts, "relative pin escapes repository")
        path = Path(source_root)/path
    return safe_path(path)


def verify_pin(pin, *, source_root=ROOT):
    path = _pin_path(pin,source_root)
    before = file_state(path)
    actual = file_pin(path)
    require(actual["sha256"] == pin["sha256"], "artifact SHA256 mismatch: "+str(path))
    require("bytes" not in pin or actual["bytes"] == pin["bytes"], "artifact byte count mismatch")
    require(file_state(path) == before, "input changed during verification")
    return {"pin":actual,"state":before}


def unchanged(states):
    require(all(file_state(path) == before for path,before in states.items()), "pinned input changed while in use")


def _read_pinned_bytes(pin, *, source_root=ROOT):
    """Hash precisely the bytes parsed, through one no-follow descriptor."""
    path = _pin_path(pin,source_root)
    with _parent_fd(path) as (directory,name):
        fd = os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=directory)
        with os.fdopen(fd,"rb") as stream:
            before = _stat_record(os.fstat(stream.fileno()))
            payload = stream.read()
            require(_stat_record(os.fstat(stream.fileno())) == before, "input changed while reading")
        require(_stat_record(os.stat(name,dir_fd=directory,follow_symlinks=False)) == before,
                "input replaced while reading")
    unchanged({str(path):before})
    require(hashlib.sha256(payload).hexdigest() == pin["sha256"], "artifact SHA256 mismatch: "+str(path))
    require("bytes" not in pin or len(payload) == pin["bytes"], "artifact byte count mismatch")
    return payload


def read_json_pin(pin, *, source_root=ROOT):
    return json.loads(_read_pinned_bytes(pin,source_root=source_root))


def _atomic(path, writer, *, root):
    """Commit an entirely written file by exclusive hard-link, never replace."""
    path = safe_path(path,root=root)
    require(path != Path(root), "artifact cannot replace output root")
    with _parent_fd(path,create=True) as (directory,name):
        temporary = "."+name+"."+uuid.uuid4().hex+".tmp"
        fd = os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
        try:
            with os.fdopen(fd,"wb") as stream:
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary,name,src_dir_fd=directory,dst_dir_fd=directory,follow_symlinks=False)
            os.fsync(directory)
        finally:
            os.unlink(temporary,dir_fd=directory)
    return file_pin(path)


def atomic_json(path, value, *, root):
    payload = json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n"
    return _atomic(path,lambda stream:stream.write(payload),root=root)


def atomic_torch(path, value, *, root):
    import torch
    return _atomic(path,lambda stream:torch.save(value,stream),root=root)


def atomic_npz(path, value, *, root):
    import numpy as np
    require(all(not np.asarray(v).dtype.hasobject for v in value.values()), "object array refused")
    return _atomic(path,lambda stream:np.savez_compressed(stream,**value),root=root)


def source_manifest(source_root=ROOT):
    root = safe_path(source_root)
    paths = set((root/"src").rglob("*.py")) | set((root/"scripts").glob("*.py"))
    paths |= {p for p in (root/"scripts/pose_temponet").rglob("*") if p.suffix in (".py",".sh",".sbatch")}
    library = root/"scripts/slurm/lib_code_dir.sh"
    if library.exists():
        paths.add(library)
    paths |= set((root/"scripts/slurm").glob("pose_temponet*.sbatch"))
    checkout = root/"scripts/slurm/campaign_checkout.sbatch"
    if checkout.exists():
        paths.add(checkout)
    require(paths, "empty source manifest")
    return {str(p.relative_to(root)):file_pin(p)["sha256"] for p in sorted(paths)}


def source_identity(source_root=ROOT):
    return digest(source_manifest(source_root))


def git_artifact(pin, *, source_root=ROOT):
    require(_hex(pin.get("commit"),40), "review/producer requires full commit SHA")
    path = Path(pin["path"])
    require(not path.is_absolute() and ".." not in path.parts and path.parts, "invalid git artifact path")
    try:
        raw = subprocess.check_output(["git","show",pin["commit"]+":"+path.as_posix()],cwd=source_root,
                                      stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        require("local_copy" in pin, "original git artifact unavailable without materialized evidence")
        copied = {**pin,"path":pin["local_copy"]}
        raw = _read_pinned_bytes(copied,source_root=source_root)
    require(hashlib.sha256(raw).hexdigest() == pin["sha256"] and
            ("bytes" not in pin or len(raw) == pin["bytes"]), "git artifact pin mismatch")
    return raw


def storage_directory(user):
    """Directory that the historical producer reads for one cache user ID.

    Same rule as wearusfm.harness.pose_eval.recordings_for and the emg2pose
    ingest: "u" followed by the ID, every unsafe character replaced by "_".
    """
    require(isinstance(user,str) and user, "invalid cache user ID")
    return "u"+_STORAGE_UNSAFE.sub("_",user)


def cache_user(directory):
    """Cache user ID of one producer storage directory, only where it is unique.

    The cache metadata stores the ID, the producer file list stores the
    directory. Every replaced character becomes "_": a directory without "_"
    has exactly one preimage. Any other name fails closed instead of guessing.
    """
    require(isinstance(directory,str) and re.fullmatch(r"u[0-9A-Za-z.@-]+",directory) is not None,
            "producer storage directory has no unique cache user ID")
    user = directory[1:]
    require(storage_directory(user) == directory, "producer storage directory does not round-trip")
    return user


def historical_producer(receipt, *, split, cache_pin, stats_sha256):
    require(split in ("train","val"), "wrong producer split")
    require(receipt["complete"] is True and receipt["checkpoint_unchanged"] is True and
            receipt["frozen_teacher_unchanged"] is True and receipt["raw_inputs"]["unchanged"] is True,
            "incomplete or mutated historical producer")
    require(receipt["protocol"] == "consolidation-long-evaluation-20261105-v1" and
            receipt["kind"] == "extract-pose-"+split and receipt["split"] == split and
            receipt["which"] == "teacher" and receipt["checkpoint_step"] == 49046, "wrong historical producer")
    artifact = receipt["artifacts"]["cache"]
    require(all(artifact[k] == cache_pin[k] for k in ("sha256","bytes")), "cache not covered by historical receipt")
    directories = set()
    for filename in receipt["raw_inputs"]["file_states"]:
        parts = Path(filename).parts
        if "emg2pose" in parts:
            directories.add(parts[parts.index("emg2pose")+1])
    # The receipt lists storage directories ("u"+ID); cache metadata and every
    # consumer of this list (benchmark, controls, folds) use the ID itself.
    users = sorted(cache_user(directory) for directory in directories)
    details = receipt["details"]
    expected_users = 40 if split == "train" else 15
    require(len(users) == details["n_users"] == expected_users and
            details["n_windows"] == cache_pin["windows"] and details["dim"] == 384,
            "historical cache population differs")
    stats = receipt["checkpoint_guard"]["checkpoint_pose_stats_sha256"]
    require(stats == stats_sha256, "producer pose statistics differ")
    producer = {key:receipt[key] for key in D.PRODUCER_FIELDS if key not in ("pose_stats_sha256","checkpoint_pose_stats_sha256")}
    producer.update(pose_stats_sha256=stats,checkpoint_pose_stats_sha256=stats)
    require(all(_hex(producer[k],40 if k == "code_sha" else 64) for k in D.PRODUCER_FIELDS
                if k.endswith("sha256") or k == "code_sha"), "malformed historical producer digest")
    return {"metadata":producer,"users":users,"storage_directories":sorted(directories),
            "details":copy.deepcopy(details),
            "original_cache":copy.deepcopy(artifact),"preservation_policy":"path moved to WORK; hash and bytes identical"}


def input_pins(proposal):
    return {"cache_train":copy.deepcopy(proposal["caches"]["train"]),
            "cache_val":copy.deepcopy(proposal["caches"]["val"]),
            "target_reference_train":copy.deepcopy(proposal["scratch_dependencies"]["pose_references"]["train"]),
            "target_reference_val":copy.deepcopy(proposal["scratch_dependencies"]["pose_references"]["val"]),
            "pose_stats":copy.deepcopy(proposal["scratch_dependencies"]["pose_stats"]),
            **copy.deepcopy(proposal["references"])}


def recipe_identity(proposal):
    return digest({key:proposal[key] for key in RECIPE_KEYS})


def execution_identity(plan):
    if "budget_amendment" in plan:
        return digest({"base_execution": execution_identity({k: v for k, v in plan.items()
                       if k != "budget_amendment"}), "budget_amendment": plan["budget_amendment"]})
    return digest({"protocol":plan["protocol"],"source_sha256":plan["reader"]["source_sha256"],
                   "proposal_sha256":plan["proposal"]["sha256"],"recipe_sha256":plan["recipe_sha256"],
                   "inputs":plan["inputs"],"producers":plan["producers"],"output_root":plan["output_root"]})


def build_plan(proposal_path, producer_receipts=None, *, source_root=ROOT, jobs=()):
    root = safe_path(source_root)
    path = Path(proposal_path)
    if not path.is_absolute():
        path = root/path
    proposal_pin = file_pin(path)
    require(root in path.parents, "proposal must be inside source repository")
    proposal_pin["path"] = str(path.relative_to(root))
    proposal = read_json_pin(proposal_pin,source_root=root)
    require(proposal["protocol"] == PROTOCOL, "wrong proposal protocol")
    pins = input_pins(proposal)
    original_pins = producer_receipts or {s:{**p,"commit":HISTORICAL_COMMIT} for s,p in HISTORICAL_RECEIPTS.items()}
    require(set(original_pins) == {"train","val"}, "both original producer receipts required")
    producers = {}
    for split,pin in original_pins.items():
        raw = git_artifact(pin,source_root=root) if "commit" in pin else canonical_bytes(read_json_pin(pin,source_root=root))
        receipt = json.loads(raw)
        producers[split] = {**historical_producer(receipt,split=split,cache_pin=pins["cache_"+split],
                                                stats_sha256=proposal["stats_sha256"]),"receipt":copy.deepcopy(pin)}
        require(producers[split]["metadata"]["checkpoint_sha256"] == proposal["checkpoint_sha256"],
                "producer checkpoint differs from approved checkpoint")
    manifest = source_manifest(root)
    plan = {"protocol":PROTOCOL,"status":"REVIEW_REQUIRED","proposal":proposal_pin,
            "reader":{"code_sha":None,"source_sha256":digest(manifest),"source_manifest":manifest},
            "recipe_sha256":recipe_identity(proposal),"inputs":pins,"producers":producers,
            "output_root":OUTPUT_ROOT,"jobs":copy.deepcopy(list(jobs)),
            "resource_policy":copy.deepcopy(proposal["resource_proposal"]),
            "numeric_pins":{"cpu8":None,"gpu8":None},
            "review":{kind:{"status":"REVIEW_REQUIRED","commit":None,"artifact":None,"proposal_sha256":proposal_pin["sha256"]}
                      for kind in ("documentary","operational")},
            "submission_released":False,"background_monitor":False}
    plan["execution_sha256"] = execution_identity(plan)
    if proposal_pin["sha256"] == DOCUMENTARY_REVIEW["proposal_sha256"]:
        git_artifact({**DOCUMENTARY_REVIEW["artifact"],"commit":DOCUMENTARY_REVIEW["commit"]},source_root=root)
        plan["review"]["documentary"] = copy.deepcopy(DOCUMENTARY_REVIEW)
    return validate_plan(plan,source_root=root)


def validate_review(plan, *, source_root=ROOT):
    for kind in ("documentary","operational"):
        entry = plan["review"][kind]
        require(entry.get("status") == "claude_review_converged" and _hex(entry.get("commit"),40),
                "actual Claude "+kind+" review required")
        require(entry.get("proposal_sha256") == plan["proposal"]["sha256"], "review targets another proposal")
        if kind == "operational":
            require(entry.get("source_sha256") == plan["reader"]["source_sha256"], "operational review targets other sources")
        artifact = {**entry["artifact"],"commit":entry["commit"]}
        git_artifact(artifact,source_root=source_root)
    return True


def validate_plan(plan, *, source_root=ROOT, require_review=False, verify_sources=True):
    require(plan["protocol"] == PROTOCOL and plan["background_monitor"] is False, "wrong protocol/monitor policy")
    safe_output_root(plan["output_root"])
    proposal = read_json_pin(plan["proposal"],source_root=source_root)
    require(proposal["protocol"] == PROTOCOL and proposal["checkpoint_step"] == 49046 and
            proposal["which"] == "teacher", "wrong frozen teacher")
    require(plan["recipe_sha256"] == recipe_identity(proposal) and plan["inputs"] == input_pins(proposal),
            "recipe or eight fixed input pins changed")
    require(set(plan["inputs"]) == set(ALL_INPUTS), "unexpected raw/new/held input")
    require(plan["resource_policy"] == proposal["resource_proposal"], "resource policy changed")
    from . import budget_amendment as A1
    A1.validate(plan, source_root=source_root)
    require(digest(plan["reader"]["source_manifest"]) == plan["reader"]["source_sha256"], "source manifest digest differs")
    if verify_sources:
        require(source_manifest(source_root) == plan["reader"]["source_manifest"], "reader sources changed")
    require(set(plan["producers"]) == {"train","val"}, "missing historical producers")
    for split,entry in plan["producers"].items():
        pin = entry["receipt"]
        receipt = (json.loads(git_artifact(pin,source_root=source_root)) if "commit" in pin
                   else read_json_pin(pin,source_root=source_root))
        expected = historical_producer(receipt,split=split,cache_pin=plan["inputs"]["cache_"+split],
                                       stats_sha256=proposal["stats_sha256"])
        require({k:v for k,v in entry.items() if k != "receipt"} == expected, "historical producer identity was rewritten")
        require(entry["metadata"]["checkpoint_sha256"] == proposal["checkpoint_sha256"], "historical checkpoint mismatch")
    require(plan["execution_sha256"] == execution_identity(plan), "execution identity differs")
    require(set(plan["numeric_pins"]) == {"cpu8","gpu8"}, "wrong numerical groups")
    for value in plan["numeric_pins"].values():
        require(value is None or _hex(value), "invalid numeric pin")
    jobs = plan["jobs"]
    require(isinstance(jobs,list) and len({j["id"] for j in jobs}) == len(jobs), "duplicate/malformed jobs")
    for job in jobs:
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",job["id"]) is not None and job["stage"] in STAGES,
                "invalid job identity/stage")
        after = job.get("after",[])
        require(isinstance(after,list) and len(after) == len(set(after)) and job["id"] not in after and
                set(after) <= {j["id"] for j in jobs}, "invalid planned predecessor identities")
    if require_review:
        require(plan["status"] == "claude_review_converged" and plan["submission_released"] is True and
                _hex(plan["reader"].get("code_sha"),40), "execution has not been explicitly released")
        validate_review(plan,source_root=source_root)
    return plan


def load_plan(path, *, source_root=ROOT, require_review=False):
    pin = file_pin(safe_path(Path(path).absolute()))
    return validate_plan(read_json_pin(pin),source_root=source_root,require_review=require_review)


def stage_inputs(plan, stage):
    require(stage in STAGES, "unknown stage")
    if stage in TRAIN_STAGES:
        keys = TRAIN_INPUTS
    elif stage in ("preflight","metrics"):
        keys = ALL_INPUTS
    elif stage == "locked_inference":
        keys = ("cache_val","target_reference_val","pose_stats")
    else:
        keys = ()
    return {k:copy.deepcopy(plan["inputs"][k]) for k in keys}


def verify_inputs(plan, stage, *, source_root=ROOT):
    return {key:verify_pin(pin,source_root=source_root) for key,pin in stage_inputs(plan,stage).items()}


def read_verified_inputs(plan, split, *, stage, depth=None):
    require(split in ("train","val"), "unregistered split")
    allowed = stage_inputs(plan,stage)
    require("cache_"+split in allowed, "stage is not permitted to open "+split+" data")
    stats = verify_pin(allowed["pose_stats"])
    producer = plan["producers"][split]
    data = D.read_cache_pin(allowed["cache_"+split],producer=producer["metadata"],
                            reference_pin=allowed["target_reference_"+split],depth=depth)
    cache_users = sorted({m["user"] for m in data["meta"]})
    require(cache_users == producer["users"], "cache user population differs from original producer")
    require(sorted(storage_directory(user) for user in cache_users) == producer["storage_directories"],
            "cache users do not map onto the producer storage directories")
    verification = data["input_verification"]
    for key in ("targets","key_time_valid_sha256","dim","n_recordings","n_users","n_windows"):
        require(verification[key] == producer["details"][key], "historical cache details changed: "+key)
    unchanged({stats["pin"]["path"]:stats["state"]})
    return data


def preflight(plan):
    """Only old reference predictions are evaluated, never new head outputs."""
    from .metrics import historical_reference_gate
    verified = verify_inputs(plan,"preflight")
    states = {v["pin"]["path"]:v["state"] for v in verified.values()}
    train = read_verified_inputs(plan,"train",stage="preflight",depth="post")
    folds = D.alternating_user_folds(train)
    require([len(f) for f in folds] == [20,20], "expected two 20-user train folds")
    train_info = {"folds":folds,"input_verification":train["input_verification"],
                  "supported_windows":int(train["central_mask"].any(axis=(1,2)).sum())}
    del train
    val = read_verified_inputs(plan,"val",stage="preflight",depth="post")
    result = historical_reference_gate(val,D.read_predictions(plan["inputs"]["post_ridge_predictions"]),
                                      D.read_predictions(plan["inputs"]["trained_head_predictions"]))
    unchanged(states)
    return {"integrity":True,"historical_R1":result,"train":train_info,"inputs":verified,
            "external_access":"historical cache/reference parity only; no new readout predictions or selection"}


def numeric_context(kind="cpu"):
    import numpy as np
    import scipy
    import sklearn
    import sklearn.linear_model  # initialize the same numerical libraries before census
    import torch
    from threadpoolctl import threadpool_info
    require(kind in ("cpu","gpu"), "unknown hardware class")
    fields = ("internal_api","prefix","version","num_threads","threading_layer","architecture")
    cpu = platform.processor() or platform.machine()
    path = Path("/proc/cpuinfo")
    if path.exists():
        cpu = next((line.split(":",1)[1].strip() for line in path.read_text().splitlines()
                    if line.startswith("model name")),cpu)
    hardware = {"kind":kind,"machine":platform.machine(),"cpu_model":cpu}
    if kind == "gpu":
        require(torch.cuda.is_available(), "GPU numeric context requires CUDA")
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        hardware.update(gpu_name=props.name,gpu_capability=[props.major,props.minor],gpu_bytes=props.total_memory)
    context = {"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
               "sklearn":sklearn.__version__,"torch":torch.__version__,"cuda":torch.version.cuda,
               "threadpools":sorted([{k:p.get(k) for k in fields} for p in threadpool_info()],key=canonical_bytes),
               "threads":{"intraop":torch.get_num_threads(),"interop":torch.get_num_interop_threads(),
                          "environment":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}},
               "flags":{"deterministic":torch.are_deterministic_algorithms_enabled(),
                        "deterministic_warn_only":torch.is_deterministic_algorithms_warn_only_enabled(),
                        "cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark,
                        "cudnn_tf32":torch.backends.cudnn.allow_tf32,"matmul_tf32":torch.backends.cuda.matmul.allow_tf32,
                        "CUBLAS_WORKSPACE_CONFIG":os.environ.get("CUBLAS_WORKSPACE_CONFIG")},"hardware":hardware}
    return context


def verify_numeric(plan, stage, context):
    """Check the actual runtime before work, not just when writing its receipt."""
    group = NUMERIC_GROUPS.get(stage)
    if group is None:
        return True
    flags = context["flags"]
    threads = context["threads"]
    require(threads["intraop"] == 8 and threads["interop"] == 1 and
            all(threads["environment"].get(k) == "8" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")) and
            all(p.get("num_threads") == 8 for p in context["threadpools"]), "unfrozen numerical thread settings")
    require(flags["deterministic"] is True and flags["deterministic_warn_only"] is False and
            flags["cudnn_deterministic"] is True and flags["cudnn_benchmark"] is False and
            flags["cudnn_tf32"] is False and flags["matmul_tf32"] is False and
            flags["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8", "unfrozen deterministic/FP32 settings")
    require(context["hardware"]["kind"] == ("gpu" if group == "gpu8" else "cpu"), "wrong measured hardware class")
    expected = plan["numeric_pins"][group]
    require(expected is None or digest(context) == expected, "current numeric context differs from benchmark")
    return True


def checkout_identity(plan, *, source_root=ROOT):
    """Metadata descendants may deploy the identical reviewed source files."""
    actual = subprocess.check_output(["git","rev-parse","HEAD"],cwd=source_root,text=True).strip()
    reviewed = plan["reader"]["code_sha"]
    require(_hex(reviewed,40) and _hex(actual,40), "full reviewed/deployment code SHA required")
    expected = os.environ.get("WEARUSFM_CODE_SHA",os.environ.get("EXPECTED_SHA"))
    require(expected is None or expected == actual, "deployment checkout differs from launcher SHA")
    require(source_manifest(source_root) == plan["reader"]["source_manifest"], "deployment sources differ from reviewed manifest")
    if actual != reviewed:
        subprocess.check_call(["git","merge-base","--is-ancestor",reviewed,actual],cwd=source_root,
                              stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    # Read archive bytes, never extract or execute anything from the reviewed commit.
    raw = subprocess.check_output(["git","archive","--format=tar",reviewed,"--",*plan["reader"]["source_manifest"]],
                                  cwd=source_root,stderr=subprocess.PIPE)
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:") as archive:
        observed = {}
        for member in archive.getmembers():
            if member.isdir():
                continue
            require(member.isfile() and member.name in plan["reader"]["source_manifest"], "unexpected reviewed source archive member")
            observed[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    require(observed == plan["reader"]["source_manifest"], "reviewed commit does not contain the pinned source manifest")
    return {"code_sha":actual,"reviewed_code_sha":reviewed}


def check_release(plan, stage, *, gates, source_root=ROOT):
    validate_plan(plan,source_root=source_root,require_review=True)
    require(stage in STAGES, "unknown execution stage")
    missing = [g for g in REQUIRED_GATES[stage] if gates.get(g) is not True]
    require(not missing, "stage gates not passed: "+", ".join(missing))
    group = NUMERIC_GROUPS.get(stage)
    if group and stage not in ("cpu_benchmark","gpu_benchmark"):
        require(_hex(plan["numeric_pins"].get(group)), "benchmark numerical context not pinned")
    return True


def validate_job_allocation(job, *, environment=None):
    env = os.environ if environment is None else environment
    resources = job["resources"]
    require(str(env.get("SLURM_JOB_ID","")).isdigit(), "not a SLURM job")
    require(int(env.get("SLURM_CPUS_PER_TASK",0)) == resources["cpus"], "allocated CPU count differs")
    gpu_text = str(env.get("SLURM_GPUS_ON_NODE",env.get("SLURM_GPUS","0")))
    require(gpu_text.isdigit() and int(gpu_text) == resources["gpus"], "allocated GPU count differs")
    require(int(env.get("SLURM_JOB_NUM_NODES",env.get("SLURM_NNODES",1))) == 1, "single node required")
    return {"slurm_job_id":str(env["SLURM_JOB_ID"]),"cpus":resources["cpus"],"gpus":resources["gpus"],"nodes":1}


def receipt_path(plan, job_id):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",job_id) is not None, "invalid receipt job id")
    return safe_path(Path(plan["output_root"])/"receipts"/(job_id+".json"),root=plan["output_root"])


def verified_receipt(pin, plan, *, stage=None, numeric_sha256=None, source_root=ROOT, verify_artifacts=True,
                     receipt_mirror_dir=None, _trail=(), _memo=None):
    if receipt_mirror_dir is not None:
        pin = {**pin,"path":str(Path(receipt_mirror_dir)/Path(pin["path"]).name)}
    memo = {} if _memo is None else _memo
    path = str(_pin_path(pin,source_root))
    key = (path,pin["sha256"])
    plan_signature = digest({"execution":plan["execution_sha256"],"reviewed_code":plan["reader"]["code_sha"],
                             "numeric":plan["numeric_pins"],"jobs":plan["jobs"]})
    cached = memo.get(key)
    if cached is not None:
        receipt = cached["receipt"]
        require(receipt["job_id"] not in _trail, "cyclic receipt chain")
        require(cached["plan_signature"] == plan_signature, "receipt memo belongs to another phase plan")
        unchanged({k[0]:entry["state"] for k,entry in memo.items()})
        require(stage is None or receipt["stage"] == stage, "wrong predecessor stage")
        require(numeric_sha256 is None or receipt["numeric_sha256"] == numeric_sha256, "numerical context differs")
        if verify_artifacts and not cached["verified_artifacts"]:
            for artifact in receipt["artifacts"].values():
                safe_path(artifact["path"],root=plan["output_root"])
                verify_pin(artifact,source_root=source_root)
            cached["verified_artifacts"] = True
        return receipt
    before = file_state(path)
    receipt = read_json_pin(pin,source_root=source_root)
    require(receipt["complete"] is True and receipt["status"] == "completed", "predecessor did not complete")
    jobs = [j for j in plan["jobs"] if j["id"] == receipt["job_id"]]
    require(len(jobs) == 1 and receipt["stage"] == jobs[0]["stage"], "receipt job/stage differs from plan")
    require(receipt["job_id"] not in _trail, "cyclic receipt chain")
    from . import budget_amendment as A1
    identity_plan = A1.receipt_identity_plan(plan, receipt, pin, source_root=source_root)
    require(receipt["protocol"] == identity_plan["protocol"] and
            receipt["execution_sha256"] == identity_plan["execution_sha256"] and
            receipt["source_sha256"] == identity_plan["reader"]["source_sha256"] and
            receipt["recipe_sha256"] == identity_plan["recipe_sha256"] and
            receipt["reviewed_code_sha"] == identity_plan["reader"]["code_sha"] and _hex(receipt["code_sha"],40),
            "predecessor source/recipe/execution identity differs")
    require(stage is None or receipt["stage"] == stage, "wrong predecessor stage")
    require(receipt["numeric_sha256"] == digest(receipt["numeric_context"]), "receipt numerical context is corrupt")
    expected = numeric_sha256 or plan["numeric_pins"].get(NUMERIC_GROUPS.get(receipt["stage"]))
    require(expected is None or expected == receipt["numeric_sha256"], "benchmark/fit numerical context differs")
    require(receipt["receipt_sha256"] == digest({k:v for k,v in receipt.items() if k != "receipt_sha256"}),
            "receipt payload digest differs")
    ancestors = [verified_receipt(p,plan,source_root=source_root,verify_artifacts=False,
                                  receipt_mirror_dir=receipt_mirror_dir,
                                  _trail=(*_trail,receipt["job_id"]),_memo=memo) for p in receipt["predecessors"]]
    expected_after = jobs[0].get("after",[])
    require(sorted(r["job_id"] for r in ancestors) == sorted(expected_after) and
            len(expected_after) == len(set(expected_after)), "receipt predecessor chain differs")
    if verify_artifacts:
        for artifact in receipt["artifacts"].values():
            require(Path(plan["output_root"]) in Path(artifact["path"]).parents, "receipt artifact outside output root")
            verify_pin(artifact,source_root=source_root)
    unchanged({path:before})
    memo[key] = {"receipt":receipt,"state":before,"verified_artifacts":verify_artifacts,"plan_signature":plan_signature}
    return receipt


def make_receipt(plan, job_id, *, numeric, predecessors=(), artifacts=None, gates=None, details=None,
                 source_root=ROOT, status="completed"):
    jobs = [j for j in plan["jobs"] if j["id"] == job_id]
    require(len(jobs) == 1, "receipt job is not uniquely planned")
    job = jobs[0]
    stage = job["stage"]
    require(status in ("completed","technical_failure","gate_failed"), "unknown receipt status")
    gates = {} if gates is None else dict(gates)
    if status == "completed":
        check_release(plan,stage,gates=gates,source_root=source_root)
    else:
        validate_plan(plan,source_root=source_root)
    code = checkout_identity(plan,source_root=source_root)
    memo = {}
    parents = [verified_receipt(p,plan,source_root=source_root,_memo=memo) for p in predecessors]
    expected = job.get("after",[])
    require(sorted(p["job_id"] for p in parents) == sorted(expected) and len(expected) == len(set(expected)),
            "receipt predecessor chain differs")
    expected_numeric = plan["numeric_pins"].get(NUMERIC_GROUPS.get(stage))
    if status == "completed":
        verify_numeric(plan,stage,numeric)
    elif expected_numeric is not None:
        # Preserve the mismatching actual context in failure evidence.
        gates["numeric_context_matches_benchmark"] = expected_numeric == digest(numeric)
    artifacts = {} if artifacts is None else copy.deepcopy(artifacts)
    for artifact in artifacts.values():
        safe_path(artifact["path"],root=plan["output_root"])
        verify_pin(artifact,source_root=source_root)
    receipt = {"protocol":plan["protocol"],"job_id":job_id,"stage":stage,"status":status,
               "complete":status == "completed",**code,"source_sha256":plan["reader"]["source_sha256"],
               "recipe_sha256":plan["recipe_sha256"],"proposal_sha256":plan["proposal"]["sha256"],
               "execution_sha256":plan["execution_sha256"],"phase_plan_sha256":digest(plan),
               "numeric_context":copy.deepcopy(numeric),"numeric_sha256":digest(numeric),
               "predecessors":copy.deepcopy(list(predecessors)),"artifacts":artifacts,"gates":gates,
               "review":copy.deepcopy(plan["review"]),"details":{} if details is None else copy.deepcopy(details)}
    receipt["receipt_sha256"] = digest(receipt)
    return receipt


def write_receipt(plan, receipt):
    require(receipt["receipt_sha256"] == digest({k:v for k,v in receipt.items() if k != "receipt_sha256"}),
            "cannot write altered receipt")
    return atomic_json(receipt_path(plan,receipt["job_id"]),receipt,root=plan["output_root"])
