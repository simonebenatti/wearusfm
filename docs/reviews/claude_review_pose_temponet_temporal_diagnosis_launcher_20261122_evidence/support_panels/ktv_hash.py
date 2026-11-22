"""Ipotesi: nel train tutte le 1.907 finestre hanno 160 patch reali (key_time_valid tutto vero).
Si confronta l'hash atteso con quello archiviato nella ricevuta del preflight (nessun dato vero letto)."""
import hashlib, json, sys
import numpy as np
def arrays_hash(*arrays):  # copia testuale di scripts/pose_temporal_readout.py:41-47
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([value.dtype.str, value.shape]).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()
receipt = json.load(open(sys.argv[1]))
iv = receipt["details"]["preflight"]["train"]["input_verification"]
full = np.ones((iv["n_windows"], 160), dtype=bool)
h = arrays_hash(full)
print("atteso se tutte le finestre hanno 160 patch:", h)
print("archiviato key_time_valid_sha256           :", iv["key_time_valid_sha256"])
print("UGUALI:", h == iv["key_time_valid_sha256"])
print("central_mask_sha256 (pose_temponet) == targets.central_mask (storico):", iv["central_mask_sha256"] == iv["targets"]["central_mask"])
print("n_windows", iv["n_windows"], "supported_windows", iv["supported_windows"])
