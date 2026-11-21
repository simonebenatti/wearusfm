# carica la copia corretta (str(tensor_path)) al posto del modulo del clone
import importlib.util, sys
spec = importlib.util.spec_from_file_location("contrastive_cn_restart_observe", sys.argv_fix)
