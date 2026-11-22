"""Scettico RC-2: rigenera i dati del giocattolo con lo script dell'altro revisore (importato in sola lettura), scrivendo nella MIA cartella."""
import sys
sys.path.insert(0, "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/residual_candidate")
import toy_res as TR
TR.WORK = "/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/rc2_work/"
TR.gen(1)
