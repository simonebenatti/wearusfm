"""Controllo delle correzioni: per ogni segnalazione cerca nella bozza congelata le frasi attese (presenti o assenti)
e stampa le righe. Sola lettura; scrive solo l'uscita accanto a questo script."""
import hashlib, os, sys

BASE = "/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/scratchpad/tcnreview2"
BOZZA = os.path.join(BASE, "controllo/bozza_controllo.md")
OUT = os.path.join(BASE, "controllo/correzioni/cerca_passi.out.txt")
SHA_ATTESO = "5e872e1d2089df94e2907c6065fbcbe27938db3ac4dcc14aa85643ed739a6dcf"

raw = open(BOZZA, "rb").read()
sha = hashlib.sha256(raw).hexdigest()
righe = raw.decode("utf-8").split("\n")
if righe and righe[-1] == "":
    righe = righe[:-1]

# (id, frase, atteso) ; atteso True = deve esserci, False = non deve esserci
CERCA = [
    ("N1", "massima verosimiglianza", False),
    ("N1", "fare meglio della forma chiusa non regolarizzata", True),
    ("N2", "K_dyn 2,5 e K_MSE 5", True),
    ("N2", "K_dyn 3,5 e K_MSE 7", True),
    ("N2", "con K 3,5 e 7 lo stesso caso passa", True),
    ("N2", "script_agenti/rilettura_riferimenti/", True),
    ("N3", "smette di passare", True),
    ("N3", "supera circa 0,90-0,94", False),
    ("N4", "HH:MM", True),
    ("N5", "lo 0,014%", True),
    ("N5", "per lo 0,014%", True),
    ("N5", "dello 0,01%", True),
    ("N6", "Controesempio (numeri d'esempio): nel controllo temporale", True),
    ("N7", "Controesempio (numeri d'esempio): la TCN-post", True),
    ("N8", "che J pinna solo per sha256", True),
    ("N8", "byte uguali ai pin", False),
    ("N9", "riga 329; accolto a R:27", True),
    ("N10", "la `pred` della cache è float32", True),
    ("N11", "prima di I/O, avvii e salvataggi", True),
    ("N11", "il gate vero è un po' più alto", True),
    ("N12", "prefisso `32e5452:`", True),
    ("N12", "`32e5452:N3`", True),
    ("R2", "prompt = `docs/prompt_claude_pose_temponet_review_20261117.md`", True),
    ("R2", "stanno in `results/infra/pose_temponet_proposal_20261117/`", True),
    ("R4", "Solo correzioni di testo", False),
    ("R4", "dettagli per il codice (sezione 6)", True),
    ("R5", "Su L0 toglie il primario, su L13 la candidatura", True),
    ("R5", "Su L0 toglie il primario (P:104), su L13 la candidatura", True),
    ("R8", "candidata, non peggiore in modo pratico di MLP/L13", True),
    ("R8", "toglierla", False),
    ("R11", "peggiore delle storiche", False),
    ("R11", "deteriorata rispetto alla ridge", True),
    ("R12", "1 no, 4 e 5 sì", True),
    ("R13", "riportando comunque direzione e ampiezza. Non chiamarlo equivalenza.", True),
    ("R14", "(note, facoltativa)", True),
    ("R14", "(note, facoltative)", True),
    ("R14", "facoltativa ma utile a Simone", True),
    ("R15", "frase della nota 12 a P:122", True),
    ("F1", "si sospendono solo le conclusioni negative o nulle", True),
    ("F1", "esclude comunque la candidatura", True),
    ("F2", "compresa l'assenza di", True),
    ("F2", "la candidata richiede L0, MLP e L13 ammessi", True),
    ("F3", "Fa eccezione la clausola MLP di BL-2", True),
    ("F4", "classificazione mia; per il verificatore converge con correzioni", True),
    ("F4", "quindi falsa un risultato", True),
    ("F4", "Basta togliere una parola", True),
    ("F4", "percorso probabile", False),
    ("F5", "prevale sulle etichette di OB-1", True),
    ("F5", "Ordine fisso degli stati", True),
    ("F6", "sia nella banda [6%, 10%] sia nel margine del cancello del 50%", True),
    ("F6", "Il cancello diventa più severo", True),
    ("F8", "oppure una banda sganciata dall'ancora", True),
    ("F8", "non qualifica la lettura temporale", True),
    ("F8", "In alternativa, una tolleranza dichiarata ora", True),
    ("F8", "Revisore e verificatore lo danno minore", True),
    ("F9", "con quanto vuole della sezione 5", False),
    ("F9", "GPT le corregge nella stessa revisione; le ricontrollo alla review del pacchetto operativo", True),
    ("F9", "Rileggo solo le righe che toccano BL, OB e le minori", True),
    ("F9", "servirebbe una modifica", True),
    ("F10", "voce vii di `ricetta-9`", True),
    ("F11", "sbaglia molto per stima", False),
    ("F11", "fa pensare grande", True),
    ("F11", "plausibile, non stabilito", True),
    ("F12", "può scattare", True),
    ("F12", "argomento qualitativo", True),
    ("F13", "(a) e (b) vanno insieme", True),
    ("F13", "penalità sulla somma dei quadrati", True),
    ("F13", "il 90,1% con l'alpha degli inner non riscalato", True),
    ("F13", "loss media", False),
    ("F13", "90,1-90,8%", True),
    ("F14", "In P:72, P:98, P:104 e nel §10", True),
    ("F14", "passo 0 escluso", True),
    ("F15", "con L in FP32 il controllo passa", True),
    ("F16", "e la ricetta non c'entra", False),
    ("F16", "quindi il controllo non dice se la ricetta sfrutta il tempo", True),
    ("F17", "Complete e fedeli nel testo; convergono con correzioni", True),
    ("F17", "complete e fedeli nel testo; convergono con correzioni", True),
    ("F19", "manca quello della scelta al primo quarto di epoca", True),
    ("F19", "manca quello del primo quarto di epoca", True),
    ("F20", "con a ≤ 1, la calibrazione può anche non trovare", True),
    ("F20", "(con l'errore di stima a K = 3)", True),
    ("F21", "Va riscritta anche P:104", True),
    ("F21", "Riscrivi anche P:104, dicendo se la catena prosegue con allarmi aperti.", True),
    ("F21", "vale per L0, non per la TCN-post", True),
]

out = []
out.append("bozza: %s" % BOZZA)
out.append("sha256: %s (%s); righe: %d" % (sha, "OK" if sha == SHA_ATTESO else "DIVERSO", len(righe)))
for sid, frase, atteso in CERCA:
    trovate = [i + 1 for i, r in enumerate(righe) if frase in r]
    esito = "OK" if (bool(trovate) == atteso) else "ATTENZIONE"
    out.append("%-4s %-10s %s atteso=%s righe=%s" % (sid, esito, repr(frase)[:90], "presente" if atteso else "assente", trovate))
open(OUT, "w").write("\n".join(out) + "\n")
print("\n".join(out))
