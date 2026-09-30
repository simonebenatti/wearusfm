#!/usr/bin/env python3
"""Controlla che le citazioni compaiano PAROLA PER PAROLA nel testo di una pagina (metodo usato il 30/09/2026 per i fogli D4).

La pagina si scarica con curl (o si legge da un file locale), si tolgono commenti HTML, script e stili e i tag, si normalizzano solo spazi, maiuscole,
apostrofi/virgolette tipografici e trattini; poi si cerca ogni citazione cosi' com'e'. Nessun riassunto, nessun modello: o la frase c'e' o non c'e'.
Per le citazioni assenti riporta il contesto della parola piu' rara della citazione, per aiutare a capire se e' una parafrasi o un difetto dell'estrazione.

  python3 scripts/check_quotes.py --url https://ninapro.hevs.ch/instructions/DB5.html --quote "tilted by 22.5 degrees clockwise" --quote "..."
  python3 scripts/check_quotes.py --json richieste.json      # [{"url": ..., "quotes": [...]}, ...]
Esce con 1 se almeno una citazione e' assente.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

_TRANSLATE = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " ", "−": "-"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(text))
    for a, b in _TRANSLATE.items():
        text = text.replace(a, b)
    text = re.sub(r"\s+", " ", text)
    # un tag tolto in mezzo a una frase lascia uno spazio prima della punteggiatura ("<b>Myo</b>, tilted" -> "Myo , tilted"): lo si toglie
    text = re.sub(r" ([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[]) ", r"\1", text)
    return text.strip().lower()


def page_text(raw: str) -> str:
    """Testo visibile di una pagina HTML (senza commenti, script e stili). Per un file di testo o codice restituisce il testo com'e'."""
    raw = re.sub(r"(?s)<!--.*?-->", " ", raw)
    raw = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
    return re.sub(r"(?s)<[^>]+>", " ", raw)


def fetch(url: str, timeout_s: int = 60) -> str:
    if not url.startswith(("http://", "https://")):
        return Path(url).read_text(errors="replace")
    r = subprocess.run(["curl", "-sL", "--max-time", str(timeout_s), "-A", "Mozilla/5.0 (quote check)", url], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl exit {r.returncode} su {url}")
    return r.stdout.decode("utf-8", "replace")


def check(url: str, quotes: list[str], raw: str | None = None, context: int = 160) -> list[dict]:
    text = normalize(page_text(raw if raw is not None else fetch(url)))
    out = []
    for q in quotes:
        nq = normalize(q)
        pos = text.find(nq)
        item = {"url": url, "quote": q, "found": pos >= 0, "page_chars": len(text)}
        if pos < 0:  # contesto intorno alla parola piu' lunga della citazione: aiuta a vedere cosa dice davvero la pagina
            words = sorted(set(re.findall(r"[\w.,%-]{4,}", nq)), key=len, reverse=True)
            for w in words:
                j = text.find(w)
                if j >= 0:
                    item["nearby"] = text[max(0, j - context): j + context]
                    break
        out.append(item)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--quote", action="append", default=[])
    ap.add_argument("--json", type=Path, help="lista di {url, quotes}")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args(argv)
    requests = json.loads(args.json.read_text()) if args.json else []
    if args.url:
        requests.append({"url": args.url, "quotes": args.quote})
    if not requests or not all(r.get("quotes") for r in requests):
        ap.error("servono --url con almeno una --quote, oppure --json")
    results = []
    for r in requests:
        try:
            results += check(r["url"], r["quotes"])
        except Exception as e:  # una pagina irraggiungibile non ferma le altre
            results += [{"url": r["url"], "quote": q, "found": False, "error": f"{type(e).__name__}: {e}"} for q in r["quotes"]]
    for x in results:
        tag = "TROVATA" if x["found"] else ("ERRORE " if "error" in x else "ASSENTE")
        print(f"{tag} | {x['url']} | {x['quote'][:90]}")
        if not x["found"] and "nearby" in x:
            print(f"        la pagina vicino: ...{x['nearby']}...")
        if "error" in x:
            print(f"        {x['error']}")
    if args.report:
        args.report.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if all(x["found"] for x in results) else 1


if __name__ == "__main__":
    sys.exit(main())
