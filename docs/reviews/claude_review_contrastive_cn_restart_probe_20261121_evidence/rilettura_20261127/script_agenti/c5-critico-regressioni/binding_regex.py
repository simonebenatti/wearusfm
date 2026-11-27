import re, json, sys
BLOCK = re.compile(r"^```cn-restart-review[ \t]*\n(.*?)\n```[ \t]*$", re.MULTILINE | re.DOTALL)
old = open(sys.argv[1]).read()
blk = '{"schema":"cn-restart-review-v1","status":"APPROVED_FOR_DIAGNOSTIC","reviewed_commit":"8c72655c3d894c82fd4918ed523baf0ac9c69e23","package_sha256":"e9310d67b8304c064e09d71b0ec55d37b57abf3b4e2090c2d02d4def586f702d","plan_sha256":"4f538cf183f6e3daf7bd30a5612eecdbbd07106f64d5f348a2f3e759e877853b"}'
cases = {
 "storico intatto": old,
 "storico + blocco a margine": old + "\n## Blocco\n\n```cn-restart-review\n" + blk + "\n```\n",
 "storico + blocco rientrato in lista": old + "\n- blocco:\n  ```cn-restart-review\n  " + blk + "\n  ```\n",
 "storico + blocco + prompt GPT che lo ripete in un recinto": old + "\n```cn-restart-review\n" + blk + "\n```\n\n````text\nprompt\n```cn-restart-review\n" + blk + "\n```\n````\n",
 "blocco con CRLF": old + "\r\n```cn-restart-review\r\n" + blk + "\r\n```\r\n",
}
for k, v in cases.items():
    b = BLOCK.findall(v)
    ok = len(b) == 1
    if ok:
        try: json.loads(b[0])
        except Exception as e: ok = f"json error {e}"
    print(f"{k}: blocchi={len(b)} valido={ok}")
