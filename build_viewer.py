#!/usr/bin/env python3
"""Build the single-file replay viewer: python build_viewer.py <replay_data.json> <template.html> <out.html>"""
import hashlib, json, sys
from pathlib import Path
src, tpl, out = map(Path, sys.argv[1:4])
if out.exists():
    sys.exit(f"{out} exists; move it away to rebuild.")
raw = src.read_bytes()
data = json.loads(raw)
blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
html = tpl.read_text(encoding="utf-8").replace("__DATA__", blob)
out.write_text(html, encoding="utf-8")
print("data SHA-256:", hashlib.sha256(raw).hexdigest().upper(), "| tasks:", data["n_tasks"], "| size:", len(html) // 1024, "KB")
