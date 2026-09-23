"""Pre-commit secret scanner: block commits that look like they contain credentials.

Checks staged files for Binance-style API keys/secrets (64 alphanumerics), private keys,
and non-empty assignments to *_KEY / *_SECRET / TOKEN / PASSWORD variables.
Allow a line explicitly with a trailing `# secret-scan: allow` comment.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERNS = [
    ("64-char key/secret", re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{64}(?![A-Za-z0-9])")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "credential in env-style assignment",
        re.compile(
            r"^\s*(export\s+)?[A-Z0-9_]*(KEY|SECRET|TOKEN|PASSWORD)[A-Z0-9_]*\s*=\s*"
            r"['\"]?[^\s'\"#]{8,}"
        ),
    ),
    (
        "credential string literal",
        re.compile(r"(?i)(key|secret|token|password)\w*['\"]?\s*[=:]\s*['\"][^'\"\s]{8,}['\"]"),
    ),
]
ALLOW = "secret-scan: allow"
SKIP_SUFFIXES = {".parquet", ".zip", ".png", ".jpg", ".lock"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def scan(path: Path) -> list[str]:
    if path.suffix in SKIP_SUFFIXES or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        if ALLOW in line:
            continue
        for label, rx in PATTERNS:
            m = rx.search(line)
            if not m:
                continue
            # Lowercase hex digests (sha256 of test data) are not credentials.
            if label == "64-char key/secret" and HEX64.match(m.group(0)):
                continue
            hits.append(f"{path}:{n}: possible {label}")
    return hits


def main(argv: list[str]) -> int:
    hits = [h for a in argv for h in scan(Path(a))]
    for h in hits:
        print(h, file=sys.stderr)
    if hits:
        print(
            "secret-scan: commit blocked. Remove the secret or mark a false positive "
            f"with '# {ALLOW}'.",
            file=sys.stderr,
        )
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
