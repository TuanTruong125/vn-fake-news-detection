from __future__ import annotations

import hashlib


# Compute deterministic MD5 hash for a UTF-8 text value.
def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()
