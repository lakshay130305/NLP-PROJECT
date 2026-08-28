"""Forces UTF-8 stdout/stderr on every CLI entry point.

Windows consoles default to a legacy codepage (cp1252 on this machine) that cannot
encode most non-Latin-1 characters -- confirmed this session: printing a real ASR
transcript crashed with `UnicodeEncodeError: 'charmap' codec can't encode character`.
This is not an edge case for a 22-Indian-language project (Devanagari, Bengali script,
Tamil script, etc. all fail the same way) -- every CLI entry point must call this before
printing anything that might contain non-ASCII text.
"""

from __future__ import annotations

import sys


def force_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
