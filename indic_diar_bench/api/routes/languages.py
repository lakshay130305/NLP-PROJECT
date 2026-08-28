"""Single source of truth for the language dropdown: built live from
data/prep/language_codes.py, not duplicated as a second hardcoded list."""

from __future__ import annotations

from fastapi import APIRouter

from data.prep.language_codes import (
    LANGUAGE_TO_ISO639_1,
    whisper_supported_language_codes,
)

router = APIRouter()


@router.get("/languages")
def list_languages():
    supported = whisper_supported_language_codes()
    return {
        "languages": [
            {"name": name, "iso639_1": code, "whisper_supported": code in supported}
            for name, code in sorted(LANGUAGE_TO_ISO639_1.items())
        ]
    }
