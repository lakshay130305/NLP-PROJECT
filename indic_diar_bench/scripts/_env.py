"""Loads a repo-root .env file (HF_TOKEN etc.) into os.environ, if one exists. Call at the
top of any script's main() before reading env vars -- .env is gitignored, so this is the
"set it once, don't retype it every session" path, purely local to this machine."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
