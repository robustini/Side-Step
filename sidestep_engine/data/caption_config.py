"""
Shared caption generation configuration.

Provides the system prompt (user-overridable from file), default
generation parameters, and prompt building utilities used by all
caption providers.  Custom prompts are loaded from
``~/.config/sidestep/caption_prompt.txt`` (or platform equivalent).
"""

from __future__ import annotations

import ast
import json
import re
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Default generation parameters ────────────────────────────────────
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_MAX_TOKENS: int = 2048
DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL: str = "gpt-4o"
DEFAULT_LOCAL_MODEL: str = "Qwen/Qwen2.5-Omni-7B"

GEMINI_MODEL_SUGGESTIONS: list[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
]

_DEFAULT_SYSTEM_PROMPT = (
    "You are a music metadata assistant. You will receive a song's "
    "title, artist, and optionally a lyrics excerpt. When an audio "
    "file is attached, listen to it carefully.\n\n"
    "Return EXACTLY the following key: value lines (no extra text):\n"
    "  caption: <1-2 sentence description of style, mood, sonic character>\n"
    "  genre: <comma-separated genre tags, e.g. 'hip hop, rap'>\n"
    "  bpm: <estimated BPM as integer, e.g. 120>\n"
    "  key: <musical key, e.g. 'C minor' or 'F# major'>\n"
    "  signature: <time signature, e.g. '4/4'>\n\n"
    "Rules:\n"
    "- Do NOT include the artist name or song title in the caption.\n"
    "- Focus on instrumentation, tempo feel, and emotional tone.\n"
    "- If you cannot determine a field, write 'N/A' as the value.\n"
    "- Output ONLY the five key: value lines, nothing else."
)


def _prompt_override_path() -> Path:
    """Path to the user's custom system prompt file."""
    from sidestep_engine.settings import settings_dir
    return settings_dir() / "caption_prompt.txt"


def get_system_prompt() -> str:
    """Return the system prompt, loading from file override if present.

    Checks ``~/.config/sidestep/caption_prompt.txt`` (or platform
    equivalent).  Falls back to the built-in default.

    Returns:
        System prompt string.
    """
    p = _prompt_override_path()
    if p.is_file():
        try:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                logger.debug("Loaded custom caption prompt from %s", p)
                return text
        except OSError as exc:
            logger.warning("Failed to read custom prompt: %s", exc)
    return _DEFAULT_SYSTEM_PROMPT


def build_user_prompt(
    title: str, artist: str, lyrics_excerpt: str = "",
) -> str:
    """Build the user prompt from song metadata.

    Args:
        title: Song title.
        artist: Artist name.
        lyrics_excerpt: Optional lyrics text (truncated to 500 chars).

    Returns:
        Formatted user prompt string.
    """
    prompt = f"Title: {title}\nArtist: {artist}"
    if lyrics_excerpt:
        prompt += f"\nLyrics excerpt:\n{lyrics_excerpt[:500]}"
    return prompt


# Keys we expect from the structured AI response
_STRUCTURED_KEYS = frozenset({"caption", "genre", "bpm", "key", "signature"})


def _clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        parts = [_clean_scalar(v) for v in value]
        return ", ".join(p for p in parts if p)
    text = str(value).strip()
    if not text or text.lower() == "n/a":
        return ""
    return text


def _extract_structured_from_mapping(data: dict[str, Any]) -> dict[str, str]:
    lowered = {str(k).strip().lower(): v for k, v in (data or {}).items()}

    def pick(*names: str) -> str:
        for name in names:
            if name in lowered:
                text = _clean_scalar(lowered.get(name))
                if text:
                    return text
        return ""

    result: dict[str, str] = {}
    caption = pick("caption", "description", "summary")
    genre = pick("genre", "genres")
    bpm = pick("bpm", "tempo", "tempo (bpm)")
    key = pick("key", "key_scale")
    signature = pick("signature", "timesignature", "time signature", "time_signature")

    if caption:
        result["caption"] = caption
    if genre:
        result["genre"] = genre
    if bpm:
        result["bpm"] = bpm
    if key:
        result["key"] = key
    if signature:
        result["signature"] = signature
    return result


def _maybe_parse_mapping_text(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    if not (stripped.startswith("{") and stripped.endswith("}")):
        return None
    try:
        parsed = json.loads(stripped)
    except Exception:
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


def _extract_from_mapping_blob_text(text: str) -> dict[str, str]:
    s = str(text or "").strip()
    if not s:
        return {}

    patterns = {
        "caption": [r'''[\'\"]caption[\'\"]\s*:\s*[\'\"]([^\'\"]+)[\'\"]'''],
        "genre": [
            r'''[\'\"]genres[\'\"]\s*:\s*\[([^\]]+)\]''',
            r'''[\'\"]genre[\'\"]\s*:\s*[\'\"]([^\'\"]+)[\'\"]''',
        ],
        "bpm": [r'''[\'\"](?:bpm|tempo|tempo \(bpm\))[\'\"]\s*:\s*[\'\"]?([^,\'\"}\]]+)'''],
        "key": [r'''[\'\"](?:key|key_scale)[\'\"]\s*:\s*[\'\"]([^\'\"]+)[\'\"]'''],
        "signature": [r'''[\'\"](?:signature|timesignature|time signature|time_signature)[\'\"]\s*:\s*[\'\"]([^\'\"]+)[\'\"]'''],
    }

    result: dict[str, str] = {}
    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, s, flags=re.I)
            if not m:
                continue
            val = (m.group(1) or "").strip()
            if not val:
                continue
            if key == "genre" and "[" in s and "," in val:
                parts = [part.strip().strip("'\" ") for part in val.split(',')]
                val = ", ".join([part for part in parts if part])
            result[key] = val
            break
    return result


def parse_structured_response(raw: Any) -> dict[str, str]:
    """Parse structured caption output into a sidecar-compatible dict.

    Accepts classic ``key: value`` text, Python/JSON-like mapping text,
    or an already-decoded dict payload from HTTP providers.
    """
    if not raw:
        return {}

    if isinstance(raw, dict):
        result = _extract_structured_from_mapping(raw)
        if result:
            return result
        raw = raw.get("caption") or raw.get("text") or raw.get("raw_text") or str(raw)

    if not isinstance(raw, str):
        raw = str(raw)

    parsed_mapping = _maybe_parse_mapping_text(raw)
    if parsed_mapping:
        result = _extract_structured_from_mapping(parsed_mapping)
        if result:
            return result

    blob_result = _extract_from_mapping_blob_text(raw)
    if blob_result:
        return blob_result

    text = raw.strip()
    if not text:
        return {}

    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key in _STRUCTURED_KEYS and value and value.lower() != "n/a":
            result[key] = value

    # Fallback: if no structured keys found, treat whole text as caption
    if not result:
        result["caption"] = text

    return result
