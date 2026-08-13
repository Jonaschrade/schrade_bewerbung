"""
Persona sampling from gss_personas.json.

The JSON file contains survey records for US respondents (General Social
Survey) as a list of dicts. Each dict maps survey question text to the
respondent's answer. Immigration-attitude fields and technical classification
codes have been removed during export (see notebooks/script.ipynb) to prevent
stance leakage into persona descriptions.

Public API:
    sample_personas(n, llm) -> list[dict]
        Randomly draws n survey records, derives an American first name and a
        concise persona description for each via the LLM, and returns
        a list of {"name": str, "persona": str} dicts ready for Agent
        construction.
"""

import json
import random
import re
from pathlib import Path

_POOL_PATH = Path(__file__).parent.parent / "data" / "gss_personas.json"


def _load_pool() -> list[dict]:
    """Load persona records from disk. Each record is a plain dict."""
    with open(_POOL_PATH, encoding="utf-8") as f:
        return json.load(f)


def _clean_value(value: str) -> str:
    """Strip leading numeric answer codes from survey values.

    Survey answers may be prefixed with codes like '(2) ' or '(11) ' that encode
    the response scale position. These are not meaningful for the LLM prompt.
    """
    return re.sub(r"^\(\d+\)\s*", "", value).strip()


def _format_record(record: dict) -> str:
    """Format a survey record dict as a human-readable attribute block."""
    return "\n".join(f"- {q}: {_clean_value(str(a))}" for q, a in record.items())


def _expand(record: dict, llm) -> dict:
    """Derive a name and persona description from a survey record via the LLM.

    Args:
        record: A single survey record dict with English question/answer pairs.
        llm:    An instantiated LangChain LLM.

    Returns:
        {"name": str, "persona": str}: falls back to generic values if the
        LLM response cannot be parsed.
    """
    attributes = _format_record(record)

    raw = llm.invoke(
        f"You are given the survey profile of a US resident:\n\n"
        f"{attributes}\n\n"
        f"Your task:\n"
        f"1. Invent a fitting American first name for this person.\n"
        f"2. Write a 2-3-sentence persona description in the second person "
        f"(start with 'You are ...') that reflects the person's life situation, professional "
        f"background, social environment, and communication style.\n\n"
        f"Reply exclusively in this format:\n"
        f"NAME: <first name>\n"
        f"PERSONA: <description>"
    ).strip()

    name, persona = None, None
    for line in raw.splitlines():
        if line.upper().startswith("NAME:"):
            name = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PERSONA:"):
            persona = line.split(":", 1)[1].strip()

    return {
        "name": name or "Alex",
        "persona": persona or "You are a US resident with your own opinions.",
    }


def sample_personas(n: int, llm) -> list[dict]:
    """Sample n personas at random and expand each with the LLM.

    Names are guaranteed to be unique across all sampled agents. If the LLM
    assigns a name that is already taken, a new record is drawn and expanded
    until a distinct name is produced.

    Args:
        n:   Number of personas to sample.
        llm: An instantiated LangChain LLM used to expand each record.

    Returns:
        List of n {"name": str, "persona": str} dicts with unique names.
    """
    pool = _load_pool()
    personas = []
    used_names: set[str] = set()

    remaining = random.sample(pool, len(pool))

    for record in remaining:
        if len(personas) == n:
            break
        p = _expand(record, llm)
        if p["name"].lower() in used_names:
            continue
        used_names.add(p["name"].lower())
        print(f"  Sampled persona: {p['name']}")
        personas.append(p)

    return personas
