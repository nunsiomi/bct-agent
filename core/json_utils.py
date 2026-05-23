"""Tolerant JSON extraction + a tiny logging helper.

LLMs sometimes wrap JSON in markdown fences or surrounding prose. This module
strips those wrappers and returns parsed JSON, raising `ValueError` only when
no parseable JSON object/array can be found.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any


def parse_json_block(text: str) -> Any:
    """Extract a JSON object or list from an LLM response.

    Tolerates: raw JSON, ```json ... ``` fences, and prose around JSON.
    """
    if text is None:
        raise ValueError("empty text")
    s = text.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            s = candidate

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = s.find(open_ch)
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        snippet = s[start : i + 1]
                        try:
                            return json.loads(snippet)
                        except json.JSONDecodeError:
                            break
            start = s.find(open_ch, start + 1)

    raise ValueError(f"could not parse JSON from response: {text[:200]!r}")


def warn(msg: str) -> None:
    """Single-line warning to stderr, used by node fallbacks."""
    print(f"[bct-agent] {msg}", file=sys.stderr)
