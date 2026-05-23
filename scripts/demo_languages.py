"""Show how Task A switches between regional registers based on persona text.

Runs the same product through four personas designed to trigger
``pidgin_only``, ``yoruba``, ``igbo``, and ``hausa`` regional palettes.
Prints the detected region and the generated review for each.

Run:
    python scripts/demo_languages.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows console default encoding can't handle ₦ etc; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, ValueError):
    pass

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import LLM_MODEL, LLM_PROVIDER  # noqa: E402
from task_a.agent.graph import build_graph  # noqa: E402

PRODUCT = "Bottle of Zobo drink at a Lagos owambe, ₦500"

PERSONAS = [
    ("Pidgin (default)",
     "A nigerian student who likes affordable street drinks and writes casually"),
    ("Yoruba",
     "Yoruba professional from Ibadan, casual reviewer, values amala and ofada"),
    ("Igbo",
     "Igbo trader in Onitsha, friendly, loves nkwobi and street food"),
    ("Hausa",
     "Hausa businessman in Kano, casual reviewer, loves suya and kunu"),
]


def main() -> None:
    print(f"# provider={LLM_PROVIDER}  model={LLM_MODEL}")
    print(f"# product: {PRODUCT}\n")

    graph = build_graph()
    for label, persona in PERSONAS:
        try:
            result = graph.invoke({"persona": persona, "product": PRODUCT})
        except Exception as exc:  # noqa: BLE001
            print(f"[{label}] FAILED: {exc}")
            continue

        region = result.get("language_region", "?")
        ng_applied = bool(result.get("nigerian_context_applied", False))
        rating = result.get("rating", 0)
        review = (result.get("review") or "").strip()
        print(f"--- {label}  (detected region: {region}, ng_applied: {ng_applied}, rating: {rating}) ---")
        print(persona)
        print(f"→ {review}\n")


if __name__ == "__main__":
    main()
