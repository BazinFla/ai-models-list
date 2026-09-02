#!/usr/bin/env python3
"""
Check Uncurated Models
======================
Lists all active models in the Ollama catalog that do not have
an entry in `curation.json`.

Usage:
    python3 check_uncurated.py
    python3 check_uncurated.py --sort pulls --limit 30
    python3 check_uncurated.py --json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List


def load_curation(curation_path: str) -> Dict[str, Any]:
    """Loads curation.json returning a lowercased dictionary."""
    if not os.path.isfile(curation_path):
        print(f"⚠️ Curation file not found: {curation_path}")
        return {}
    try:
        with open(curation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {k.lower().strip(): v for k, v in data.items() if isinstance(v, dict)}
    except Exception as e:
        print(f"❌ Error loading curation file {curation_path}: {e}")
        return {}


def load_active_models(catalog_path: str, models_dir: str) -> List[Dict[str, Any]]:
    """Loads active (non-deprecated) models from catalog index or individual files."""
    models = []
    if os.path.isfile(catalog_path):
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_models = data.get("models", [])
                models = [m for m in all_models if not m.get("is_deprecated", False)]
                return models
        except Exception as e:
            print(f"⚠️ Warning loading catalog index: {e}, falling back to models directory.")

    if os.path.isdir(models_dir):
        for fname in sorted(os.listdir(models_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(models_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        m = json.load(f)
                        if not m.get("is_deprecated", False):
                            models.append(m)
                except Exception:
                    pass
    return models


def format_number(n: int) -> str:
    """Formats numbers with K/M suffixes."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="List active Ollama models that are missing from curation.json"
    )
    parser.add_argument(
        "--curation",
        "-c",
        default=os.path.join(script_dir, "curation.json"),
        help="Path to curation.json (default: ai-models-list/curation.json)",
    )
    parser.add_argument(
        "--catalog",
        default=os.path.join(script_dir, "ollama", "ollama-list.json"),
        help="Path to ollama-list.json index",
    )
    parser.add_argument(
        "--models-dir",
        default=os.path.join(script_dir, "ollama", "ollama-models"),
        help="Path to individual models directory",
    )
    parser.add_argument(
        "--sort",
        "-s",
        choices=["pulls", "name", "category"],
        default="pulls",
        help="Sort criterion (default: pulls)",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=0,
        help="Limit number of displayed models (0 for all)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON array of uncurated models",
    )
    args = parser.parse_args()

    curation = load_curation(args.curation)
    active_models = load_active_models(args.catalog, args.models_dir)

    if not active_models:
        print("❌ No active models found. Make sure to run `python3 scraper.py` first.")
        sys.exit(1)

    # Filter uncurated models
    uncurated = [m for m in active_models if m.get("id", "").lower() not in curation]

    # Sort
    if args.sort == "pulls":
        uncurated.sort(key=lambda m: m.get("pulls_count", 0), reverse=True)
    elif args.sort == "name":
        uncurated.sort(key=lambda m: m.get("id", "").lower())
    elif args.sort == "category":
        uncurated.sort(key=lambda m: (m.get("category", ""), -m.get("pulls_count", 0)))

    if args.limit > 0:
        displayed_models = uncurated[:args.limit]
    else:
        displayed_models = uncurated

    if args.json:
        output_data = [
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "category": m.get("category"),
                "pulls_count": m.get("pulls_count", 0),
                "description": m.get("description", ""),
                "page_url": m.get("page_url", ""),
            }
            for m in displayed_models
        ]
        print(json.dumps(output_data, indent=2, ensure_ascii=False))
        return

    total_active = len(active_models)
    total_curated = len(active_models) - len(uncurated)
    pct_curated = (total_curated / total_active * 100) if total_active > 0 else 0

    print("=" * 80)
    print("🔍 AUDIT : MODÈLES ACTIFS NON RENSEIGNÉS DANS CURATION.JSON")
    print("=" * 80)
    print(f"📊 Modèles actifs au total : {total_active}")
    print(f"✅ Modèles déjà curatés    : {total_curated} ({pct_curated:.1f}%)")
    print(f"⚠️ Modèles sans curation   : {len(uncurated)} ({100 - pct_curated:.1f}%)")
    if args.limit > 0 and len(uncurated) > args.limit:
        print(f"📋 Affichage des {args.limit} premiers modèles (tri par {args.sort}) :")
    print("-" * 80)

    print(f"{'#':<4} {'ID MODÈLE':<28} {'CATÉGORIE':<22} {'PULLS':<10} {'DESCRIPTION'}")
    print("-" * 80)

    for idx, m in enumerate(displayed_models, 1):
        m_id = m.get("id", "")
        cat = m.get("category", "General LLM")
        pulls = format_number(m.get("pulls_count", 0))
        desc = m.get("description", "").replace("\n", " ").strip()
        if len(desc) > 40:
            desc = desc[:37] + "..."
        print(f"{idx:<4} {m_id:<28} {cat:<22} {pulls:<10} {desc}")

    print("=" * 80)
    print(f"💡 Pour ajouter un modèle, éditez `{os.path.relpath(args.curation)}` puis lancez `python3 scraper.py`.")
    print("=" * 80)


if __name__ == "__main__":
    main()
