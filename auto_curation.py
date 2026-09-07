#!/usr/bin/env python3
"""
Auto-Curation Generator
========================
Generates a consolidated `new-curation.json` template containing ALL available
Ollama models in alphabetical order.

- Preserves existing curated values (creator, publisher, family, etc.) from `curation.json`.
- Automatically populates uncurated models with empty placeholder fields ready to be filled.
- Can read local catalog files (`ollama-list.json` / `ollama-models/`) or scrape live from ollama.com.

Usage:
    python3 auto_curation.py
    python3 auto_curation.py --output new-curation.json
    python3 auto_curation.py --live-scrape
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from typing import Any, Dict, List, Set


DEFAULT_TEMPLATE_KEYS = ["creator", "publisher", "family"]


def load_curation(curation_path: str) -> Dict[str, Dict[str, Any]]:
    """Loads existing curation.json returning a dictionary."""
    if not os.path.isfile(curation_path):
        print(f"ℹ️ Curation file not found: {curation_path} (starting from empty)")
        return {}
    try:
        with open(curation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {k.strip(): v for k, v in data.items() if isinstance(v, dict)}
    except Exception as e:
        print(f"⚠️ Error loading curation file {curation_path}: {e}")
    return {}


def load_local_models(catalog_path: str, models_dir: str) -> Set[str]:
    """Loads all model IDs from local catalog index or individual model files."""
    model_ids: Set[str] = set()

    # 1. From catalog index
    if os.path.isfile(catalog_path):
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for m in data.get("models", []):
                    m_id = m.get("id")
                    if m_id:
                        model_ids.add(m_id.strip())
        except Exception as e:
            print(f"⚠️ Warning loading {catalog_path}: {e}")

    # 2. From individual model files
    if os.path.isdir(models_dir):
        for fname in os.listdir(models_dir):
            if fname.endswith(".json"):
                m_id = fname[:-5]
                model_ids.add(m_id.strip())

    return model_ids


def scrape_live_library_ids() -> Set[str]:
    """Scrapes all model IDs directly from https://ollama.com/library."""
    url = "https://ollama.com/library"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NeuraDex-Scraper/1.0"}
    req = urllib.request.Request(url, headers=headers)
    model_ids: Set[str] = set()

    try:
        print(f"🌐 Fetching live models from {url}...")
        with urllib.request.urlopen(req, timeout=15) as response:
            html_text = response.read().decode("utf-8", errors="ignore")

        # Extract model IDs from href attributes like href="/library/llama3.3" or href="/llama3.3"
        links = re.findall(r'href=["\'](?:/library/)?([a-zA-Z0-9_\-\.]+?)["\']', html_text)
        excluded = {
            "", "library", "search", "blog", "download", "models", "api", "login",
            "signup", "pricing", "terms", "privacy", "github", "discord", "twitter"
        }
        for link in links:
            clean = link.strip()
            if clean and clean not in excluded and not clean.startswith(("http", "#", "?", "mailto:")):
                model_ids.add(clean)

        print(f"✅ Found {len(model_ids)} models from live scrape.")
    except Exception as e:
        print(f"⚠️ Live scrape failed: {e}. Falling back to local catalog.")

    return model_ids


def create_auto_curation(
    curation_path: str,
    catalog_path: str,
    models_dir: str,
    output_path: str,
    live_scrape: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Generates the consolidated alphabetically-sorted curation dictionary."""
    existing_curation = load_curation(curation_path)
    model_ids: Set[str] = set()

    if live_scrape:
        model_ids = scrape_live_library_ids()

    # If live scrape returned nothing or wasn't requested, load local models
    if not model_ids:
        model_ids = load_local_models(catalog_path, models_dir)

    # Also include models already in existing curation even if not currently in catalog
    for m_id in existing_curation.keys():
        model_ids.add(m_id)

    if not model_ids:
        print("❌ No models found to curate.")
        sys.exit(1)

    result: Dict[str, Dict[str, Any]] = {}
    already_curated_count = 0
    new_placeholders_count = 0

    # Sort all model IDs alphabetically (case-insensitive)
    for m_id in sorted(model_ids, key=lambda s: s.lower()):
        # Check if existing curation has this ID (exact or case-insensitive)
        existing_entry = existing_curation.get(m_id)
        if not existing_entry:
            for k, v in existing_curation.items():
                if k.lower() == m_id.lower():
                    existing_entry = v
                    break

        if existing_entry:
            # Reorder keys starting with default template keys, then any custom keys
            ordered_meta: Dict[str, Any] = {}
            for k in DEFAULT_TEMPLATE_KEYS:
                if k in existing_entry:
                    ordered_meta[k] = existing_entry[k]
                else:
                    ordered_meta[k] = ""
            for k, v in existing_entry.items():
                if k not in ordered_meta:
                    ordered_meta[k] = v
            result[m_id] = ordered_meta
            already_curated_count += 1
        else:
            # New empty placeholder template
            result[m_id] = {k: "" for k in DEFAULT_TEMPLATE_KEYS}
            new_placeholders_count += 1

    # Write output JSON file
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return {
        "total": len(result),
        "curated": already_curated_count,
        "placeholders": new_placeholders_count,
        "output_path": os.path.abspath(output_path),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_curation = os.path.join(script_dir, "curation.json")
    default_catalog = os.path.join(script_dir, "ollama", "ollama-list.json")
    default_models_dir = os.path.join(script_dir, "ollama", "ollama-models")
    default_output = os.path.join(script_dir, "new-curation.json")

    parser = argparse.ArgumentParser(
        description="Generate an alphabetical new-curation.json containing all Ollama models ready to be filled."
    )
    parser.add_argument(
        "--curation",
        "-c",
        default=default_curation,
        help="Path to current curation.json (default: ai-models-list/curation.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=default_output,
        help="Path for generated output file (default: ai-models-list/new-curation.json)",
    )
    parser.add_argument(
        "--catalog",
        default=default_catalog,
        help="Path to ollama-list.json catalog (default: ai-models-list/ollama/ollama-list.json)",
    )
    parser.add_argument(
        "--models-dir",
        default=default_models_dir,
        help="Path to ollama-models folder (default: ai-models-list/ollama/ollama-models)",
    )
    parser.add_argument(
        "--live-scrape",
        "-s",
        action="store_true",
        help="Scrape live model list from https://ollama.com/library before building",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🪄 AUTO-CURATION TEMPLATE GENERATOR")
    print("=" * 70)
    print(f"📖 Source curation: {args.curation}")
    print(f"📦 Catalog source:  {args.catalog}")
    print(f"💾 Output file:     {args.output}")
    print("-" * 70)

    stats = create_auto_curation(
        curation_path=args.curation,
        catalog_path=args.catalog,
        models_dir=args.models_dir,
        output_path=args.output,
        live_scrape=args.live_scrape,
    )

    print("✨ GENERATION COMPLETED SUCCESSFULLY!")
    print(f"📊 Total models in template: {stats['total']}")
    print(f"✅ Pre-filled from curation: {stats['curated']}")
    print(f"📝 New empty placeholders:  {stats['placeholders']}")
    print(f"📁 Output file saved at:     {stats['output_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
