#!/usr/bin/env python3
"""
Check Missing Icons
===================
Lists all models defined in `curation.json` whose icon files
(monochrome `.svg` and/or color `-color.svg`) are missing in NeuraDex.

Usage:
    python3 check_missing_icons.py
    python3 check_missing_icons.py --icons-dir ../neuradex/data/icons/models
    python3 check_missing_icons.py --json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Set


def load_curation(curation_path: str) -> Dict[str, Any]:
    """Loads curation.json returning a dictionary."""
    if not os.path.isfile(curation_path):
        print(f"❌ Curation file not found: {curation_path}")
        sys.exit(1)
    try:
        with open(curation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {k.strip(): v for k, v in data.items() if isinstance(v, dict)}
    except Exception as e:
        print(f"❌ Error loading curation file {curation_path}: {e}")
        sys.exit(1)


def get_available_icons(icons_dir: str) -> Set[str]:
    """Lists all available SVG icon filenames in the icons directory."""
    if not os.path.isdir(icons_dir):
        return set()
    return {f for f in os.listdir(icons_dir) if f.endswith(".svg")}


def slugify(text: str) -> str:
    """Standardizes text to lowercase hyphenated slug."""
    return text.strip().lower().replace(" ", "-")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_icons_dir = os.path.abspath(
        os.path.join(script_dir, "..", "neuradex", "data", "icons", "models")
    )
    default_curation_path = os.path.join(script_dir, "curation.json")

    parser = argparse.ArgumentParser(
        description="List models in curation.json that are missing SVG icons in NeuraDex"
    )
    parser.add_argument(
        "--curation",
        "-c",
        default=default_curation_path,
        help="Path to curation.json (default: ai-models-list/curation.json)",
    )
    parser.add_argument(
        "--icons-dir",
        "-i",
        default=default_icons_dir,
        help="Path to NeuraDex models icon folder (default: neuradex/data/icons/models)",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only display models that are missing at least one icon",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON analysis",
    )
    args = parser.parse_args()

    curation = load_curation(args.curation)
    available_icons = get_available_icons(args.icons_dir)

    results = []
    missing_both = []
    missing_mono = []
    missing_color = []
    complete = []

    # Map each unique icon_name to models using it
    unique_icons = set()

    for model_id, meta in sorted(curation.items()):
        creator = meta.get("creator", "").strip()
        family = meta.get("family", "").strip()

        # Construct icon name from $CREATOR-$FAMILY (supports clean, compact, and first-word slug variants)
        if creator and family:
            clean_c = slugify(creator)
            compact_c = clean_c.replace("-", "")
            first_c = clean_c.split("-")[0]
            f_slug = slugify(family)

            candidates = [f"{clean_c}-{f_slug}"]
            if compact_c != clean_c:
                candidates.append(f"{compact_c}-{f_slug}")
            if first_c != clean_c:
                candidates.append(f"{first_c}-{f_slug}")

            # Select the matching candidate if already existing in available icons, otherwise prefer first_c-f_slug
            matched = next(
                (c for c in candidates if f"{c}.svg" in available_icons or f"{c}-color.svg" in available_icons),
                None,
            )
            icon_name = matched if matched else f"{first_c}-{f_slug}"
        elif meta.get("icon_name"):
            icon_name = meta.get("icon_name", "").strip()
        else:
            icon_name = ""

        if not icon_name:
            mono_file = "default.svg"
            color_file = "default-color.svg"
            has_icon_name = False
        else:
            mono_file = f"{icon_name}.svg"
            color_file = f"{icon_name}-color.svg"
            has_icon_name = True
            unique_icons.add(icon_name)

        has_mono = mono_file in available_icons
        has_color = color_file in available_icons

        entry = {
            "model_id": model_id,
            "creator": creator or "Unknown",
            "family": family or "-",
            "icon_name": icon_name,
            "mono_file": mono_file,
            "has_mono": has_mono,
            "color_file": color_file,
            "has_color": has_color,
            "is_complete": has_mono and has_color,
        }
        results.append(entry)

        if not has_mono and not has_color:
            missing_both.append(entry)
        elif not has_mono:
            missing_mono.append(entry)
        elif not has_color:
            missing_color.append(entry)
        else:
            complete.append(entry)

    if args.json:
        output_data = {
            "icons_dir": args.icons_dir,
            "total_curated_models": len(curation),
            "unique_icon_names": len(unique_icons),
            "complete_count": len(complete),
            "missing_both_count": len(missing_both),
            "missing_mono_count": len(missing_mono),
            "missing_color_count": len(missing_color),
            "models": results if not args.missing_only else [r for r in results if not r["is_complete"]],
        }
        print(json.dumps(output_data, indent=2, ensure_ascii=False))
        return

    print("=" * 88)
    print("🎨 AUDIT : DISPONIBILITÉ DES LOGOS DANS NEURADEX")
    print("=" * 88)
    print(f"📁 Répertoire des icônes : {args.icons_dir}")
    print(f"📦 Fichiers SVG trouvés  : {len(available_icons)}")
    print(f"📋 Modèles curatés       : {len(curation)} (représentant {len(unique_icons)} familles de logos uniques)")
    print("-" * 88)
    print(f"🟢 Logos complets (Mono + Couleur) : {len(complete)} / {len(curation)}")
    print(f"🟡 Logos couleur manquants          : {len(missing_color)} / {len(curation)}")
    print(f"🟡 Logos monochrome manquants       : {len(missing_mono)} / {len(curation)}")
    print(f"🔴 Aucun logo trouvé (Manque 2/2)   : {len(missing_both)} / {len(curation)}")
    print("=" * 88)

    displayed_entries = [r for r in results if not r["is_complete"]] if args.missing_only else results

    print(f"{'ID MODÈLE':<20} {'CRÉATEUR':<15} {'FAMILLE':<14} {'NOM FICHIER':<22} {'MONO':<10} {'COULEUR':<10}")
    print("-" * 88)

    for r in displayed_entries:
        mono_status = "✅ Présent" if r["has_mono"] else "❌ Manque"
        color_status = "✅ Présent" if r["has_color"] else "❌ Manque"
        print(f"{r['model_id']:<20} {r['creator']:<15} {r['family']:<14} {r['icon_name']:<22} {mono_status:<10} {color_status:<10}")

    print("=" * 85)

    # List unique missing files to create
    missing_files = set()
    for r in results:
        if not r["has_mono"]:
            missing_files.add(r["mono_file"])
        if not r["has_color"]:
            missing_files.add(r["color_file"])

    if missing_files:
        print(f"🛠️  Fichiers SVG manquants à ajouter dans `{os.path.relpath(args.icons_dir)}` ({len(missing_files)} au total) :")
        for f in sorted(missing_files):
            print(f"   • {f}")
        print("=" * 85)
    else:
        print("✨ Tous les modèles de curation.json ont leurs logos disponibles dans NeuraDex !")
        print("=" * 85)


if __name__ == "__main__":
    main()
