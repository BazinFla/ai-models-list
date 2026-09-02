#!/usr/bin/env python3
"""
AI Models Catalog Scraper - Ollama Module
==========================================
Scrapes the complete model catalog from https://ollama.com/library
and generates individual JSON files under ollama/ollama-models/
as well as a consolidated catalog index in ollama/ollama-list.json.

Key Features:
- Preserves removed/deprecated models with `is_deprecated: true`, `status: 'deprecated'`, and `deprecated_at`.
- Supports manual curation overrides via `curation.json` (creator, publisher, icon_name, license, homepage, etc.).
- Performs intelligent in-place merge so manual edits in individual model JSON files are never overwritten.
- Multi-threaded deep scraping for exact variants, context lengths, and digests.

Usage:
    python3 scraper.py                 # Fast catalog scrape
    python3 scraper.py --deep          # Deep scrape (fetches /tags page for each model)
    python3 scraper.py --model llama3.3 # Scrapes a single model
    python3 scraper.py --curation custom_curation.json # Use custom curation file
    python3 scraper.py --include-deprecated # Include deprecated models in ollama-list.json
"""

import argparse
import concurrent.futures
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36 NeuraDex-Scraper/1.0"


def fetch_url(url: str, timeout: int = 15) -> str:
    """Performs an HTTP GET request with custom User-Agent."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_size_bytes(size_str: str) -> int:
    """Converts a size string (e.g. '4.9GB', '850MB') to bytes integer."""
    if not size_str:
        return 0
    s = size_str.strip().upper()
    val_matches = re.findall(r"[0-9\.]+", s)
    if not val_matches:
        return 0
    try:
        val = float(val_matches[0])
    except ValueError:
        return 0

    if "TB" in s or "T" in s:
        return int(val * 1024 * 1024 * 1024 * 1024)
    if "GB" in s or "G" in s:
        return int(val * 1024 * 1024 * 1024)
    if "MB" in s or "M" in s:
        return int(val * 1024 * 1024)
    if "KB" in s or "K" in s:
        return int(val * 1024)
    return int(val)


def parse_context_length(ctx_str: str) -> int:
    """Converts a context string (e.g. '128K', '32K', '8192') to number of tokens."""
    if not ctx_str:
        return 0
    s = ctx_str.strip().upper()
    val_matches = re.findall(r"[0-9\.]+", s)
    if not val_matches:
        return 0
    try:
        val = float(val_matches[0])
    except ValueError:
        return 0

    if "M" in s:
        return int(val * 1024 * 1024)
    if "K" in s:
        return int(val * 1024)
    return int(val)


def parse_pulls_count(text: str) -> int:
    """Extracts total download (pulls) count as an integer."""
    pulls_pat = r"<span[^>]*>\s*([0-9\.]+)([KMBkmb]?)\s*</span>\s*<span[^>]*>[^<]*Pulls"
    m = re.search(pulls_pat, text, re.IGNORECASE)
    if not m:
        m = re.search(r"([0-9\.]+)([KMBkmb]?)\s*Pulls", text, re.IGNORECASE)
    if m:
        num_str = m.group(1)
        unit = m.group(2).upper()
        try:
            val = float(num_str)
            if unit == "B":
                return int(round(val * 1_000_000_000))
            if unit == "M":
                return int(round(val * 1_000_000))
            if unit == "K":
                return int(round(val * 1_000))
            return int(round(val))
        except ValueError:
            pass
    return 0


def parse_tags_count(text: str) -> int:
    """Extracts available tags count as an integer."""
    tags_pat = r"<span[^>]*>\s*([0-9]+)\s*</span>\s*<span[^>]*>[^<]*Tags"
    m = re.search(tags_pat, text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return 0


def parse_date_title_to_key(title: str) -> int:
    """Converts a date string (e.g. 'Nov 30, 2024 10:34 PM UTC') to sortable integer key (YYYYMMDDHHMM)."""
    months = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    parts = title.split()
    if len(parts) >= 3:
        m_str = parts[0][:3].lower()
        month = months.get(m_str, 1)
        try:
            day = int(parts[1].rstrip(","))
            year = int(parts[2])
            hour = 0
            minute = 0
            if len(parts) >= 5 and ":" in parts[3]:
                t_parts = parts[3].split(":")
                hour = int(t_parts[0])
                minute = int(t_parts[1])
                ampm = parts[4].upper()
                if ampm == "PM" and hour < 12:
                    hour += 12
                elif ampm == "AM" and hour == 12:
                    hour = 0
            return year * 100000000 + month * 1000000 + day * 10000 + hour * 100 + minute
        except ValueError:
            pass
    return 0


def detect_capabilities(slug: str, desc: str, badges: List[str]) -> Dict[str, bool]:
    """Detects model functional capabilities from badges and description."""
    s_low = slug.lower()
    d_low = desc.lower()
    b_low = [b.lower() for b in badges]

    is_think = (
        "thinking" in b_low
        or "reasoning" in b_low
        or "r1" in s_low
        or "qwq" in s_low
        or "reasoning" in d_low
        or "think" in d_low
        or "chain-of-thought" in d_low
    )

    is_vision = (
        "vision" in b_low
        or "multimodal" in b_low
        or "vision" in s_low
        or "llava" in s_low
        or "minicpm-v" in s_low
        or "bakllava" in s_low
        or "multimodal" in d_low
        or "image" in d_low
    )

    is_audio = (
        "audio" in b_low
        or "voice" in b_low
        or "speech" in b_low
        or "whisper" in s_low
        or "audio" in s_low
        or "omni" in s_low
        or "audio" in d_low
        or "speech" in d_low
        or "voice" in d_low
    )

    is_tools = (
        "tools" in b_low
        or "function calling" in d_low
        or "tool use" in d_low
    )

    is_code = (
        "code" in b_low
        or "coding" in b_low
        or "coder" in s_low
        or "codellama" in s_low
        or "starcoder" in s_low
        or "deepseek-coder" in s_low
        or "programming" in d_low
        or "code generation" in d_low
    )

    is_embed = (
        "embedding" in b_low
        or "embed" in s_low
        or "embedding" in d_low
        or "embeddings" in d_low
        or "bge" in s_low
        or "nomic" in s_low
    )

    return {
        "think": is_think,
        "vision": is_vision,
        "audio": is_audio,
        "tools": is_tools,
        "code": is_code,
        "embedding": is_embed,
    }


def determine_category(slug: str, desc: str, badges: List[str], capabilities: Dict[str, bool]) -> str:
    """Determines the primary category of the model."""
    if capabilities["think"]:
        return "Reasoning"
    if capabilities["vision"]:
        return "Vision & Multimodal"
    if capabilities["audio"]:
        return "Audio & Multimodal"
    if capabilities["code"]:
        return "Code & Programming"
    if capabilities["embedding"]:
        return "Embeddings & Search"
    return "General LLM"


def parse_tags_page(html_content: str, slug: str) -> List[Dict[str, Any]]:
    """Parses model `/tags` page extracting all variants (local and cloud)."""
    variants = []
    containers = html_content.split('<div class="group px-4 py-3">')[1:]

    for block in containers:
        tag_m = re.search(r'href="/library/(' + re.escape(slug) + r':[^"]+)"', block)
        if not tag_m:
            continue
        full_tag = tag_m.group(1).strip()
        tag_name = full_tag.split(":", 1)[1] if ":" in full_tag else full_tag

        digest_m = re.search(r'<span class="font-mono[^"]*">\s*([0-9a-fA-F]+)\s*</span>', block)
        digest = digest_m.group(1).strip() if digest_m else ""

        mobile_line = ""
        meta_m = re.search(r'<span class="font-mono[^"]*">[\s\S]*?</span>\s*•\s*([^<]+)<', block)
        if meta_m:
            mobile_line = meta_m.group(1).strip()

        ctx_m = re.search(r'([0-9\.]+[kKmMgG]?)\s*context', mobile_line, re.IGNORECASE)
        if not ctx_m:
            ctx_m = re.search(r'<p class="col-span-2 text-neutral-500 text-\[13px\]">([^<]*[0-9]+[kKmMgG]?)</p>', block)
        ctx_str = ctx_m.group(1).strip() if ctx_m else ""

        size_m = re.search(r'([0-9\.]+\s*(?:GB|MB|KB|TB))\b', mobile_line, re.IGNORECASE)
        if not size_m:
            size_m = re.search(r'<p class="col-span-2 text-neutral-500 text-\[13px\]">([^<]*(?:GB|MB|KB|TB))</p>', block, re.IGNORECASE)

        if size_m:
            size_str = size_m.group(1).strip()
            is_cloud = False
            size_bytes = parse_size_bytes(size_str)
        else:
            usage_m = re.search(r'([A-Za-z]+\s*Usage)', mobile_line, re.IGNORECASE)
            size_str = usage_m.group(1).strip() if usage_m else "Cloud"
            is_cloud = True
            size_bytes = 0

        input_m = re.search(r'([A-Za-z\s]+)\s*input', block, re.IGNORECASE)
        input_type = input_m.group(1).strip() if input_m else "Text"

        date_m = re.search(r'([0-9]+\s+[a-zA-Z]+\s+ago|[a-zA-Z]{3}\s+[0-9]{1,2},\s+[0-9]{4})', block)
        updated_str = date_m.group(1).strip() if date_m else ""

        param_match = re.search(r'([0-9\.]+[bmkBMK])', tag_name)
        param_size = param_match.group(1).upper() if param_match else ""
        quant_match = re.search(r'(q[0-9]_[a-z0-9_]+|fp16|f16|fp32|f32)', tag_name, re.IGNORECASE)
        quant = quant_match.group(1).upper() if quant_match else ("Cloud" if is_cloud else "Default")

        variants.append({
            "tag": full_tag,
            "name": tag_name,
            "parameter_size": param_size,
            "quantization": quant,
            "size_bytes": size_bytes,
            "size_formatted": size_str,
            "context_length": parse_context_length(ctx_str),
            "context_formatted": ctx_str,
            "input_type": input_type,
            "digest": digest,
            "updated_at": updated_str,
            "is_cloud": is_cloud,
        })

    return variants


def build_fallback_variants(slug: str, badges: List[str]) -> List[Dict[str, Any]]:
    """Builds estimated variants from badges when /tags page is not fetched."""
    variants = []
    param_badges = [b for b in badges if re.match(r"^[0-9\.]+[bmkBMK]$", b, re.IGNORECASE)]

    if not param_badges:
        param_badges = ["latest"]

    for b in param_badges:
        tag = f"{slug}:{b}"
        p_size = b.upper() if b != "latest" else ""
        est_bytes = 0
        if "B" in p_size:
            try:
                num = float(p_size.rstrip("B"))
                est_bytes = int(num * 0.6 * 1024 * 1024 * 1024)
            except ValueError:
                est_bytes = 4 * 1024 * 1024 * 1024
        else:
            est_bytes = 4 * 1024 * 1024 * 1024

        variants.append({
            "tag": tag,
            "name": b,
            "parameter_size": p_size,
            "quantization": "Q4_0 (Estimated)",
            "size_bytes": est_bytes,
            "size_formatted": f"~{est_bytes // (1024*1024*1024)} GB",
            "context_length": 8192,
            "context_formatted": "8K",
            "input_type": "Text",
            "digest": "",
            "updated_at": "",
            "is_cloud": False,
        })

    return variants


def scrape_library_index() -> List[Dict[str, Any]]:
    """Downloads and extracts the model index from Ollama Library."""
    url = "https://ollama.com/library"
    print(f"📡 Fetching main index from {url}...")
    html_content = fetch_url(url)

    card_pattern = (
        r'<a\s+href="/library/([^"]+)"[\s\S]*?'
        r'<h2[^>]*>[\s\S]*?<span[^>]*>([^<]+)</span>[\s\S]*?'
        r'<p[^>]*>([^<]+)</p>([\s\S]*?)</a>'
    )
    tag_pattern = r'<span[^>]*class="[^"]*rounded-md[^"]*"[^>]*>([^<]+)</span>'
    title_pattern = r'title="([A-Za-z]{3}\s+[0-9]{1,2},\s+[0-9]{4}[^"]*)"'
    rel_updated_pattern = r'Updated\s*(?:&nbsp;)?\s*</span>\s*<span>\s*([^<]+)\s*</span>'

    models = []
    matches = list(re.finditer(card_pattern, html_content))
    print(f"✅ Found {len(matches)} models on ollama.com/library.")

    for m in matches:
        slug = m.group(1).strip()
        name = m.group(2).strip()
        desc_raw = m.group(3).strip()
        rest = m.group(4)

        if not slug or not name:
            continue

        desc = html.unescape(desc_raw)

        # Badges extraction
        badges = []
        for t_cap in re.finditer(tag_pattern, rest):
            b = t_cap.group(1).strip().lower()
            if b and b not in badges:
                badges.append(b)

        # Date extraction
        date_title_m = re.search(title_pattern, rest)
        updated_title = date_title_m.group(1).strip() if date_title_m else ""
        date_key = parse_date_title_to_key(updated_title)

        rel_date_m = re.search(rel_updated_pattern, rest)
        updated_rel = rel_date_m.group(1).strip() if rel_date_m else updated_title

        pulls_count = parse_pulls_count(rest)
        tags_count = parse_tags_count(rest)
        capabilities = detect_capabilities(slug, desc, badges)
        category = determine_category(slug, desc, badges, capabilities)
        is_cloud = "cloud" in slug.lower() or "cloud" in badges

        models.append({
            "id": slug,
            "name": name,
            "source": "ollama_library",
            "is_official": True,
            "namespace": "library",
            "description": desc,
            "category": category,
            "page_url": f"https://ollama.com/library/{slug}",
            "tags_page_url": f"https://ollama.com/library/{slug}/tags",
            "badges": badges,
            "capabilities": capabilities,
            "pulls_count": pulls_count,
            "tags_count": tags_count,
            "updated_at": updated_rel,
            "updated_full_date": updated_title,
            "updated_date_key": date_key,
            "is_cloud": is_cloud,
            "default_tag": f"{slug}:latest",
            "variants": [],
        })

    return models


def scrape_model_variants(model: Dict[str, Any]) -> Dict[str, Any]:
    """Fetches exact tag variants for a given model."""
    slug = model["id"]
    tags_url = model["tags_page_url"]
    try:
        html_tags = fetch_url(tags_url, timeout=12)
        variants = parse_tags_page(html_tags, slug)
        if variants:
            model["variants"] = variants
            model["tags_count"] = len(variants)
            latest_var = next((v for v in variants if v["name"] == "latest"), variants[0])
            model["default_tag"] = latest_var["tag"]
        else:
            model["variants"] = build_fallback_variants(slug, model["badges"])
    except Exception:
        model["variants"] = build_fallback_variants(slug, model["badges"])

    return model


# ==============================================================================
# Curation and Deprecation Management
# ==============================================================================

def load_curation_file(curation_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Loads curated metadata overrides from JSON file."""
    search_paths = []
    if curation_path:
        search_paths.append(curation_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths.append(os.path.join(script_dir, "curation.json"))
    search_paths.append(os.path.join(script_dir, "ollama", "curation.json"))

    for p in search_paths:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        print(f"📖 Loaded curation file: {p} ({len(data)} entries)")
                        return {k.lower().strip(): v for k, v in data.items() if isinstance(v, dict)}
            except Exception as e:
                print(f"⚠️ Warning: Could not parse curation file {p}: {e}")

    print("ℹ️ No curation file found. Proceeding without manual overrides.")
    return {}


def load_existing_models(models_dir: str) -> Dict[str, Dict[str, Any]]:
    """Loads all previously saved model JSON files from disk."""
    existing = {}
    if not os.path.isdir(models_dir):
        return existing

    for filename in os.listdir(models_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(models_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    m_id = data.get("id") or filename[:-5]
                    existing[m_id.lower()] = data
            except Exception as e:
                print(f"⚠️ Warning: Could not read existing model file {filepath}: {e}")

    return existing


def merge_model_metadata(
    scraped: Dict[str, Any],
    existing: Optional[Dict[str, Any]],
    curation: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merges newly scraped data with existing model JSON (preserving manual fields)
    and applies manual overrides from curation.json.
    """
    model_id = scraped["id"].lower()
    merged = dict(scraped)

    # Freshly scraped model is active and available
    merged["is_deprecated"] = False
    merged["status"] = "active"
    if "deprecated_at" in merged:
        del merged["deprecated_at"]

    # 1. Preserve custom/manual fields from previously saved file
    if existing:
        fresh_scraper_keys = {
            "id", "name", "source", "is_official", "namespace", "description",
            "category", "page_url", "tags_page_url", "badges", "capabilities",
            "pulls_count", "tags_count", "updated_at", "updated_full_date",
            "updated_date_key", "is_cloud", "default_tag", "variants",
            "is_deprecated", "status", "deprecated_at"
        }
        for k, v in existing.items():
            if k not in fresh_scraper_keys:
                merged[k] = v

    # 2. Apply curation overrides if present (explicit curation takes highest precedence)
    if model_id in curation:
        for k, v in curation[model_id].items():
            merged[k] = v

    return merged


def mark_deprecated_model(
    existing: Dict[str, Any],
    curation: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Marks an existing model as deprecated (removed from Ollama library)."""
    model = dict(existing)
    model["is_deprecated"] = True
    model["status"] = "deprecated"
    if not model.get("deprecated_at"):
        model["deprecated_at"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # Apply curation overrides if any
    model_id = model.get("id", "").lower()
    if model_id in curation:
        for k, v in curation[model_id].items():
            model[k] = v

    return model


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI Models Catalog Scraper - Ollama Module"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(os.path.dirname(__file__), "ollama"),
        help="Output directory for JSON files (default: ollama)",
    )
    parser.add_argument(
        "--curation",
        "-c",
        type=str,
        default=None,
        help="Path to curation.json overrides file (default: curation.json in script dir)",
    )
    parser.add_argument(
        "--deep",
        "-d",
        action="store_true",
        help="Deep scrape fetching exact sizes, contexts and digests from /tags pages",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=8,
        help="Number of concurrent threads for deep scraping (default: 8)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Scrape a single model by identifier (e.g. llama3.3, deepseek-r1)",
    )
    parser.add_argument(
        "--include-deprecated",
        action="store_true",
        help="Include deprecated/removed models in ollama-list.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate scrape without writing files to disk",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    models_dir = os.path.join(output_dir, "ollama-models")
    if not args.dry_run:
        os.makedirs(models_dir, exist_ok=True)

    start_time = time.time()
    print("=" * 70)
    print("🚀 AI MODELS CATALOG SCRAPER - OLLAMA MODULE")
    print(f"📁 Output root: {output_dir}")
    print(f"📂 Model files: {models_dir}")
    print(f"📋 Global index: {os.path.join(output_dir, 'ollama-list.json')}")
    print(f"🔍 Mode: {'Deep Scraping (--deep)' if args.deep else 'Fast Index'}")
    if args.dry_run:
        print("⚠️ DRY RUN MODE: No files will be modified on disk.")
    print("=" * 70)

    # 1. Load existing models from disk
    existing_models = load_existing_models(models_dir)
    print(f"📦 Found {len(existing_models)} existing model JSON files on disk.")

    # 2. Load curation overrides
    curation_data = load_curation_file(args.curation)

    # 3. Scrape main library index
    scraped_models = scrape_library_index()

    # Handle single model filter
    if args.model:
        m_id = args.model.lower().strip()
        scraped_models = [m for m in scraped_models if m["id"].lower() == m_id]
        if not scraped_models:
            print(f"❌ No model found matching '{args.model}'.")
            sys.exit(1)
        print(f"🎯 Target model: {scraped_models[0]['id']}")
        deprecated_models = []
    else:
        # Full scrape: detect models that were previously saved but are no longer in Ollama library
        scraped_ids = {m["id"].lower() for m in scraped_models}
        deprecated_ids = [m_id for m_id in existing_models if m_id not in scraped_ids]
        deprecated_models = []
        for m_id in deprecated_ids:
            dep_m = mark_deprecated_model(existing_models[m_id], curation_data)
            deprecated_models.append(dep_m)

        if deprecated_models:
            print(f"\n⚠️ Detected {len(deprecated_models)} deprecated/removed model(s) (preserved with 'deprecated_at'):")
            for dep_m in deprecated_models[:10]:
                print(f"   • {dep_m['id']} (deprecated_at: {dep_m.get('deprecated_at')})")
            if len(deprecated_models) > 10:
                print(f"   ... and {len(deprecated_models) - 10} more.")

    # 4. Fetch variants for scraped models
    if args.deep:
        total = len(scraped_models)
        print(f"\n🔄 Deep scraping tag variants for {total} models ({args.workers} workers)...")
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_model = {executor.submit(scrape_model_variants, m): m for m in scraped_models}
            for future in concurrent.futures.as_completed(future_to_model):
                completed += 1
                m = future_to_model[future]
                try:
                    res_m = future.result()
                    v_count = len(res_m.get("variants", []))
                    if completed % 10 == 0 or completed == total:
                        print(f"  [{completed}/{total}] {res_m['id']} ({v_count} variants)")
                except Exception as e:
                    print(f"  ⚠️ Error fetching {m['id']}: {e}")
    else:
        for m in scraped_models:
            m["variants"] = build_fallback_variants(m["id"], m["badges"])

    # 5. Merge metadata (in-place preservation + curation overrides)
    active_models = []
    curated_count = 0
    for m in scraped_models:
        m_id = m["id"].lower()
        existing_m = existing_models.get(m_id)
        m_merged = merge_model_metadata(m, existing_m, curation_data)
        if m_id in curation_data:
            curated_count += 1
        active_models.append(m_merged)

    # 6. Write individual JSON files
    if not args.dry_run:
        print(f"\n💾 Writing {len(active_models)} active model files to {models_dir}...")
        for m in active_models:
            model_file = os.path.join(models_dir, f"{m['id']}.json")
            with open(model_file, "w", encoding="utf-8") as f:
                json.dump(m, f, indent=2, ensure_ascii=False)

        if deprecated_models:
            print(f"💾 Updating {len(deprecated_models)} deprecated model files in {models_dir}...")
            for m in deprecated_models:
                model_file = os.path.join(models_dir, f"{m['id']}.json")
                with open(model_file, "w", encoding="utf-8") as f:
                    json.dump(m, f, indent=2, ensure_ascii=False)

    # 7. Write consolidated catalog index
    catalog_path = os.path.join(output_dir, "ollama-list.json")
    catalog_models = active_models + (deprecated_models if args.include_deprecated else [])

    if not args.dry_run and not args.model:
        with open(catalog_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": "1.1",
                    "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    "total_models": len(catalog_models),
                    "total_active_models": len(active_models),
                    "total_deprecated_models": len(deprecated_models),
                    "models": catalog_models,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"✨ SCRAPING COMPLETED IN {elapsed:.2f}s!")
    print(f"🟢 Active models: {len(active_models)}")
    print(f"🔴 Deprecated models preserved: {len(deprecated_models)}")
    print(f"🎨 Curated models enriched: {curated_count}")
    if not args.dry_run:
        print(f"📄 Model files: {models_dir}")
        if not args.model:
            print(f"📦 Catalog index: {catalog_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
