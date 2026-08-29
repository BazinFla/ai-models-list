# AI Models Catalog (`ai-models-list`)

Automated, structured JSON catalog for open-source AI models, built for desktop and server LLM managers.

The catalog is automatically synchronized every 6 hours via GitHub Actions.

---

## 📂 Repository Structure

```
ai-models-list/
├── .github/workflows/
│   └── sync-catalog.yml        # Automated sync workflow (runs every 6 hours)
├── ollama/
│   ├── ollama-list.json        # Consolidated Ollama catalog index
│   └── ollama-models/          # 236+ individual model JSON specifications
├── scraper.py                  # Python 3 catalog scraper
├── requirements.txt            # Zero-dependency specification
├── LICENSE                     # MIT License
└── README.md
```

---

## 🌐 CDN Endpoints

The catalog can be consumed directly via **jsDelivr CDN** (with edge caching) or raw GitHub:

### Ollama Models

| Resource | jsDelivr CDN Endpoint |
| :--- | :--- |
| **Catalog Index** | `https://cdn.jsdelivr.net/gh/BazinFla/ai-models-list@main/ollama/ollama-list.json` |
| **Single Model** | `https://cdn.jsdelivr.net/gh/BazinFla/ai-models-list@main/ollama/ollama-models/{model_id}.json` |

---

## 📋 Schema Example (`ollama/ollama-models/deepseek-r1.json`)

```json
{
  "id": "deepseek-r1",
  "name": "deepseek-r1",
  "source": "ollama_library",
  "is_official": true,
  "namespace": "library",
  "description": "DeepSeek-R1 is a family of open reasoning models with performance approaching that of leading models, such as O3 and Gemini 2.5 Pro.",
  "category": "Reasoning",
  "page_url": "https://ollama.com/library/deepseek-r1",
  "tags_page_url": "https://ollama.com/library/deepseek-r1/tags",
  "badges": [
    "tools",
    "thinking",
    "1.5b",
    "7b",
    "8b",
    "14b",
    "32b",
    "70b",
    "671b"
  ],
  "capabilities": {
    "think": true,
    "vision": false,
    "audio": false,
    "tools": true,
    "code": false,
    "embedding": false
  },
  "pulls_count": 91900000,
  "tags_count": 35,
  "updated_at": "Jul 2, 2025 6:09 AM UTC",
  "updated_full_date": "Jul 2, 2025 6:09 AM UTC",
  "updated_date_key": 202507020609,
  "is_cloud": false,
  "default_tag": "deepseek-r1:latest",
  "variants": [
    {
      "tag": "deepseek-r1:latest",
      "name": "latest",
      "parameter_size": "",
      "quantization": "Default",
      "size_bytes": 5583457484,
      "size_formatted": "5.2GB",
      "context_length": 131072,
      "context_formatted": "128K",
      "input_type": "Text",
      "digest": "6995872bfe4c",
      "updated_at": "1 year ago",
      "is_cloud": false
    }
  ]
}
```

---

## 🛠️ Local Scraping

```bash
# Full deep scrape (all model tags in parallel)
python3 scraper.py --deep --workers 16 --output-dir ollama

# Single model update
python3 scraper.py --model deepseek-r1 --deep --output-dir ollama
```

---

## 📜 License

Distributed under the [MIT License](LICENSE). Copyright © 2026 BazinFla.
