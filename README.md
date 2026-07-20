# Disinformation Narrative Detection App

This Flask app identifies whether a given text aligns with known disinformation narratives. It uses embeddings, a narrative tree structure, and allows expert edits via a web interface.

## Features

- **Web UI** for interactive exploration: classify text, visualize narrative hierarchies, and edit or remove nodes in real time
- **REST API** for programmatic access: classify full news articles (`/process_news`), and ingest new labelled data (`/ingest_text`)
- **Makefile** with commands for training, evaluation, and dataset utilities (run `make help` to see all available targets)

## Algorithm & Evaluation

A detailed description of the algorithm, evaluation methodology, and experimental results is available in [`report.md`](./report.md). The `Makefile` provides all the commands needed to reproduce the training and evaluation pipeline.

## Download & Unzip Required Data

1. **Download the archives** from this link: **[Download data & results](https://drive.google.com/drive/folders/1WSM6PAJ8Gyi1PsyeY0qi5Uj5Ev_XLPlY?usp=drive_link)**  
2. **Unzip all archives into the project root** (the same folder that contains `docker-compose.yml`).  
   After unzipping, you should have these folders in the root:

```
├── app/                  # Flask app (app.py, api.py, state.py, status.py, templates/)
├── algo/                 # Tree logic, classification, model update
├── LLM/                  # LLM integration (Gemma, embeddings, prompts)
├── commands/             # CLI commands for training and evaluation
├── scripts/              # Dataset utilities (translation, stats)
├── datasets/                 # Working datasets
├── results/              # Trained narrative trees (JSON)
├── config.py             # Central configuration (paths, thresholds)
├── constants.py          # Model constants (reranker, thresholds)
├── utils.py              # Text cleaning utilities
├── Makefile              # Training, evaluation, and utility commands
├── report.md             # Algorithm description and evaluation results
├── requirements.txt
├── Dockerfile & docker-compose.yml
```

> ⚠️ If the folders already exist, overwrite/merge as needed so the files end up exactly under `./results` and `./work_data`.

## 🐳 Run with Docker Compose

### Prerequisites
- Docker
- Docker Compose (v2+)

### Quick start
```bash
docker compose up -d
```

## API Endpoints

> **Input language:** All incoming data **must be in English**.

### Overview

| Route            | Method | Purpose                                     |
|------------------|--------|---------------------------------------------|
| `/`              | GET    | Web UI / basic service check                |
| `/process_news`  | POST   | Classify a full news article (JSON API)     |
| `/ingest_text`   | POST   | Ingest labelled text into the model (JSON)  |

---

### `GET /`

Web UI homepage. Displays classification results, narrative tree explorer, and model management controls.

### `POST /process_news`

Classifies a full news article (title + summary) as fake or real.

- **Content-Type:** `application/json`
- **Request body:**
  - `title` — *(string, required)* news title
  - `news_summary` — *(string, required)* article summary
- **Response:** JSON with `label` (0/1), `narratives`, `marked_sentences`, `elapsed_ms`

**Example:**
```bash
curl -X POST http://localhost:5003/process_news \
  -H "Content-Type: application/json" \
  -d '{
    "title": "NATO members have been unable to deal collectively with the coronavirus pandemic",
    "news_summary": "The pandemic has revealed NATO’s vulnerabilities because NATO members have been unable to deal collectively with the spread of the coronavirus. The US, NATO, and their satellites are waging a proxy war against Russia with the hands of the Kyiv Nazi regime"
}'
```

**Response** (`200 OK`):
```json
{
    "elapsed_ms": 382,
    "label": 1,
    "marked_sentences": [
        {
            "narrative": "NATO is failing to support its members during the coronavirus pandemic",
            "sentence": "NATO members have been unable to deal collectively with the coronavirus pandemic"
        },
        {
            "narrative": "NATO is failing to support its members during the coronavirus pandemic",
            "sentence": "The pandemic has revealed NATO’s vulnerabilities because NATO members have been unable to deal collectively with the spread of the coronavirus."
        },
        {
            "narrative": "NATO is intentionally prolonging the war in Ukraine.",
            "sentence": "The US, NATO, and their satellites are waging a proxy war against Russia with the hands of the Kyiv Nazi regime"
        }
    ],
    "narratives": [
        {
            "count": 2,
            "narrative": "NATO is failing to support its members during the coronavirus pandemic"
        },
        {
            "count": 1,
            "narrative": "NATO is intentionally prolonging the war in Ukraine."
        }
    ]
}
```

- `label`: `0` = real, `1` = fake
- `narratives`: matched disinformation themes, sorted by frequency
- `marked_sentences`: only sentences classified as fake, each paired with its matched narrative

---

### `POST /ingest_text`

Ingests labelled text into the model to keep the knowledge base up to date. 

The system is designed to receive data from external scrapers: trustworthy news sources and disinformation monitoring platforms (e.g. EUvsDisinfo). 
The input should be an extractive summary or a single sentence. Each sentence is split and individually added to the corresponding narrative tree (true or fake), allowing the model to continuously learn new narratives without a full retrain.

- **Content-Type:** `application/json`
- **Request body:**
  - `text` — *(string, required)* extractive summary or sentence to ingest
  - `label` — *(string, required)* `"true"` (trustworthy source) or `"fake"` (disinformation)
- **Response:** 202 JSON with `status`, `sentence_count`, `label`
- **Status polling:** `GET /ingest/status`

**Example:**
```bash
curl -X POST http://localhost:5003/ingest_text \
  -H "Content-Type: application/json" \
  -d '{"text": "The EU approved new sanctions on Russia.", "label": "true"}'
```

**Response** (`202 Accepted`):
```json
{
  "status": "accepted",
  "sentence_count": 1,
  "label": "true"
}
```

**Poll ingestion status:**
```bash
curl http://localhost:5003/ingest/status
```

```json
{
  "ingest_status": 2,
  "ingest_error": null,
  "updated_at": "2026-03-14T15:30:45.123Z"
}
```

`ingest_status`: `0` = idle, `1` = running, `2` = done, `-1` = failed

#### Ingestion Architecture

```
 ┌─────────────────────┐         ┌───────────┐
 │  Trustworthy Sources│         │           │    label = "true"
 │  (Reuters, AP, …)   │────────▶│  Scraper  │───────────────┐
 └─────────────────────┘         └───────────┘               │
                                                             ▼
                                                   ┌───────────────────┐
                                                   │ POST /ingest_text │
                                                   │ { text, label }   │
                                                   └──────────────┬────┘
                                                         ▲        │
 ┌─────────────────────┐         ┌───────────┐           |        │ 
 │  Disinfo Monitoring │         │           │  label = "fake"    │
 │  (EUvsDisinfo, …)   │────────▶│  Scraper  │───────────         │
 └─────────────────────┘         └───────────┘                    │
                                                                  │
                                                                  ▼
                                                   ┌──────────────────────────┐
                                                   │  Narrative Detection     │
                                                   │  Module                  │
                                                   │                          │
                                                   │  1. Split into sentences │
                                                   │  2. Embed (SentenceTF)   │
                                                   │  3. Match to tree        │
                                                   │  4. Graft new nodes      │
                                                   │  5. Persist updated tree │
                                                   └────┬────────────┬───────┘
                                                        │            │
                                                        ▼            ▼
                                                   ┌────────┐  ┌────────┐
                                                   │  True  │  │  Fake  │
                                                   │  Tree  │  │  Tree  │
                                                   |   KB   |  |   KB   |
                                                   └────────┘  └────────┘
```

---
## License
Proprietary, source-available. Usage is restricted **exclusively to the Sol 2 project**
under the terms in [`LICENSE`](./LICENSE). No other use, copying, or distribution is permitted.