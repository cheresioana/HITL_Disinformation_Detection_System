# Self-Explanatory Disinformation Detection System with Human-in-the-loop

Research code and paper for a self-explanatory disinformation-detection system. The system embeds statements into a **narrative tree**, classifies text against known true and fake narratives, explains every decision by the narrative it matched, and lets human experts correct the model through a web interface. This repository holds the full pipeline (training, evaluation, experiments), the interactive web app with its REST API, and the paper itself.

## Features

- **Narrative-tree pipeline** for building true/fake narrative trees from labelled data, classifying statements and full articles, and comparing against classical baselines (SVM, LR, GradientBoosting, DecisionTree, KNN). Driven through the `Makefile` (`make help`).
- **Human-in-the-loop** review: experts inspect the learned narratives and add or remove nodes so the knowledge base stays correct without a full retrain.
- **Web UI** for interactive exploration: classify text, visualize narrative hierarchies, and edit or remove nodes in real time.
- **REST API** for programmatic access: classify full news articles (`/process_news`) and ingest new labelled data (`/ingest_text`).

## Setup

Two large folders are **not** included and must be added before running anything: the **datasets** and the **results** (the trained narrative trees).

**1. Add the dataset** at the project root as `datasets/`:

```
datasets/
├── mindbugs_updated/           # primary train/eval split (train.csv, evaluation.csv, ...)
├── covid/                      # COVID-19 dataset (train/eval/test.csv)
├── liar/                       # LIAR dataset (train/valid/test.tsv)
├── fake_news_net/              # FakeNewsNet split
├── complete_news_data/         # full-article eval (complete_news_test_df.csv)
└── tvr_info/json_translations/ # Romanian in-the-wild articles
```

Download it from Kaggle: **[hitl-paper-mindbugs-dataset](https://www.kaggle.com/datasets/ioanacheres/hitl-paper-mindbugs-dataset)**

**2. Add the results (trained trees)** at the project root as `results/`:

```
results/
├── narrative_mbd_new/
│   ├── true/results/full_result_0.5.json    # true-narrative tree
│   └── false/results/full_result_0.5.json   # fake-narrative tree
└── backup/                                  # restore point for the web app
```

Download the results from: **[trained trees & results](https://drive.google.com/drive/folders/1iUAALLYmXeMUeEZ1d1Agb_5ZoX6MD9Ts)**

Paths are defined in [`config.py`](./config.py) (`DATASETS_DIR`, `RESULTS_DIR`, `ACTIVE_THRESHOLD`). To keep the folders elsewhere, edit those constants.

**3. Install dependencies** (Python 3):

```bash
make install          # or: pip install -r requirements.txt
```

**4. Run Ollama with the Gemma 3 models.** The pipeline needs an [Ollama](https://ollama.com) server with **both** `gemma3:12b` and `gemma3:27b` pulled: `gemma3:27b` handles generation and entailment, `gemma3:12b` handles extraction.

```bash
ollama pull gemma3:27b
ollama pull gemma3:12b
```

Then point the code at that server:

```bash
export OLLAMA_BASE_URL=http://your-ollama-host:11434/
```

## Run the pipeline

```bash
make train                                                   # build narrative trees (English, MindBugs)
make eval-val FILE=datasets/mindbugs_updated/evaluation.csv  # evaluate statements
make eval-news                                               # evaluate full news articles
make eval-sota                                               # classical baselines on all datasets
make dataset-stats                                           # label counts per dataset
make help                                                    # list every target
```

> **`eval-sota` needs all four datasets present**, each with the same train/val/test layout under `datasets/`:
> - `covid/` — `train.csv`, `eval.csv`, `test.csv`
> - `liar/` — `train.tsv`, `valid.tsv`, `test.tsv`
> - `fake_news_net/` — `train_fakenewsnet.csv`, `val_fakenewsnet.csv`, `test_fakenewsnet.csv`
> - `mindbugs_updated/` — `train.csv`, `validation_df.csv`, `evaluation.csv`
>
> A missing file stops the run with e.g. `FileNotFoundError: datasets/covid/train.csv`. To smoke-test without the full data, point it at the bundled 6-row set: `DATASETS_DIR=datasets_test make eval-sota`.

A prose description of the algorithm and the full evaluation results is in [`report.md`](./report.md).

## 🐳 Run with Docker Compose

### Prerequisites
- Docker
- Docker Compose (v2+)
- **~100 GB free in Docker's disk** for the models

Check, and reclaim if needed, **before** starting:

```bash
docker system df          # how much of Docker's disk is used / reclaimable
docker system prune -af   # frees unused images + build cache (keeps your Ollama models; do NOT add --volumes)
```

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