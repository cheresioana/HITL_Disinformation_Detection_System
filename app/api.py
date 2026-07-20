"""
API Blueprint: text and news classification endpoints.
"""

import logging
import time
import sys
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Blueprint, request, redirect, url_for, jsonify, current_app
from nltk.tokenize import sent_tokenize

from config import NARRATIVE_DIR, UPLOAD_DIR
from LLM.orchestrator import fetch_embedding
from utils import clean_text
from algo.get_label_dual import get_label_dual_v3
from algo.parse_news import get_news_label
from algo.update_model import update_trees_with_new_data

from status import set_ingest_status
from state import (
    get_fake_matches, get_real_matches, get_intro_texts,
    reload_trees, update_intro_text_entry, add_intro_text,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

# Background pool for /process_text classifications. Threads share the
# in-memory tree state with the main request thread (the process pool
# used for ingest would fork before the trees are loaded).
_analysis_executor = ThreadPoolExecutor(max_workers=2)


def _run_analysis(user_text):
    """Classify *user_text* against the current trees and update its card.

    Runs on a background thread so the Analyse request returns immediately.
    """
    try:
        row = {
            'text': user_text,
            'embedding': fetch_embedding(clean_text(user_text)),
        }
        start_time = time.time()
        label, node, _, _, explanation = get_label_dual_v3(
            row, get_fake_matches(), get_real_matches()
        )
        elapsed_time = time.time() - start_time
        # Coerce to plain Python int so jsonify never chokes on np.int64
        label = int(label) if label is not None else None
        logger.info(
            "Classification took %.4fs (label=%s, narrative=%s)",
            elapsed_time, label, node.text if node else "",
        )
        ok = update_intro_text_entry(
            user_text, label=label, node=node, ingest_status='idle',
            explanation=explanation,
        )
        if not ok:
            logger.warning(
                "Analysis finished but intro_texts entry not found for %r",
                user_text[:120],
            )
    except Exception as exc:
        logger.exception("Analysis failed for %r", user_text[:80])
        update_intro_text_entry(user_text, ingest_status='failed')


@api_bp.route('/process_text', methods=['POST'])
def process_text():
    """
    Classify a single text snippet against narrative trees asynchronously.

    Expects:
        Form field 'user_text'.

    Returns:
        Redirect to home — the card lands in the correction set in the
        ``processing`` state and the status poller flips it to its final
        state once classification finishes.
    """
    user_text = request.form.get('user_text') or ''
    logger.info("Received text: %s", user_text)
    if not user_text.strip():
        return redirect(url_for('home'))

    add_intro_text({
        'label': None,
        'node': None,
        'user_text': user_text,
        'source': 'manual',
        'ingest_status': 'processing',
    })

    _analysis_executor.submit(_run_analysis, user_text)
    return redirect(url_for('home'))


@api_bp.route('/process_news', methods=['POST'])
def process_news():
    """
    Classify a full news article (title + summary) as fake or real.

    Expects:
        JSON body with 'title' and 'news_summary' fields.

    Returns:
        JSON with label, narratives, and elapsed_ms.
    """
    fake_matches = get_fake_matches()
    real_matches = get_real_matches()

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    title = data.get('title')
    news_summary = data.get('news_summary')
    logger.info("Processing news — title: %s", title)
    logger.debug("Summary: %s", news_summary)
    start_time = time.time()
    try:
        label, narratives, marked_sentences = get_news_label(
            title, news_summary, fake_matches, real_matches, 0
        )
    except Exception as e:
        return jsonify({"error": "Failed to compute label", "detail": str(e)}), 500

    elapsed_ms = int((time.time() - start_time) * 1000)
    return jsonify({
        "label": label,
        "narratives": narratives,
        "marked_sentences": marked_sentences,
        "elapsed_ms": elapsed_ms
    }), 200


# ---------------------------------------------------------------------------
# Text ingestion (async model update from raw text)
# ---------------------------------------------------------------------------

LABEL_MAP = {"fake": 1, "true": 0}

def _ingest_job(csv_path):
    """Background job: read the ingested CSV and update the trees."""
    new_df = pd.read_csv(csv_path, skipinitialspace=True)
    new_df.columns = new_df.columns.str.strip()
    update_trees_with_new_data(new_df, folder=str(NARRATIVE_DIR) + '/')
    return 1

def _on_done_ingest(fut):
    try:
        fut.result()
        set_ingest_status(2, None)
        reload_trees()
        logger.info("Ingestion update completed")
    except Exception as e:
        set_ingest_status(-1, str(e))
        logger.error("Ingestion update failed: %s", e)

@api_bp.get("/cards/status")
def cards_status():
    """Return per-card ingest state for the correction set.

    Used by the UI to poll progress of /mark_item_as_fake without a full reload.
    ``node_text`` is the parent narrative text the UI needs to wire up the
    "View tree" button once a card transitions to ``done``.
    """
    payload = []
    for e in get_intro_texts():
        node = e.get("node")
        raw_label = e.get("label")
        label = int(raw_label) if raw_label is not None else None
        explanation = e.get("explanation")
        explanation_reason = explanation.get("reason") if isinstance(explanation, dict) else None
        payload.append({
            "user_text": e.get("user_text"),
            "status": e.get("ingest_status", "idle"),
            "label": label,
            "is_wrong_prediction": bool(e.get("is_wrong_prediction", False)),
            "source": e.get("source", "manual"),
            "node_text": node.text if node is not None else None,
            "explanation_reason": explanation_reason,
        })
    return jsonify(payload)


@api_bp.post("/ingest_text")
def ingest_text():
    """Ingest a text with a label.

    Splits the text into sentences, builds a CSV, and triggers
    an async incremental model update.

    Expects JSON: {"text": "...", "label": "fake" | "true"}
    Returns 202 with sentence count on success.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    text = (data.get("text") or "").strip()
    label = (data.get("label") or "").strip().lower()

    if not text:
        return jsonify({"error": "Field 'text' is required and cannot be empty"}), 400
    if label not in LABEL_MAP:
        return jsonify({"error": f"Field 'label' must be 'true' or 'fake', got '{label}'"}), 400

    label_numeric = LABEL_MAP[label]

    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return jsonify({"error": "No sentences could be extracted from the text"}), 400

    # Per-call CSV path so concurrent /ingest_text requests don't overwrite
    # each other before the single-worker executor picks them up. The done
    # callback unlinks the file when the job completes.
    df = pd.DataFrame({"text": sentences, "label": [label_numeric] * len(sentences)})
    ingest_path = UPLOAD_DIR / f"ingest_dataset_{uuid.uuid4().hex}.csv"
    df.to_csv(ingest_path, index=False)

    set_ingest_status(1, "Work in progress")
    executor = current_app.config['EXECUTOR']
    future = executor.submit(_ingest_job, ingest_path)

    def _cleanup_and_finalize(fut):
        try:
            _on_done_ingest(fut)
        finally:
            try:
                ingest_path.unlink(missing_ok=True)
            except Exception as cleanup_exc:
                logger.warning("Could not remove temp ingest CSV %s: %s",
                               ingest_path, cleanup_exc)
    future.add_done_callback(_cleanup_and_finalize)

    logger.info("Ingestion accepted: %d sentences, label=%s", len(sentences), label)
    return jsonify({
        "status": "accepted",
        "sentence_count": len(sentences),
        "label": label,
    }), 202
