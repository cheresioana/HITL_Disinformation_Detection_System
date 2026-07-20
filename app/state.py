"""
Shared mutable application state.

Holds the loaded narrative trees and UI state. All access goes through
getter / setter / reload helpers so that callers do not need to touch
module-level variables directly.
"""

import json
import logging
import os
import re
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

# Older CSVs generated before reasons were trimmed still contain a tail like
# `... far above the real side (Z).` or `... far above the fake side (Z).`.
# Strip it at read time so the UI shows the concise form.
_REASON_TAIL_RE = re.compile(r"\s+far above the (?:real|fake) side \(-?\d+(?:\.\d+)?\)\.?\s*$")


def _shorten_reason(text: str) -> str:
    return _REASON_TAIL_RE.sub(".", text)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import FAKE_TREE_PATH, REAL_TREE_PATH, CORRECTION_SET_PATH
from algo.algo_utils import load_structure_file
from utils import sort_tree_recursive

logger = logging.getLogger(__name__)

_persist_executor = ThreadPoolExecutor(max_workers=1)

# ── Private mutable state ─────────────────────────────────────────
_all_nodes = []
_fake_matches = []
_real_nodes = []      # roots of the real-news tree, needed for node-deletion paths
_real_matches = []
_intro_texts = []
_spotlight_node = None
_spot_children = []


# ── Initialization (called once at startup) ───────────────────────

def init_state():
    """Load trees from disk and populate module-level state."""
    global _all_nodes, _fake_matches, _real_nodes, _real_matches
    global _intro_texts, _spotlight_node, _spot_children

    _all_nodes, _fake_matches = load_structure_file(FAKE_TREE_PATH)
    _real_nodes, _real_matches = load_structure_file(REAL_TREE_PATH)
    _all_nodes = sort_tree_recursive(_all_nodes)
    _intro_texts = _load_wrong_predictions(CORRECTION_SET_PATH, _fake_matches, _real_matches)
    _spotlight_node = None
    _spot_children = []


def _gold_to_int(value):
    """Map the CSV gold ``label`` to model convention (0 = real, 1 = fake)."""
    if isinstance(value, bool):
        return 0 if value else 1
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "real"):
            return 0
        if s in ("false", "0", "fake"):
            return 1
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)
    return None


def _load_wrong_predictions(csv_path, fake_matches, real_matches=None):
    """Return intro_texts entries seeded from the correction-set CSV.

    Includes a row in intro_texts when ANY of:
      - ``source == 'manual'`` (runtime-added manual input persisted from a
        previous session)
      - ``ingest_status`` is non-idle (user has interacted with the row)
      - the row is a "wrong prediction" (``gold != predicted_label``).

    Rows with ``dismissed=True`` are always skipped. In-flight statuses
    (``processing`` / ``ingesting``) are decayed to ``failed`` since the
    background job that owned them died with the previous process.

    The node is resolved by looking up ``predicted_narrative`` text against
    both the fake-tree and real-tree match lists so Mark-as-true completions
    (which land in the real tree) survive a restart with their narrative
    line intact.
    """
    if not os.path.exists(csv_path):
        logger.info("No prepared correction set at %s; skipping intro_texts seeding", csv_path)
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        logger.exception("Failed to read correction set at %s", csv_path)
        return []

    required = {"text", "label", "predicted_label"}
    if not required.issubset(df.columns):
        logger.warning(
            "Correction set %s missing columns %s; have %s",
            csv_path, required - set(df.columns), list(df.columns),
        )
        return []

    fake_text_to_node = {node.text: node for node, _, _ in fake_matches}
    real_text_to_node = {}
    if real_matches:
        real_text_to_node = {node.text: node for node, _, _ in real_matches}

    entries = []
    decayed = 0
    skipped_dismissed = 0
    for _, row in df.iterrows():
        # Skip explicitly dismissed rows.
        dismissed_cell = row.get("dismissed") if "dismissed" in df.columns else None
        if dismissed_cell is not None and pd.notna(dismissed_cell):
            if isinstance(dismissed_cell, bool):
                if dismissed_cell:
                    skipped_dismissed += 1
                    continue
            elif str(dismissed_cell).strip().lower() in ("true", "1", "yes"):
                skipped_dismissed += 1
                continue

        # Read runtime fields with defaults.
        src_cell = row.get("source") if "source" in df.columns else None
        if src_cell is None or (isinstance(src_cell, float) and pd.isna(src_cell)):
            src = "expert_correction"
        else:
            src = str(src_cell).strip() or "expert_correction"

        status_cell = row.get("ingest_status") if "ingest_status" in df.columns else None
        if status_cell is None or (isinstance(status_cell, float) and pd.isna(status_cell)):
            status = "idle"
        else:
            status = str(status_cell).strip() or "idle"
        # Decay stale in-flight states — the worker process that owned them is gone.
        if status in ("processing", "ingesting"):
            decayed += 1
            status = "failed"

        gold = _gold_to_int(row["label"])
        try:
            pred = int(row["predicted_label"]) if pd.notna(row.get("predicted_label")) else None
        except (ValueError, TypeError):
            pred = None

        # Decide whether this row belongs in intro_texts.
        if src == "manual":
            include = True
        elif status and status != "idle":
            include = True
        elif gold is not None and pred is not None and gold != pred:
            include = True
        else:
            include = False
        if not include:
            continue

        narrative = row.get("predicted_narrative") if "predicted_narrative" in df.columns else None
        node = None
        if isinstance(narrative, str) and narrative:
            # Prefer the tree side that matches the current label; fall back
            # to either if the lookup misses (covers cross-tree mark flips).
            if pred == 1:
                node = fake_text_to_node.get(narrative) or real_text_to_node.get(narrative)
            elif pred == 0:
                node = real_text_to_node.get(narrative) or fake_text_to_node.get(narrative)
            else:
                node = fake_text_to_node.get(narrative) or real_text_to_node.get(narrative)

        # For a disinformation match, a stored predicted_narrative may be a raw
        # leaf (a specific fake article) when the row was classified before
        # get_label_dual_v3 began lifting leaves to their parent narrative.
        # Mirror that lift here so the correction set shows the first parent,
        # never a leaf — unless the leaf has no parent, in which case there is
        # nothing more general to show. (Real/true rows keep their leaf so the
        # spotlight + trash workflow can target that specific real-news entry.)
        if pred == 1 and node is not None and not node.children and node.parent is not None:
            node = node.parent

        reason = row.get("predicted_reason") if "predicted_reason" in df.columns else None
        if isinstance(reason, str) and reason.strip():
            # If we lifted a leaf above, the stored reason still quotes the leaf
            # text; swap in the parent narrative so both lines name the same one.
            if node is not None and isinstance(narrative, str) and narrative and narrative != node.text:
                reason = reason.replace(narrative, node.text)
            explanation = {"reason": _shorten_reason(reason)}
        else:
            explanation = None

        entries.append({
            "label": pred,
            "node": node,
            "user_text": str(row["text"]),
            "is_wrong_prediction": (
                gold is not None and pred is not None and gold != pred
            ),
            "source": src,
            "explanation": explanation,
            "ingest_status": status,
        })

    logger.info("Loaded %d intro_texts from %s (dismissed skipped: %d, decayed: %d)",
                len(entries), csv_path, skipped_dismissed, decayed)
    return entries


# ── Reload (called after model retrain / update / hard reset) ─────

def reload_trees():
    """Re-read trees from disk after a model update.

    Preserves _intro_texts, _spotlight_node, _spot_children since
    those are transient UI state the user is actively working with.
    """
    global _all_nodes, _fake_matches, _real_nodes, _real_matches

    _all_nodes, _fake_matches = load_structure_file(FAKE_TREE_PATH)
    _real_nodes, _real_matches = load_structure_file(REAL_TREE_PATH)
    _all_nodes = sort_tree_recursive(_all_nodes)


# ── Getters ───────────────────────────────────────────────────────

def get_all_nodes():
    return _all_nodes


def get_fake_matches():
    return _fake_matches


def get_real_matches():
    return _real_matches


def get_real_nodes():
    return _real_nodes


def get_intro_texts():
    return _intro_texts


def get_spotlight_node():
    return _spotlight_node


def get_spot_children():
    return _spot_children


# ── Setters ───────────────────────────────────────────────────────

def set_all_nodes(nodes):
    global _all_nodes
    _all_nodes = nodes


def set_fake_matches(matches):
    global _fake_matches
    _fake_matches = matches


def set_real_nodes(nodes):
    global _real_nodes
    _real_nodes = nodes


def set_real_matches(matches):
    global _real_matches
    _real_matches = matches


def set_spotlight_node(node):
    global _spotlight_node
    _spotlight_node = node


def set_spot_children(children):
    global _spot_children
    _spot_children = children


def update_intro_text_entry(user_text, **fields):
    """Update fields of an intro_texts entry matched by user_text.

    Returns True if an entry was found and updated, else False.
    Persists the resulting state to ``CORRECTION_SET_PATH`` so the change
    survives a process restart.
    """
    matched = False
    for e in _intro_texts:
        if e['user_text'] == user_text:
            e.update(fields)
            matched = True
            break
    if matched:
        _persist_intro_texts_to_csv()
    return matched


def add_intro_text(entry):
    """Prepend a new entry to intro_texts and persist.

    Used by ``/process_text`` for manual inputs. Returns the inserted entry
    so callers can chain further mutations through ``update_intro_text_entry``.
    """
    _intro_texts.insert(0, entry)
    _persist_intro_texts_to_csv()
    return entry


def remove_intro_text_matching(predicate):
    """Remove all entries matching ``predicate(entry) -> bool``.

    The texts of the removed entries are recorded on disk as
    ``dismissed=True`` so they don't come back on the next restart.
    Returns the list of removed entry dicts.
    """
    removed = [e for e in _intro_texts if predicate(e)]
    if not removed:
        return []
    _intro_texts[:] = [e for e in _intro_texts if not predicate(e)]
    _persist_intro_texts_to_csv(
        extra_dismissed=[e.get('user_text') for e in removed if e.get('user_text')],
    )
    return removed


# ── Async persistence ─────────────────────────────────────────────

def _write_tree_to_file(nodes_data, file_path):
    """Serialize node dicts to *file_path* atomically."""
    parent_dir = os.path.dirname(file_path)
    os.makedirs(parent_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=parent_dir, suffix=".json") as tmp:
        json.dump(nodes_data, tmp, indent=4)
        tmp_path = tmp.name
    os.replace(tmp_path, file_path)


def persist_fake_tree():
    """Snapshot the current fake tree and save to disk in a background thread.

    Serialization (``to_dict()``) happens on the calling thread so the
    snapshot is consistent.  Only the file I/O runs in the background.
    """
    nodes = _all_nodes
    sorted_trees = sorted(nodes, key=lambda x: x.level, reverse=True)
    snapshot = [tree.to_dict() for tree in sorted_trees]

    readable_path = FAKE_TREE_PATH.replace("full_result_", "readable_result_")
    readable_snapshot = [tree.to_clean_dict() for tree in sorted_trees]

    def _write():
        try:
            _write_tree_to_file(snapshot, FAKE_TREE_PATH)
            _write_tree_to_file(readable_snapshot, readable_path)
            logger.info("Fake tree persisted to %s", FAKE_TREE_PATH)
        except Exception:
            logger.exception("Failed to persist fake tree")

    _persist_executor.submit(_write)


def persist_real_tree():
    """Snapshot the current real tree and save to disk in a background thread.

    Mirror of ``persist_fake_tree`` for the real-news side. Used when the
    user deletes a node from the real tree via the X button on a card
    whose prediction landed in the real tree.
    """
    nodes = _real_nodes
    sorted_trees = sorted(nodes, key=lambda x: x.level, reverse=True)
    snapshot = [tree.to_dict() for tree in sorted_trees]

    readable_path = REAL_TREE_PATH.replace("full_result_", "readable_result_")
    readable_snapshot = [tree.to_clean_dict() for tree in sorted_trees]

    def _write():
        try:
            _write_tree_to_file(snapshot, REAL_TREE_PATH)
            _write_tree_to_file(readable_snapshot, readable_path)
            logger.info("Real tree persisted to %s", REAL_TREE_PATH)
        except Exception:
            logger.exception("Failed to persist real tree")

    _persist_executor.submit(_write)


# ── Debounced CSV persistence ────────────────────────────────────
#
# When many cards transition state in quick succession (e.g. several
# ingest jobs finishing back-to-back, or the poller observing rapid
# processing→ingesting→done flips), every call to
# update_intro_text_entry submits a CSV read+write to _persist_executor.
# The executor is single-worker, so writes pile up serially — and each
# round trip reads the multi-MB CSV from disk, mutates a few rows, then
# writes it back. Several queued writes can stall the executor and make
# the UI feel sluggish.
#
# This pair of module-level slots coalesces persists: at most one task
# is queued at a time. While a write is running, any further calls just
# update _pending_dismissed to be merged with the next write. When the
# in-flight task finishes, if _pending_dirty is True it submits one more
# task — so the last state always lands on disk.
_persist_lock = threading.Lock()
_persist_in_flight = False
_persist_dirty = False
_persist_pending_dismissed = set()


def _persist_intro_texts_to_csv(extra_dismissed=None):
    """Schedule a coalesced write of intro_texts state to ``CORRECTION_SET_PATH``.

    Multiple calls within the same write-cycle merge into a single
    eventual flush — see the ``_persist_*`` module slots above for the
    coalescing strategy. The actual snapshot is taken inside the
    executor task right before the write so it captures the latest state.

    ``extra_dismissed`` is an iterable of ``user_text`` values whose rows
    should be marked ``dismissed=True``. Across coalesced calls the
    dismissed set accumulates so no removal is dropped.
    """
    global _persist_in_flight, _persist_dirty
    extras = {str(t) for t in (extra_dismissed or []) if isinstance(t, str) and t}

    with _persist_lock:
        _persist_pending_dismissed.update(extras)
        if _persist_in_flight:
            # Someone else is already writing — mark dirty so they (or the
            # follow-up task they schedule) will pick up our updates too.
            _persist_dirty = True
            return
        _persist_in_flight = True
        _persist_dirty = False

    _persist_executor.submit(_persist_run_until_clean)


def _persist_run_until_clean():
    """Executor task: snapshot + write until no more updates are queued."""
    global _persist_in_flight, _persist_dirty
    while True:
        # Snapshot _intro_texts and accumulated dismissals atomically.
        with _persist_lock:
            dismissed_list = sorted(_persist_pending_dismissed)
            _persist_pending_dismissed.clear()
            _persist_dirty = False
        entries_snapshot = []
        for e in _intro_texts:
            ut = e.get("user_text")
            if not isinstance(ut, str) or not ut.strip():
                continue
            node = e.get("node")
            explanation = e.get("explanation") or {}
            entries_snapshot.append({
                "user_text": ut,
                "label": e.get("label"),
                "node_text": node.text if node is not None else None,
                "reason": (
                    explanation.get("reason") if isinstance(explanation, dict) else None
                ),
                "source": e.get("source", "expert_correction"),
                "ingest_status": e.get("ingest_status", "idle"),
            })

        _persist_csv_write(entries_snapshot, dismissed_list)

        # If new updates came in during the write, run another pass —
        # otherwise release the in-flight flag.
        with _persist_lock:
            if _persist_dirty or _persist_pending_dismissed:
                continue
            _persist_in_flight = False
            return


def _persist_csv_write(entries_snapshot, dismissed_list):
    """Synchronous write — called from the persist-executor task thread.

    Reads the existing CSV, applies updates and dismissals, atomically
    writes back. Errors are logged, never raised — a failed write must
    not break the executor loop.
    """
    csv_path = CORRECTION_SET_PATH
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = pd.DataFrame(columns=["text", "label"])

        # Make sure all columns we want to write exist.
        for col, default in (
            ("predicted_label", ""),
            ("predicted_narrative", ""),
            ("predicted_reason", ""),
            ("ingest_status", ""),
            ("source", ""),
            ("dismissed", False),
        ):
            if col not in df.columns:
                df[col] = default

        # Index existing rows by text for O(1) updates.
        text_to_idx = {}
        for idx, val in df["text"].items():
            if isinstance(val, str):
                text_to_idx[val] = idx

        new_rows = []
        for snap in entries_snapshot:
            ut = snap["user_text"]
            if ut in text_to_idx:
                idx = text_to_idx[ut]
                df.at[idx, "ingest_status"] = snap["ingest_status"]
                df.at[idx, "source"] = snap["source"]
                df.at[idx, "dismissed"] = False
                if snap["label"] is not None:
                    df.at[idx, "predicted_label"] = int(snap["label"])
                if snap["node_text"] is not None:
                    df.at[idx, "predicted_narrative"] = snap["node_text"]
                if snap["reason"] is not None:
                    df.at[idx, "predicted_reason"] = snap["reason"]
            else:
                new_rows.append({
                    "text": ut,
                    "label": None,
                    "predicted_label": (
                        int(snap["label"]) if snap["label"] is not None else None
                    ),
                    "predicted_narrative": snap["node_text"] or "",
                    "predicted_reason": snap["reason"] or "",
                    "ingest_status": snap["ingest_status"],
                    "source": snap["source"],
                    "dismissed": False,
                })

        # Mark explicitly-dismissed rows.
        for ut in dismissed_list:
            if ut in text_to_idx:
                df.at[text_to_idx[ut], "dismissed"] = True
            else:
                # Dismissed entry isn't on disk (likely a manual input we
                # never persisted). Append a tombstone row so it stays
                # dismissed across restarts.
                new_rows.append({
                    "text": ut,
                    "label": None,
                    "predicted_label": None,
                    "predicted_narrative": "",
                    "predicted_reason": "",
                    "ingest_status": "",
                    "source": "manual",
                    "dismissed": True,
                })

        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        parent_dir = os.path.dirname(csv_path)
        os.makedirs(parent_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=parent_dir, suffix=".csv",
        ) as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        os.replace(tmp_path, csv_path)
        logger.debug(
            "Persisted intro_texts to %s (%d entries, %d dismissed)",
            csv_path, len(entries_snapshot), len(dismissed_list),
        )
    except Exception:
        logger.exception("Failed to persist intro_texts to %s", csv_path)
