import logging
import shutil
import tempfile
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, url_for, redirect, jsonify

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    FAKE_TREE_PATH, REAL_TREE_PATH, NARRATIVE_DIR,
    BACKUP_FAKE_TREE_PATH, BACKUP_REAL_TREE_PATH,
    UPLOAD_DIR, UPDATE_DATASET_PATH, FULL_DATASET_PATH, ALLOWED_EXT,
)
from LLM.orchestrator import fetch_embedding
from algo.algo_utils import traverse_tree
from algo.manual_clean import remove_node
from algo.get_label_dual import invalidate_cache
from algo.update_model import update_trees_with_new_data
from algo.create_trees import train_narrative_tree_from_dataframe
from utils import clean_text

from status import status_bp, set_full_retrain, set_update_model, set_hard_reset, set_ingest_status
from api import api_bp, _ingest_job, _run_analysis, _analysis_executor
from state import (
    init_state, reload_trees, persist_fake_tree, persist_real_tree,
    get_all_nodes, set_all_nodes,
    get_fake_matches, set_fake_matches,
    get_real_nodes, set_real_nodes,
    get_real_matches, set_real_matches,
    get_intro_texts, update_intro_text_entry,
    remove_intro_text_matching,
    get_spotlight_node, set_spotlight_node,
    get_spot_children, set_spot_children,
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.register_blueprint(status_bp)
app.register_blueprint(api_bp)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
init_state()

executor = ProcessPoolExecutor(max_workers=1)
app.config['EXECUTOR'] = executor

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

def _spotlight_tree_side(node):
    """Return 'true' if *node* lives in the real tree, 'false' if in the fake
    tree, or '' if it's not currently in either match list.

    Identity check (``is``) is safe because both spotlight transitions
    (``/get_internal_structure``, ``/go_to_children``, ``/go_to_parent``) pull
    the live ``TreeNode`` instance straight out of the in-memory tree, and
    those instances are what populate the flat match lists.
    """
    if node is None:
        return ''
    for n, _, _ in get_real_matches():
        if n is node:
            return 'true'
    for n, _, _ in get_fake_matches():
        if n is node:
            return 'false'
    return ''


@app.route("/")
def home():
    spotlight_node = get_spotlight_node()
    return render_template("index.html",
                           intro_texts=get_intro_texts(),
                           spotlight_node=spotlight_node,
                           spotlight_tree=_spotlight_tree_side(spotlight_node),
                           children=get_spot_children())

# ---------------------------------------------------------------------------
# Tree management routes
# ---------------------------------------------------------------------------

@app.route('/get_internal_structure', methods=['POST'])
def get_internal_structure():
    """Set the spotlight to the node whose text matches ``narrative_text``.

    Lookup order:
      1. intro_texts — preferred because it preserves the card → node link.
      2. fake_matches (flat list of all fake-tree nodes).
      3. real_matches (flat list of all real-tree nodes).

    The fallbacks matter because: a card's node could be ``None`` (e.g.
    a True prediction without a strong real-tree match), and a button
    rendered before an edit may carry stale text — but the node still
    lives in one of the match lists by its current text via identity.
    """
    narrative_text = request.form.get('narrative_text')
    logger.info("Get internal structure: %s", narrative_text)

    target = None
    # 1) intro_texts
    for element in get_intro_texts():
        node = element.get('node')
        if node is not None and node.text == narrative_text:
            target = node
            break
    # 2) fake tree
    if target is None:
        for node, _emb, _idx in get_fake_matches():
            if node.text == narrative_text:
                target = node
                break
    # 3) real tree
    if target is None:
        for node, _emb, _idx in get_real_matches():
            if node.text == narrative_text:
                target = node
                break

    if target is not None:
        set_spotlight_node(target)
        set_spot_children(target.children)
        logger.debug("Spotlight set: %d children", len(get_spot_children()))
    else:
        logger.warning(
            "get_internal_structure: no node found for narrative_text=%r",
            narrative_text,
        )
    return redirect(url_for('home'))

@app.route('/edit_node', methods=['POST'])
def edit_node():
    spotlight_node = get_spotlight_node()
    if spotlight_node is None:
        return redirect(url_for('home'))

    old_text = request.form.get('old_text')
    new_text = request.form.get('new_text')
    logger.info("Edit node: %r -> %r", old_text, new_text)
    new_embedding = fetch_embedding(clean_text(new_text))
    # spotlight_node is the live TreeNode object shared with both the fake-
    # and real-tree match lists. Mutating it in place propagates to those
    # lists automatically — the tuples store the node by reference. We
    # still have to invalidate the retrieval cache (keyed on
    # ``id(matches)``) so subsequent classifications recompute the
    # candidate embeddings with the new value.
    spotlight_node.text = new_text
    spotlight_node.embedding = new_embedding
    invalidate_cache()

    # Persist whichever tree this node lives in.
    tree_side = _spotlight_tree_side(spotlight_node)
    if tree_side == 'false':
        persist_fake_tree()
    elif tree_side == 'true':
        persist_real_tree()

    # Any card whose ``node`` IS this spotlight node has a stale reason
    # string and possibly stale label/scores (the LLM reasoning mentions
    # the old text, the cross-encoder scored against the old text). Mark
    # them ``processing`` and re-classify so the cards reflect the new
    # narrative wording.
    affected = [e for e in get_intro_texts() if e.get('node') is spotlight_node]
    for entry in affected:
        ut = entry.get('user_text')
        if not ut:
            continue
        update_intro_text_entry(
            ut, ingest_status='processing', node=None, label=None, explanation=None,
        )
        logger.info("Re-classifying after edit: %s", ut[:80])
        _analysis_executor.submit(_run_analysis, ut)

    return redirect(url_for('home'))

def _find_node_by_text(roots, text):
    for root in roots:
        if root.text == text:
            return root
        found = _find_node_by_text(root.children, text)
        if found is not None:
            return found
    return None


def _find_real_node_by_text(text):
    """Locate a node in the real tree by exact text match.

    The real tree is held as a flat ``(node, embedding, tree_id)`` list in
    state — search it linearly. Used to surface the insertion point for
    Mark-as-true on a correction card.
    """
    for node, _emb, _idx in get_real_matches():
        if node.text == text:
            return node
    return None


def _kickoff_correction_ingest(text, numeric_label):
    """Flip the card to ingesting and run update_trees_with_new_data async.

    numeric_label: 1 = fake, 0 = true. Only the fake path looks up a
    spotlight target — the true path stores no node since the UI's
    spotlight panel renders fake narratives.
    """
    update_intro_text_entry(text, ingest_status='ingesting', source='expert_correction')

    # update_trees_with_new_data strips quotes — mirror that here so lookup matches.
    lookup_text = str(text).replace('"', '')

    # Per-call CSV path. The fixed name "ingest_dataset.csv" used to be
    # reused across calls, which races when two cards are marked while a
    # previous ingest is still queued: the second click overwrites the CSV
    # before the first worker reads it, so jobA ends up ingesting B's text
    # and then fails its post-ingest tree lookup. Unique suffix per ingest
    # serialises the inputs, the callback unlinks the file when it's done.
    df = pd.DataFrame({"text": [text], "label": [numeric_label]})
    ingest_path = UPLOAD_DIR / f"ingest_dataset_{uuid.uuid4().hex}.csv"
    df.to_csv(ingest_path, index=False)

    set_ingest_status(1, "Work in progress")
    future = executor.submit(_ingest_job, ingest_path)

    def _on_done(fut):
        try:
            fut.result()
            print(f"\n>>> CALLBACK: ingest subprocess finished, reloading trees from disk...", flush=True)
            reload_trees()
            print(f">>> CALLBACK: trees reloaded; looking for leaf {lookup_text[:120]!r}", flush=True)
            updates = {
                'label': numeric_label,
                'ingest_status': 'done',
                'is_wrong_prediction': False,
            }
            if numeric_label == 1:
                target = _find_node_by_text(get_all_nodes(), lookup_text)
                if target is not None:
                    focus = target.parent if target.parent is not None else target
                    updates['node'] = focus
                    print(f">>> CALLBACK: FOUND leaf — parent (spotlight) = {focus.text[:120]!r}", flush=True)
                    logger.info("Fake ingest done, card linked to '%s...'", focus.text[:50])
                else:
                    update_intro_text_entry(text, ingest_status='failed')
                    set_ingest_status(2, None)
                    print(f">>> CALLBACK: NOT FOUND in reloaded tree for {lookup_text[:120]!r}", flush=True)
                    prefix = lookup_text.split()[0] if lookup_text else ""
                    neighbors = []
                    def _collect(nodes, depth=0):
                        for n in nodes:
                            if prefix and prefix.lower() in (n.text or "").lower():
                                neighbors.append((depth, n.text[:120]))
                            _collect(n.children, depth + 1)
                    _collect(get_all_nodes())
                    print(f">>> CALLBACK: {len(neighbors)} near-neighbor text(s) containing {prefix!r} (depth, text):", flush=True)
                    for d, t in neighbors[:15]:
                        print(f"      d={d} {t!r}", flush=True)
                    return
            else:
                target = _find_real_node_by_text(lookup_text)
                if target is not None:
                    focus = target.parent if target.parent is not None else target
                    updates['node'] = focus
                    print(f">>> CALLBACK: FOUND real leaf — parent (theme) = {focus.text[:120]!r}", flush=True)
                    logger.info("True ingest done, card linked to '%s...'", focus.text[:50])
                else:
                    updates['node'] = None
                    print(f">>> CALLBACK: NOT FOUND in real tree for {lookup_text[:120]!r}", flush=True)
                    logger.info("True ingest done, no real-tree node found for: %s", lookup_text[:80])
            update_intro_text_entry(text, **updates)
            set_ingest_status(2, None)
        except Exception as exc:
            update_intro_text_entry(text, ingest_status='failed')
            set_ingest_status(-1, str(exc))
            logger.error("Ingestion failed: %s", exc)
        finally:
            # Clean up the per-call CSV regardless of outcome — keeps the
            # uploads directory from accumulating one file per click.
            try:
                ingest_path.unlink(missing_ok=True)
            except Exception as cleanup_exc:
                logger.warning("Could not remove temp ingest CSV %s: %s",
                               ingest_path, cleanup_exc)

    future.add_done_callback(_on_done)


@app.route('/mark_item_as_fake', methods=['POST'])
def mark_item_as_fake():
    text = request.form.get('narrative_text')
    logger.info("Mark item as fake: %s", text)
    _kickoff_correction_ingest(text, 1)
    return redirect(url_for('home'))


@app.route('/mark_item_as_true', methods=['POST'])
def mark_item_as_true():
    text = request.form.get('narrative_text')
    logger.info("Mark item as true: %s", text)
    _kickoff_correction_ingest(text, 0)
    return redirect(url_for('home'))

@app.route('/go_to_parent', methods=['POST'])
def go_to_parent():
    spotlight_node = get_spotlight_node()
    parent = spotlight_node.parent
    set_spotlight_node(parent)
    if parent is None:
        set_spot_children(get_all_nodes()[:15])
    else:
        set_spot_children(parent.children)
    return redirect(url_for('home'))

@app.route('/go_to_children', methods=['POST'])
def go_to_children():
    spotlight_node = get_spotlight_node()
    narrative_text = request.form.get('narrative_text')
    logger.info("Go to children: %s", narrative_text)
    if spotlight_node is not None:
        for element in spotlight_node.children:
            if element.text == narrative_text:
                logger.debug("Found element %s", element.text)
                set_spotlight_node(element)
                set_spot_children(element.children)
    else:
        for element in get_spot_children():
            if element.text == narrative_text:
                set_spotlight_node(element)
                set_spot_children(element.children)
    return redirect(url_for('home'))


def remove_node_nodes(node):
    all_nodes = get_all_nodes()
    new_nodes = []
    for element in all_nodes:
        new_nodes.extend(remove_node(element, node.text, []))
    set_all_nodes(new_nodes)
    new_matches = []
    node_id = 0
    for n in new_nodes:
        traverse_tree(n, new_matches, node_id)
        node_id = node_id + 1
    set_fake_matches(new_matches)
    invalidate_cache()
    logger.info(
        "remove_node_nodes: dropped fake node %r (fake matches: %d)",
        node.text[:80], len(new_matches),
    )


def remove_real_node_nodes(node):
    """Real-tree mirror of ``remove_node_nodes``.

    Walks the real-tree root list, drops every node whose text matches
    ``node.text``, then rebuilds the flat match cache so retrieval picks
    up the change immediately.
    """
    real_nodes = get_real_nodes()
    before_len = sum(1 for _ in _walk_all(real_nodes))
    new_nodes = []
    for element in real_nodes:
        new_nodes.extend(remove_node(element, node.text, []))
    set_real_nodes(new_nodes)
    new_matches = []
    node_id = 0
    for n in new_nodes:
        traverse_tree(n, new_matches, node_id)
        node_id = node_id + 1
    set_real_matches(new_matches)
    # Drop any embedding cache keyed on the old match-list id so subsequent
    # retrievals don't accidentally hit stale embeddings if Python recycles
    # the id of the discarded list.
    invalidate_cache()
    logger.info(
        "remove_real_node_nodes: dropped node %r (real nodes: %d → %d)",
        node.text[:80], before_len, len(new_matches),
    )


def _walk_all(roots):
    """Yield every node reachable from *roots* (depth-first)."""
    stack = list(roots)
    while stack:
        n = stack.pop()
        yield n
        stack.extend(n.children)

@app.route('/remove_node', methods=['POST'])
def remove_node_endpoint():
    spotlight_node = get_spotlight_node()
    spot_children = get_spot_children()
    narrative_text = request.form.get('narrative_text')
    logger.info("Remove node: %s", narrative_text)
    fake_changed = False
    if spotlight_node is not None and spotlight_node.text == narrative_text:
        logger.debug("Removing spotlight node")
        remove_node_nodes(spotlight_node)
        set_spotlight_node(None)
        set_spot_children([])
        fake_changed = True
    else:
        new_children = []
        for element in spot_children:
            if element.text == narrative_text:
                remove_node_nodes(element)
                fake_changed = True
            else:
                new_children.append(element)
        set_spot_children(new_children)

    # Also try the real tree — the spotlight trash icon on a real-tree
    # node needs to drop that leaf from the real tree, not just the fake
    # one. Walk the real tree by text; if the article is there as a leaf,
    # drop it and persist the real-tree JSON to disk.
    real_changed = False
    real_target = _find_real_node_by_text(narrative_text)
    if real_target is not None:
        logger.debug("Removing real-tree node by text match: %s", narrative_text)
        remove_real_node_nodes(real_target)
        real_changed = True

    # Any card whose `node` pointed at the just-deleted text now references
    # a stale node. Don't drop those cards from the list — instead mark
    # them ``processing`` and re-run classification, so the user sees the
    # updated label (with the deleted leaf no longer in the trees).
    affected = [
        e for e in get_intro_texts()
        if e.get('node') is not None and e['node'].text == narrative_text
    ]
    for entry in affected:
        ut = entry.get('user_text')
        if not ut:
            continue
        update_intro_text_entry(
            ut, ingest_status='processing', node=None, label=None, explanation=None,
        )
        logger.info("Re-classifying affected card: %s", ut[:80])
        _analysis_executor.submit(_run_analysis, ut)

    if fake_changed:
        persist_fake_tree()
    if real_changed:
        persist_real_tree()
    return redirect(url_for('home'))


@app.route('/dismiss_card', methods=['POST'])
def dismiss_card_endpoint():
    """Dismiss a correction-set card without modifying the fake tree.

    Used when the reviewer believes the prediction is correct, or isn't
    sure — the card disappears from the UI queue but the underlying
    narrative node in the tree is untouched (unlike /remove_node, which
    also deletes the node from the fake tree). Logged separately so we
    can distinguish "no signal from reviewer" from "reviewer marked the
    narrative as deletable".
    """
    narrative_text = request.form.get('narrative_text')
    logger.info("Dismiss card (no correction): %s", narrative_text)
    remove_intro_text_matching(lambda e: e.get('user_text') == narrative_text)
    return redirect(url_for('home'))

# ---------------------------------------------------------------------------
# CSV upload helper
# ---------------------------------------------------------------------------

def save_uploaded_csv(file_storage, dest_path: Path) -> Path:
    """Validate and save an uploaded CSV file atomically."""
    fn = (file_storage.filename or "").lower()
    if not any(fn.endswith(ext) for ext in ALLOWED_EXT):
        raise ValueError("Only .csv files are accepted")

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=dest_path.parent) as tmp:
        stream = file_storage.stream
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            tmp.write(chunk)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, dest_path)  # atomic on POSIX/NTFS
    df = pd.read_csv(dest_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Incorrect columns")
    labels = pd.to_numeric(df['label'], errors="coerce")
    if not labels.notna().all() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("label column must contain only 0/1 with no NaNs.")
    return dest_path

# ---------------------------------------------------------------------------
# Model management routes (async background jobs)
# ---------------------------------------------------------------------------

def update_model_incremental(path):
    new_df = pd.read_csv(path, skipinitialspace=True)
    new_df.columns = new_df.columns.str.strip()
    update_trees_with_new_data(new_df, folder=str(NARRATIVE_DIR) + '/')
    return 1

def full_retrain_from_scratch(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    train_narrative_tree_from_dataframe(df, folder=str(NARRATIVE_DIR) + '/')
    return 1

def on_done_update(fut):
    try:
        fut.result()
        set_update_model(2, None)
        reload_trees()
        logger.info("Model update completed")
    except Exception as e:
        set_update_model(-1, str(e))
        logger.error("Model update failed: %s", e)

@app.post("/update_model")
def update_model():
    file = request.files.get("dataset")
    logger.info("Update model request received")
    if not file:
        set_update_model(-1, str("File in incorrect format"))
        return redirect('/')
    try:
        file_path = save_uploaded_csv(file, UPDATE_DATASET_PATH)
        set_update_model(1, str("Work in progress"))
        future = executor.submit(update_model_incremental, file_path)
        future.add_done_callback(on_done_update)
        return redirect('/')
    except Exception as e:
        set_update_model(-1, str(e))
        return redirect('/')

def on_done(fut):
    try:
        fut.result()
        set_full_retrain(2, None)
        reload_trees()
        logger.info("Full retrain completed")
    except Exception as e:
        set_full_retrain(-1, str(e))
        logger.error("Full retrain failed: %s", e)

@app.post("/full_retrain")
def full_retrain():
    file = request.files.get("dataset")
    if not file:
        set_full_retrain(-1, str("File in incorrect format"))
        return redirect('/')
    try:
        file_path = save_uploaded_csv(file, FULL_DATASET_PATH)
        set_full_retrain(1, str("Work in progress"))
        future = executor.submit(full_retrain_from_scratch, file_path)
        future.add_done_callback(on_done)
        return redirect('/')
    except Exception as e:
        set_full_retrain(-1, str(e))
        return redirect('/')


def _atomic_copy(src: Path, dst: Path):
    """Copy *src* to *dst* atomically (write to temp, then rename)."""
    if not src.exists():
        raise FileNotFoundError(f"Backup file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fsrc, tempfile.NamedTemporaryFile("wb", delete=False, dir=dst.parent) as tmp:
        shutil.copyfileobj(fsrc, tmp)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, dst)


def hard_reset_job():
    _atomic_copy(Path(BACKUP_FAKE_TREE_PATH), Path(FAKE_TREE_PATH))
    _atomic_copy(Path(BACKUP_REAL_TREE_PATH), Path(REAL_TREE_PATH))
    return 1

def on_done_hard_reset(fut):
    try:
        fut.result()
        set_hard_reset(2, None)
        reload_trees()
        logger.info("Hard reset completed")
    except Exception as e:
        set_hard_reset(-1, str(e))
        logger.error("Hard reset failed: %s", e)

@app.post("/hard_reset")
def hard_reset():
    set_hard_reset(1, None)
    fut = executor.submit(hard_reset_job)
    fut.add_done_callback(on_done_hard_reset)
    return redirect(url_for("home"))

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    app.run(host='0.0.0.0', port=5003)
