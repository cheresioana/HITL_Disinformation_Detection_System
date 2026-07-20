"""
Incremental tree update: graft new labeled articles into existing narrative trees
without rebuilding from scratch.

Main entry point: update_trees_with_new_data()
  - Embeds new articles
  - For each article, finds best match in the existing tree (cross-encoder)
  - If score > threshold: grafts as child of matching narrative
  - Otherwise: collects for clustering into new subtrees
  - Regenerates narrative text for any modified parent nodes
  - Saves the updated tree
"""

# Standard library
import glob
import logging
import os
import re

# Third-party
import pandas as pd

# Local
from algo.TreeNode import TreeNode
from algo.algo_utils import load_structure_file
from config import ACTIVE_THRESHOLD
from algo.create_trees import save_state, set_truth_mode, cluster_new_items
from algo.get_label_dual import get_best_score
from LLM.orchestrator import fetch_embedding, get_common_narrative, get_common_narrative_truth
from utils import clean_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_latest_result(tree_folder):
    """Find the result file with the highest threshold in a tree folder.

    Scans for files matching full_result_*.json and returns the path
    with the largest numeric threshold.
    """
    pattern = os.path.join(tree_folder, "results", "full_result_*.json")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No result files found in {tree_folder}results/")

    best_path = None
    best_threshold = -1.0

    for f in files:
        basename = os.path.basename(f)
        match = re.search(r'full_result_([\d.]+)\.json', basename)
        if match:
            threshold = float(match.group(1))
            if threshold > best_threshold:
                best_threshold = threshold
                best_path = f

    if best_path is None:
        raise FileNotFoundError(f"No valid result files in {tree_folder}results/")

    logger.info("Latest result: %s (threshold=%.2f)", best_path, best_threshold)
    return best_path


def regenerate_modified_narratives(node, modified_ids, use_truth=False):
    """Recursively find modified narrative nodes and regenerate their text.

    After grafting new articles, any parent node that received new children
    should have its narrative re-generated from ALL children (old + new).
    Only the immediate parent is regenerated — ancestors higher up are not
    touched, since a few new articles won't change the broad theme.
    """
    if id(node) in modified_ids and node.children:
        child_texts = [c.text for c in node.children]

        new_narrative = (get_common_narrative_truth(child_texts) if use_truth
                         else get_common_narrative(child_texts))

        if new_narrative and new_narrative != "no narrative":
            old_text = node.text
            node.text = new_narrative
            node.embedding = fetch_embedding(clean_text(new_narrative))
            logger.info("Updated narrative: '%s...' -> '%s...'",
                        old_text[:50], new_narrative[:50])

    for child in node.children:
        regenerate_modified_narratives(child, modified_ids, use_truth)


# ---------------------------------------------------------------------------
# Single-node graft
# ---------------------------------------------------------------------------

def add_node_to_tree(text, existing_nodes, matches, match_threshold=1.0):
    """Add a single article to an existing narrative tree.

    Embeds *text*, finds the best matching node in *matches* via
    cross-encoder, and grafts a new child node if the score exceeds
    *match_threshold*.  Otherwise the text becomes a new root node.

    Returns the (possibly extended) *existing_nodes* list and the new
    TreeNode that was created.
    """
    embedding = fetch_embedding(clean_text(str(text)))
    row_data = {'text': text, 'embedding': embedding}
    best_node, best_score = get_best_score(row_data, matches)

    new_node = TreeNode(text, embedding)

    if best_score > match_threshold and best_node.parent:
        parent = best_node.parent
        new_node.parent = parent
        parent.children.append(new_node)
        logger.info("Grafted under '%s...' (score=%.2f)", parent.text[:50], best_score)
    elif best_score > match_threshold and not best_node.parent:
        new_node.parent = best_node
        best_node.children.append(new_node)
        logger.info("Grafted under root '%s...' (score=%.2f)", best_node.text[:50], best_score)
    else:
        existing_nodes.append(new_node)
        logger.info("Added as new root (best score=%.2f < threshold=%.1f)",
                     best_score, match_threshold)

    return existing_nodes, new_node


# ---------------------------------------------------------------------------
# Batch update function
# ---------------------------------------------------------------------------

def update_trees_with_new_data(new_df, folder="", match_threshold=1.0):
    """Incrementally update both fake and real narrative trees with new data.

    For each new article:
      1. Embed it with SBERT
      2. Find nearest node in the existing tree (cosine + cross-encoder)
      3. If cross-encoder score > match_threshold -> graft as child
      4. Otherwise -> collect for clustering into new subtrees
      5. Regenerate narrative text for modified parent nodes
      6. Cluster unmatched items -> new subtrees
      7. Save updated tree

    Args:
        new_df: DataFrame with 'text' and 'label' columns (1=fake, 0=real)
        folder: tree folder prefix (e.g., 'narrative_mbd/')
        match_threshold: cross-encoder score above which a new article
                         is grafted into an existing narrative node
    """
    # ── Clean and deduplicate ──
    new_df = new_df[new_df['text'].astype(str).str.strip() != ""].copy()
    new_df['text'] = new_df['text'].astype(str).str.replace('"', '', regex=False)
    new_df = new_df.dropna(subset=['text', 'label'])
    new_df = new_df.drop_duplicates(subset='text', keep='first')

    logger.info("New data after cleaning: %d articles", len(new_df))

    # ── Embed new articles ──
    logger.info("Embedding new articles...")
    embeddings = []
    for idx, text in enumerate(new_df['text']):
        if idx % 50 == 0 and idx > 0:
            logger.info("  Progress: %d/%d", idx, len(new_df))
        embeddings.append(fetch_embedding(clean_text(str(text))))
    new_df['embedding'] = embeddings

    # ── Split by label ──
    labels = pd.to_numeric(new_df['label'], errors='coerce')
    new_fake = new_df[labels.eq(1)]
    new_real = new_df[labels.eq(0)]

    logger.info("New fake articles: %d", len(new_fake))
    logger.info("New real articles: %d", len(new_real))

    # ── Update each tree ──
    for tree_type, new_data, subfolder, use_truth in [
        ('fake', new_fake, 'false/', False),
        ('real', new_real, 'true/', True),
    ]:
        if len(new_data) == 0:
            logger.info("[%s] No new articles to add, skipping.", tree_type)
            continue

        tree_folder = folder + subfolder
        logger.info("=" * 60)
        logger.info("Updating %s tree (%d new articles)", tree_type, len(new_data))
        logger.info("=" * 60)

        # Load existing tree at highest threshold
        try:
            latest_path = find_latest_result(tree_folder)
            existing_nodes, matches = load_structure_file(latest_path)
            logger.info("Loaded tree: %d root nodes, %d total nodes",
                        len(existing_nodes), len(matches))
        except FileNotFoundError as e:
            logger.error("Skipping %s tree: %s", tree_type, e)
            continue

        # Track which parent nodes get modified
        modified_narratives = set()
        unmatched = []
        grafted = 0
        new_nodes_added = []  # leaves we add this round — used for trace logs

        print(f"\n========== INGEST [{tree_type}] processing {len(new_data)} row(s) ==========", flush=True)

        for _, row in new_data.iterrows():
            row_data = {'text': row['text'], 'embedding': row['embedding']}
            best_node, best_score = get_best_score(row_data, matches)
            text_preview = str(row['text'])[:120]
            match_preview = (best_node.text if best_node else "?")[:120]

            print(f"\n--- new row: {text_preview!r}", flush=True)
            print(f"    best match found: {match_preview!r}", flush=True)
            print(f"    score: {best_score:.4f}  (threshold={match_threshold:.2f})", flush=True)
            if best_node and best_node.parent:
                print(f"    best match's parent: {best_node.parent.text[:120]!r}", flush=True)
            else:
                print(f"    best match's parent: <root, no parent>", flush=True)

            if best_score > match_threshold and best_node.parent:
                parent = best_node.parent
                new_node = TreeNode(row['text'], row['embedding'])
                new_node.parent = parent
                parent.children.append(new_node)
                modified_narratives.add(id(parent))
                grafted += 1
                new_nodes_added.append(new_node)
                print(f"    => DECISION: GRAFT as child of parent {parent.text[:120]!r}", flush=True)
            elif best_score > match_threshold and not best_node.parent:
                new_node = TreeNode(row['text'], row['embedding'])
                new_node.parent = best_node
                best_node.children.append(new_node)
                modified_narratives.add(id(best_node))
                grafted += 1
                new_nodes_added.append(new_node)
                print(f"    => DECISION: GRAFT as child of root {best_node.text[:120]!r}", flush=True)
            else:
                node = TreeNode(row['text'], row['embedding'])
                unmatched.append(node)
                new_nodes_added.append(node)
                print(f"    => DECISION: UNMATCHED (score did not exceed threshold) — will be clustered", flush=True)

        print(f"\n[{tree_type}] summary: grafted={grafted}, unmatched={len(unmatched)}", flush=True)

        # Regenerate narratives for modified parent nodes
        if modified_narratives:
            print(f"[{tree_type}] regenerating {len(modified_narratives)} modified parent narrative(s)...", flush=True)
            set_truth_mode(use_truth)
            for node in existing_nodes:
                regenerate_modified_narratives(node, modified_narratives, use_truth)

        # Cluster unmatched items
        if unmatched:
            print(f"[{tree_type}] clustering {len(unmatched)} unmatched item(s)...", flush=True)
            set_truth_mode(use_truth)
            new_subtrees = cluster_new_items(unmatched)
            existing_nodes.extend(new_subtrees)
            print(f"[{tree_type}] created {len(new_subtrees)} new subtree(s) from unmatched:", flush=True)
            for sub in new_subtrees:
                sub_label = (sub.text or '')[:120]
                print(f"    new subtree root={sub_label!r} children={len(sub.children)}", flush=True)

        # Trace: confirm every leaf we created is still reachable in the in-memory
        # tree right before save_state runs.
        def _is_in_tree(roots, target):
            for r in roots:
                if r is target:
                    return True
                if _is_in_tree(r.children, target):
                    return True
            return False

        for n in new_nodes_added:
            present = _is_in_tree(existing_nodes, n)
            print(f"    PRE-SAVE: leaf {(n.text or '')[:120]!r} reachable_in_tree={present}", flush=True)

        # Save updated tree (use the active threshold so reload picks it up)
        save_state(existing_nodes, f"result_{ACTIVE_THRESHOLD}", folder=tree_folder)
        print(f"[{tree_type}] saved tree to {tree_folder}results/full_result_{ACTIVE_THRESHOLD}.json", flush=True)

        # Trace: read the saved file back and confirm each new leaf's text appears.
        try:
            saved_path = os.path.join(tree_folder, "results", f"full_result_{ACTIVE_THRESHOLD}.json")
            import json as _json
            with open(saved_path) as _fh:
                _saved = _json.load(_fh)

            def _walk_texts(n):
                yield n.get("text", "")
                for c in n.get("children", []) or []:
                    yield from _walk_texts(c)

            saved_texts = set()
            for _root in _saved:
                for _t in _walk_texts(_root):
                    saved_texts.add(_t)

            for n in new_nodes_added:
                hit = n.text in saved_texts
                print(f"    POST-SAVE: leaf {(n.text or '')[:120]!r} present_in_file={hit}", flush=True)
        except Exception as _exc:
            print(f"[{tree_type}] post-save verification FAILED: {_exc}", flush=True)

        print(f"========== INGEST [{tree_type}] done ==========\n", flush=True)

    set_truth_mode(False)
    print("INGEST: both trees processed.", flush=True)
