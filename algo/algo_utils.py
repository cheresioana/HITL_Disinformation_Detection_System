import json

from algo.TreeNode import TreeNode

# ── Tree loading ─────────────────────────────────────────────────────

def traverse_tree(node, matches, node_id):
    """Recursively collect (node, embedding, tree_id) tuples."""
    matches.append((node, node.embedding, node_id))
    for child in node.children:
        traverse_tree(child, matches, node_id)


def load_structure_file(filename):
    """Load a tree JSON file and return (root_nodes, flat_matches)."""
    with open(filename, "r") as fh:
        data = json.load(fh)
    root_nodes = [TreeNode.from_dict(tree) for tree in data]

    matches = []
    for node_id, node in enumerate(root_nodes):
        traverse_tree(node, matches, node_id)
    return root_nodes, matches


