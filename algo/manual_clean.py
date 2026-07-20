"""Manual tree cleaning utilities."""


def remove_node(node, del_narrative, keep_child):
    """Remove a node from the tree by matching its text.

    If the node's text matches *del_narrative*, its children listed in
    *keep_child* are promoted.  Otherwise the node is kept and its
    subtree is recursively cleaned.

    Parameters
    ----------
    node : TreeNode
        Root of the (sub)tree to process.
    del_narrative : str | list[str]
        Narrative text(s) to remove.
    keep_child : list[str]
        Child texts to promote when their parent is removed.

    Returns
    -------
    list[TreeNode]
        Remaining nodes after removal.
    """
    # Normalise to list so both str and list work
    targets = [del_narrative] if isinstance(del_narrative, str) else del_narrative

    if node.text in targets:
        new_nodes = []
        for child in node.children:
            if not keep_child or child.text in keep_child:
                new_nodes.append(child)
        return new_nodes

    new_children = []
    for child in node.children:
        new_children.extend(remove_node(child, del_narrative, keep_child))
    for child in new_children:
        child.parent = node
    node.children = new_children
    return [node]
