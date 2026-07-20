class TreeNode:
    def __init__(self, text, embedding, children=None, parent=None):
        self.text = text
        self.embedding = embedding
        self.children = children if children else []
        self.level = sum(node.level for node in children) if children else 1
        self.parent = parent
        for child in self.children:
            child.parent = self

    def count_leaf_nodes(self):
        if not self.children:  # Leaf node condition
            return 1
        return sum(child.count_leaf_nodes() for child in self.children)

    def add_child(self, child_node):
        self.children.append(child_node)

    def to_dict(self):
        return {
            "text": self.text,
            "level": self.level,
            "embedding": self.embedding,
            "children": [child.to_dict() for child in self.children],
        }

    def to_clean_dict(self):
        return {
            "text": self.text,
            "level": self.level,
            "children": [child.to_clean_dict() for child in self.children],
        }

    @staticmethod
    def from_dict(data, parent=None):
        node = TreeNode(data["text"], data["embedding"], parent=parent)
        node.children = [TreeNode.from_dict(child, node) for child in data["children"]]
        node.level = sum(node.level for node in node.children) if node.children else 1
        return node