import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from constants import MAX_TOKENS

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def clean_text(my_text):
    try:
        my_text = my_text.lower()
        # lowercasing all the text
        tokens = my_text.split()
        # tokenizing
        tokens = [word for word in tokens if word not in stop_words]
        # stemming
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
        # tokens = [token for token in tokens if len(token) > 2]
        final_text = ' '.join(tokens)
        if len(final_text) < MAX_TOKENS:
            return final_text
        return final_text[:MAX_TOKENS]
    except Exception as e:
        print(f"Exception {e} for {my_text}")
        return ""


def sort_tree_recursive(nodes):
    # Sort current level
    sorted_nodes = sorted(nodes, key=lambda x: x.level, reverse=True)

    # Recurse into children
    for node in sorted_nodes:
        if node.level > 1:
            node.children = sort_tree_recursive(node.children)
    return sorted_nodes
