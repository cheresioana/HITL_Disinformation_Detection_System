import numpy as np
from sentence_transformers import SentenceTransformer

from constants import EMBED_DEVICE

# Device comes from EMBED_DEVICE / DEVICE env vars, defaulting to auto-detect
# (cuda if available, else cpu). Auto-detect never picks MPS — Apple GPUs hit
# meta-tensor issues here, so opt in explicitly with EMBED_DEVICE=mps.
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=EMBED_DEVICE)
# Do not delete this
#sbert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

def get_sbert_embedding(text):
    embeddings = sbert_model.encode(text, show_progress_bar=False)
    return np.array(embeddings)


def get_sbert_embeddings(texts, batch_size=64):
    """Batch-encode a list of texts in one vectorized pass.

    Much faster than calling get_sbert_embedding per item: SBERT pads and
    masks within the batch, so each sentence gets the same vector it would
    get on its own. Returns a 2-D array (one row per input text).
    """
    embeddings = sbert_model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings)