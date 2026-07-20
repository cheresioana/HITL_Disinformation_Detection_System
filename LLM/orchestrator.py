from LLM.BERT import get_sbert_embedding, get_sbert_embeddings
from LLM.Gemma import get_gemma_narrative, get_gemma_narrative_truth, gemma_is_narrative_entailment

from constants import EMB
from utils import clean_text


def fetch_embedding(title):
    if EMB == "SBERT":
        return get_sbert_embedding(clean_text(title)).tolist()
    else:
        print("ERROR NO EMBEDDING SELECTED")


def fetch_embeddings(titles, batch_size=64):
    """Batched counterpart of fetch_embedding.

    Applies clean_text to every title (matching fetch_embedding), encodes the
    whole list in one batched pass, and returns a list of per-item embeddings
    (each a plain Python list, so the output is drop-in for
    [fetch_embedding(t) for t in titles]).
    """
    if EMB == "SBERT":
        cleaned = [clean_text(t) for t in titles]
        return [emb.tolist() for emb in get_sbert_embeddings(cleaned, batch_size=batch_size)]
    else:
        print("ERROR NO EMBEDDING SELECTED")
        return None


def get_common_narrative(texts: list):
    return get_gemma_narrative(texts)


def get_common_narrative_truth(texts: list):
    return get_gemma_narrative_truth(texts)


def is_narrative(fake_news, narrative):
    '''

    :param fake_news:
    :param narrative:
    :return:
    ent label (1 if entailment)
    result: full result
    '''
    return gemma_is_narrative_entailment(fake_news, narrative)
