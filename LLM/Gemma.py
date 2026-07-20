import logging
import time
import json
import os
import requests

from config import OLLAMA_MODEL_GEN, OLLAMA_URL, OLLAMA_MODEL_SMALL, OLLAMA_MODEL_ENT
from constants import LANGUAGE

if LANGUAGE == "RO":
    from LLM.propmpts_ro import (
        SYS_PROMPT_NARRATIVE, EXAMPLES_NARRATIVE, USER_PROMPT_NARRATIVE,
        SYS_PROMPT_ENT, EXAMPLES_ENT, USER_PROMPT_ENT,
        SYS_PROMPT_NARRATIVE_TRUTH, EXAMPLES_NARRATIVE_TRUTH, USER_PROMPT_NARRATIVE_TRUTH,
        SYS_PROMPT_JUDGE, EXAMPLES_JUDGE, USER_PROMPT_JUDGE
    )
else:
    from LLM.propmpts import (
        SYS_PROMPT_NARRATIVE, EXAMPLES_NARRATIVE, USER_PROMPT_NARRATIVE,
        SYS_PROMPT_ENT, EXAMPLES_ENT, USER_PROMPT_ENT,
        SYS_PROMPT_NARRATIVE_TRUTH, EXAMPLES_NARRATIVE_TRUTH, USER_PROMPT_NARRATIVE_TRUTH,
        SYS_PROMPT_JUDGE, EXAMPLES_JUDGE, USER_PROMPT_JUDGE
    )

def gemma_chat(messages, model=OLLAMA_MODEL_GEN, temperature=None):
    data = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if temperature is not None:
        # Ollama reads generation params from "options" — a top-level
        # "temperature" key is silently ignored and the model falls back to
        # its default (~0.8). Pin a seed too so temperature=0 is reproducible.
        data["options"] = {
            "temperature": temperature,
            "seed": 42,
        }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(OLLAMA_URL, headers=headers, json=data)
    return response.json()["message"]["content"]


def change_to_gemma_format(messages):
    new_format = []
    for message in messages:
        new_format.append({
            "role": message['role'],
            "content": message['content']
        })
    return new_format


def extract_narrative_from_verbose(resp_text):
    """Try to extract the narrative from a verbose response."""
    # Try to find bold text like **Narrative here.**
    import re
    bold_match = re.search(r'\*\*(.+?)\*\*', resp_text)
    if bold_match:
        narrative = bold_match.group(1).strip().rstrip('.')
        if len(narrative) < 200:
            return narrative

    # Otherwise take the first non-empty line that looks like a sentence
    for line in resp_text.strip().split('\n'):
        line = line.strip().lstrip('- ').lstrip('* ')
        if 10 < len(line) < 200 and not line.startswith(('Let', 'Here', 'Okay', 'Do you', 'I ', 'The ', 'This ')):
            return line

    return ""


def get_gemma_narrative(texts):
    for attempt in range(2):
        try:
            messages = [{"role": "system", "content": SYS_PROMPT_NARRATIVE}]
            messages.extend(EXAMPLES_NARRATIVE)
            messages.append({
                "role": "user",
                "content": USER_PROMPT_NARRATIVE.format(joined_statements='\n'.join(texts))
            })
            resp_text = gemma_chat(change_to_gemma_format(messages), temperature=0, model=OLLAMA_MODEL_SMALL)

            # Clean up: strip whitespace and quotes
            resp_text = resp_text.strip().strip('"').strip()

            if "I can't" in resp_text or "I cannot" in resp_text:
                logging.warning(f"Narrative rejected: {resp_text}")
                continue

            # If short enough, use it directly
            if len(resp_text) < 200:
                return resp_text

            # If verbose, try to extract the narrative
            extracted = extract_narrative_from_verbose(resp_text)
            if extracted:
                logging.info(f"Extracted narrative from verbose response: {extracted}")
                return extracted

            logging.warning(f"Narrative too long or rejected: {resp_text[:200]}...")
        except Exception as e:
            logging.error(f"Error generating narrative: {e}")
            time.sleep(1)
    return ""


def get_gemma_narrative_truth(texts):
    """Generate a common factual news theme from a list of real news texts."""
    for attempt in range(2):
        try:
            messages = [{"role": "system", "content": SYS_PROMPT_NARRATIVE_TRUTH}]
            messages.extend(EXAMPLES_NARRATIVE_TRUTH)
            messages.append({
                "role": "user",
                "content": USER_PROMPT_NARRATIVE_TRUTH.format(joined_statements='\n'.join(texts))
            })
            resp_text = gemma_chat(change_to_gemma_format(messages), temperature=0, model=OLLAMA_MODEL_SMALL)
            resp_text = resp_text.strip().strip('"').strip()

            if "I can't" in resp_text or "I cannot" in resp_text:
                logging.warning(f"Truth narrative rejected: {resp_text}")
                continue

            if len(resp_text) < 200:
                return resp_text

            extracted = extract_narrative_from_verbose(resp_text)
            if extracted:
                logging.info(f"Extracted truth narrative from verbose response: {extracted}")
                return extracted

            logging.warning(f"Truth narrative too long or rejected: {resp_text[:200]}...")
        except Exception as e:
            logging.error(f"Error generating truth narrative: {e}")
            time.sleep(1)
    return ""


def gemma_judge_fake_or_real(text, fake_narratives, real_section):
    """Ask Gemma to classify text as FAKE or REAL given narrative context.

    ``fake_narratives`` is a pre-formatted block of numbered disinformation
    narratives (e.g. "Disinformation narrative 1: \"…\"\\nDisinformation
    narrative 2: \"…\""), constructed by ``build_judge_context`` from the
    top-N fake-tree parents deduplicated.
    """
    for attempt in range(3):
        try:
            messages = [{"role": "system", "content": SYS_PROMPT_JUDGE}]
            messages.extend(EXAMPLES_JUDGE)
            messages.append({
                "role": "user",
                "content": USER_PROMPT_JUDGE.format(
                    text=text,
                    fake_narratives=fake_narratives,
                    real_section=real_section,
                )
            })

            response = gemma_chat(change_to_gemma_format(messages), temperature=0)
            response = response.strip().upper()
            if "FAKE" in response:
                return "FAKE"
            return "REAL"
        except Exception as e:
            logging.error(f"Judge call failed: {e}")
            time.sleep(1)
    return "REAL"


def gemma_is_narrative_entailment(headline, narrative):
    for attempt in range(5):
        response_text = "og"
        try:
            messages = [{"role": "system", "content": SYS_PROMPT_ENT}]
            messages.extend(EXAMPLES_ENT)
            messages.append({
                "role": "user",
                "content": USER_PROMPT_ENT.format(headline=headline, narrative=narrative)
            })
            response_text = gemma_chat(change_to_gemma_format(messages), temperature=0, model=OLLAMA_MODEL_ENT)
            result = json.loads(response_text)

            if result.get("label") == "entailment":
                return 1, result
            return 0, result
        except Exception as e:
            logging.error(
                f"Entailment check failed: {e} | Headline: {headline} | Narrative: {narrative} | Result {response_text}")
            time.sleep(1)
    return 0, {"label": "error", "score": 1}
