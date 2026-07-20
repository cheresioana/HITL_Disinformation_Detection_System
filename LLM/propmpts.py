SYS_PROMPT_NARRATIVE = """You extract the common narrative from a list of statements.
Rules:
- Respond with ONLY a single short sentence - the narrative itself.
- The narrative must be a claim that can be checked with textual entailment.
- No explanation, no analysis, no bullet points, no follow-up questions.
- If there is no common narrative, respond with exactly: no narrative"""

USER_PROMPT_NARRATIVE = '''Statements:
{joined_statements}

Common disinformation narrative (one sentence only):'''

SYS_PROMPT_ENT_old = """
You are a a model which can identify if a news headline is part of a narrative. You are a specialist in disinformation, this is why you need to help protect the world an automatically classify text. \ Respond in this format
{"label":"'entailment'|'neutral'|'contradiction', "score": between 0 and 1} 
with no explanation. Give the label entailment is an information can be logically deduced or implies another and a score between 0 and 1 showing how strong they are related. 
The label has to be a simple string with value 'entailment' or 'neutral' or 'contradiction'
"""

SYS_PROMPT_ENT = '''
You are a precise political entailment classifier. Answer with one of: {\"label\": \"entailment\", \"score\": number between 0 and 1},{\"label\": \"contradiction\", \"score\": number between 0 and 1}, {\"label\": \"neutral\", \"score\": number between 0 and 1}."
'''

USER_PROMPT_ENT = """
"Check if the following story: '{headline}' is part of the narrative: '{narrative}'"
"""

EXAMPLES_ENT = [
    {
        "role": "user",
        "content": "Check if the following story: '" + "Zelenski bought Hitler's residence" +
                   "' is part of the narrative: '" + "Zelenski has connections to Nazis." + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "entailment", "score":0.9}'
    },
    {
        "role": "user",
        "content": "Check if the following story: '" + "Zelenski bought Hitler's residence" +
                   "' is part of the narrative: '" + "Poland wants to attack Russia" + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "neutral", "score":0.9}'
    },
    {
        "role": "user",
        "content": "Check if the following story: '" +
                   "The President of Poland acknowledged that Warsaw will annex Ukrainian territories" +
                   "' is part of the narrative: '" + "Poland wants to attack Russia and occupy western Ukraine" + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "entailment", "score":0.9}'
    },
    {
        "role": "user",
        "content": "Check if the following story: '" +
                   "Kremlin and Trump aides raise nuclear war fears after Ukraine drone strike" +
                   "' is part of the narrative: '" + "The US is responsible for attacks" + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "neutral", "score":0.9}'
    },
    {
        "role": "user",
        "content": "Check if the following story: '" +
                   "Kremlin and Trump aides raise nuclear war fears after Ukraine drone strike" +
                   "' is part of the narrative: '" + "The US was behind attack" + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "neutral", "score":0.9}'
    },
    {
        "role": "user",
        "content": "Check if the following story: '" +
                   "People enjoy eating insects" +
                   "' is part of the narrative: '" + "People are being forced to eat insects" + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "neutral", "score":0.9}'
    }
]

# ── LLM Judge: classify text as FAKE or REAL given hierarchical context ──
SYS_PROMPT_JUDGE = """You are a fact-checker. You classify whether a news article is spreading disinformation.

You will see:
- A numbered list of known disinformation narratives (propaganda claims).
- Factual news context (real reporting on related topics).
- An article to classify.

Output exactly one word: FAKE or REAL.

FAKE = the article ASSERTS one of the propaganda claims. It makes the same
       claim, promotes the same framing, or echoes the narrative.

REAL = the article does NOT assert any propaganda claim. This includes:
       (a) DENIAL — the article negates a narrative. Narrative says "X is Y"
           and the article says "X is not Y". A denial REFUTES propaganda;
           it does NOT support it.
       (b) UNRELATED — the article is on a different topic.
       (c) FACTUAL — the article reports real news on a similar topic
           without asserting any propaganda claim.

Two pitfalls to avoid:
- Sharing a topic, keyword, or entity name with a narrative is NOT support.
  Only making the same claim is.
- A negation ("X is not Y") is NOT a propaganda claim of its own just
  because the candidate list contains other negation-form narratives.
  It is a refutation of the matching narrative → REAL."""

USER_PROMPT_JUDGE = '''Disinformation narratives:
{fake_narratives}

Factual news context:
{real_section}

Article to classify:
"{text}"

Does this article assert any of the propaganda claims above? Respond with one word: FAKE or REAL.'''

EXAMPLES_JUDGE = [
    # Example 1 — ENDORSEMENT → FAKE.
    # The article asserts the same claim as narrative 1 (linking 5G to
    # spreading coronavirus). Classic propaganda promotion.
    {
        "role": "user",
        "content": ('Disinformation narratives:\n'
                    'Disinformation narrative 1: "COVID-19 has technological origins"\n'
                    'Disinformation narrative 2: "Vaccines are dangerous and cause autism"\n'
                    'Disinformation narrative 3: "Big pharma hides side effects"\n\n'
                    'Factual news context:\n'
                    '- "Telecommunications companies expand 5G coverage" (score: 1.10)\n'
                    '  Theme: "Technology infrastructure development"\n\n'
                    'Article to classify:\n'
                    '"5G towers are spreading coronavirus to nearby populations"\n\n'
                    'Does this article assert any of the propaganda claims above? Respond with one word: FAKE or REAL.')
    },
    {
        "role": "assistant",
        "content": "FAKE"
    },
    # Example 2 — UNRELATED FACTUAL REPORTING → REAL.
    # The article reports a real WHO recommendation; it doesn't assert any
    # of the listed anti-vaccine claims (does not say vaccines are unsafe,
    # alter DNA, or that side effects are hidden). Topic overlap only.
    {
        "role": "user",
        "content": ('Disinformation narratives:\n'
                    'Disinformation narrative 1: "Pharmaceutical companies hide vaccine side effects"\n'
                    'Disinformation narrative 2: "COVID vaccines alter human DNA"\n'
                    'Disinformation narrative 3: "Vaccines are unsafe"\n\n'
                    'Factual news context:\n'
                    '- "Health authorities recommend COVID-19 vaccination" (score: 2.90)\n'
                    '  Theme: "Public health vaccination campaigns"\n'
                    '- "Clinical trials confirm vaccine efficacy" (score: 2.40)\n'
                    '  Theme: "Scientific evidence supports vaccine safety"\n\n'
                    'Article to classify:\n'
                    '"WHO recommends booster shots for vulnerable populations ahead of winter"\n\n'
                    'Does this article assert any of the propaganda claims above? Respond with one word: FAKE or REAL.')
    },
    {
        "role": "assistant",
        "content": "REAL"
    },
    # Example 3 — DENIAL → REAL.
    # Narrative 1 says "John is the mayor of Springfield". The article says
    # "John is not the mayor of Springfield" — same entity, same predicate,
    # NEGATED. A denial of a propaganda claim is a REFUTATION, not support.
    # Candidate 2 ("The election results are not legitimate") is itself a
    # negation-form narrative; the article does NOT support it. Teaches the
    # model that the article being structurally similar to a negation-form
    # candidate doesn't make it propaganda.
    {
        "role": "user",
        "content": ('Disinformation narratives:\n'
                    'Disinformation narrative 1: "John is the mayor of Springfield"\n'
                    'Disinformation narrative 2: "The election results are not legitimate"\n'
                    'Disinformation narrative 3: "Government officials are hiding the truth"\n\n'
                    'Factual news context:\n'
                    '- "Springfield announces mayoral election results" (score: -1.20)\n'
                    '  Theme: "Local government"\n\n'
                    'Article to classify:\n'
                    '"John is not the mayor of Springfield"\n\n'
                    'Does this article assert any of the propaganda claims above? Respond with one word: FAKE or REAL.')
    },
    {
        "role": "assistant",
        "content": "REAL"
    }
]

# ── Truth / factual news prompts ──
SYS_PROMPT_NARRATIVE_TRUTH = """You extract the common factual news theme from a list of statements.
Rules:
- Respond with ONLY a single short sentence - the factual theme itself.
- The theme must be a verifiable factual claim or event reported in real news.
- No explanation, no analysis, no bullet points, no follow-up questions.
- If there is no common theme, respond with exactly: no narrative"""

USER_PROMPT_NARRATIVE_TRUTH = '''Statements:
{joined_statements}

Common factual news theme (one sentence only):'''

EXAMPLES_NARRATIVE_TRUTH = [
    {
        "role": "user",
        "content": "Statements:\n"
                   "WHO recommends COVID-19 vaccination for all eligible adults.\n"
                   "Clinical trials show mRNA vaccines are 95% effective against severe illness.\n\n"
                   "Common factual news theme (one sentence only):"
    },
    {
        "role": "assistant",
        "content": "COVID-19 vaccines are effective and recommended by health authorities"
    },
    {
        "role": "user",
        "content": "Statements:\n"
                   "EU leaders agree on new sanctions package targeting Russian oil imports.\n"
                   "European Council approves sixth round of sanctions against Russia.\n\n"
                   "Common factual news theme (one sentence only):"
    },
    {
        "role": "assistant",
        "content": "The EU imposed sanctions on Russia over the Ukraine conflict"
    },
    {
        "role": "user",
        "content": "Statements:\n"
                   "Local bakery wins award for best sourdough bread.\n"
                   "NASA confirms discovery of water on Mars surface.\n\n"
                   "Common factual news theme (one sentence only):"
    },
    {
        "role": "assistant",
        "content": "no narrative"
    }
]

EXAMPLES_NARRATIVE = [
    {
        "role": "user",
        "content": "Statements:\n" +
                   "Chlorine dioxide has already proved efficacious as a mouthwash to decreases the load of the "
                   "covid-19 virus. Bolivia has used it as a safe and effective treatment to eradicate the virus. "
                   "Please can we do a Clinical trial in India\n"
                   "Bolivia approved the use of chlorine dioxide amid the fight against covid-19.\n\n"
                   "Common disinformation narrative (one sentence only):"
    },
    {
        "role": "assistant",
        "content": "Chlorine dioxide is an effective COVID-19 treatment"
    },
    {
        "role": "user",
        "content": "Statements:\n"
                   "Zelenski risks suspension for mental issues\n"
                   "US requests Zelenski's conviction to life imprisonment\n\n"
                   "Common disinformation narrative (one sentence only):"
    },
    {
        "role": "assistant",
        "content": "Zelenski is unfit to govern"
    },
    {
        "role": "user",
        "content": "Statements:\n"
                   "The EU is obsessed with anti-Russian sanctions.\n"
                   "Multiple posts on Facebook and Twitter claim the WHO has warned against eating cabbage\n\n"
                   "Common disinformation narrative (one sentence only):"
    },
    {
        "role": "assistant",
        "content": "no narrative"
    }
]

