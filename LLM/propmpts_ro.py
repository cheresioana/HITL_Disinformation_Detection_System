# ═══════════════════════════════════════════════════════════════
# Prompturi în limba română pentru algoritmul de narativ
# Versiune identică cu propmpts.py dar tradusă în română
# Tokenii de răspuns (FAKE, REAL, no narrative) rămân în engleză
# pentru compatibilitate cu codul Python existent.
# ═══════════════════════════════════════════════════════════════

# ── Extragerea narativului de dezinformare ──

SYS_PROMPT_NARRATIVE = """Extragi narativul comun dintr-o listă de afirmații.
Reguli:
- Răspunde DOAR cu o singură propoziție scurtă - narativul propriu-zis.
- Narativul trebuie să fie o afirmație care poate fi verificată prin implicație textuală.
- Fără explicații, fără analiză, fără liste, fără întrebări suplimentare.
- Dacă nu există un narativ comun, răspunde exact cu: no narrative"""

USER_PROMPT_NARRATIVE = '''Afirmații:
{joined_statements}

Narativul comun de dezinformare (o singură propoziție):'''

EXAMPLES_NARRATIVE = [
    {
        "role": "user",
        "content": "Afirmații:\n"
                   "Dioxidul de clor s-a dovedit deja eficient ca apă de gură pentru a reduce "
                   "încărcătura virusului COVID-19. Bolivia l-a folosit ca tratament sigur și eficient "
                   "pentru a eradica virusul.\n"
                   "Bolivia a aprobat utilizarea dioxidului de clor în lupta împotriva COVID-19.\n\n"
                   "Narativul comun de dezinformare (o singură propoziție):"
    },
    {
        "role": "assistant",
        "content": "Dioxidul de clor este un tratament eficient pentru COVID-19"
    },
    {
        "role": "user",
        "content": "Afirmații:\n"
                   "Zelenski riscă suspendarea pentru probleme de sănătate mintală\n"
                   "SUA solicită condamnarea lui Zelenski la închisoare pe viață\n\n"
                   "Narativul comun de dezinformare (o singură propoziție):"
    },
    {
        "role": "assistant",
        "content": "Zelenski este inapt să conducă"
    },
    {
        "role": "user",
        "content": "Afirmații:\n"
                   "UE este obsedată de sancțiunile anti-rusești.\n"
                   "Mai multe postări pe Facebook și Twitter susțin că OMS a avertizat împotriva consumului de varză\n\n"
                   "Narativul comun de dezinformare (o singură propoziție):"
    },
    {
        "role": "assistant",
        "content": "no narrative"
    }
]

# ── Clasificare entailment ──

SYS_PROMPT_ENT = '''
Ești un clasificator precis de implicație politică. Răspunde cu una dintre: {\"label\": \"entailment\", \"score\": număr între 0 și 1},{\"label\": \"contradiction\", \"score\": număr între 0 și 1}, {\"label\": \"neutral\", \"score\": număr între 0 și 1}."
'''

USER_PROMPT_ENT = """
"Verifică dacă următoarea știre: '{headline}' face parte din narativul: '{narrative}'"
"""

EXAMPLES_ENT = [
    {
        "role": "user",
        "content": "Verifică dacă următoarea știre: '"
                   "Zelenski a cumpărat reședința lui Hitler"
                   "' face parte din narativul: '"
                   "Zelenski are legături cu naziștii." + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "entailment", "score":0.9}'
    },
    {
        "role": "user",
        "content": "Verifică dacă următoarea știre: '"
                   "Zelenski a cumpărat reședința lui Hitler"
                   "' face parte din narativul: '"
                   "Polonia vrea să atace Rusia" + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "neutral", "score":0.9}'
    },
    {
        "role": "user",
        "content": "Verifică dacă următoarea știre: '"
                   "Președintele Poloniei a recunoscut că Varșovia va anexa teritorii ucrainene"
                   "' face parte din narativul: '"
                   "Polonia vrea să atace Rusia și să ocupe vestul Ucrainei" + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "entailment", "score":0.9}'
    },
    {
        "role": "user",
        "content": "Verifică dacă următoarea știre: '"
                   "Kremlinul și consilierii lui Trump ridică temeri de război nuclear după atacul cu drone al Ucrainei"
                   "' face parte din narativul: '"
                   "SUA este responsabilă pentru atacuri" + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "neutral", "score":0.9}'
    },
    {
        "role": "user",
        "content": "Verifică dacă următoarea știre: '"
                   "Kremlinul și consilierii lui Trump ridică temeri de război nuclear după atacul cu drone al Ucrainei"
                   "' face parte din narativul: '"
                   "SUA a stat în spatele atacului" + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "neutral", "score":0.9}'
    },
    {
        "role": "user",
        "content": "Verifică dacă următoarea știre: '"
                   "Oamenii se bucură să mănânce insecte"
                   "' face parte din narativul: '"
                   "Oamenii sunt forțați să mănânce insecte" + "'"
    },
    {
        "role": "assistant",
        "content": '{"label": "neutral", "score":0.9}'
    }
]

# ── Judecător LLM: clasificare articol ca FAKE sau REAL ──

SYS_PROMPT_JUDGE = """Ești un detector de narativ de dezinformare.
Reguli:
- Vei primi un narativ de dezinformare cunoscut și un articol de știri.
- Determină dacă afirmațiile articolului SUSȚIN sau SE ALINIAZĂ cu narativul de dezinformare.
- Dacă articolul promovează, repetă sau întărește narativul, răspunde FAKE.
- Dacă articolul este o relatare factuală care doar menționează un subiect similar, răspunde REAL.
- Răspunde cu UN SINGUR cuvânt: FAKE sau REAL
- Dacă nu ești sigur, răspunde REAL"""

USER_PROMPT_JUDGE = '''Narativ de dezinformare:
"{fake_narrative}"

Dovezi din știri false cunoscute:
{fake_section}

Context din știri factuale:
{real_section}

Articol de clasificat:
"{text}"

Acest articol susține narativul de dezinformare de mai sus, sau este o relatare factuală? FAKE sau REAL?'''

EXAMPLES_JUDGE = [
    {
        "role": "user",
        "content": ('Narativ de dezinformare:\n'
                    '"COVID-19 are origini tehnologice"\n\n'
                    'Dovezi din știri false cunoscute:\n'
                    '- "Tehnologia 5G cauzează COVID-19" (scor: 3.20)\n'
                    '- "Radiațiile 5G slăbesc sistemul imunitar" (scor: 2.10)\n\n'
                    'Context din știri factuale:\n'
                    '- "Companiile de telecomunicații extind acoperirea 5G" (scor: 1.10)\n'
                    '  Temă: "Dezvoltarea infrastructurii tehnologice"\n'
                    '- "OMS investighează originile COVID-19" (scor: 0.70)\n'
                    '  Temă: "Răspunsul internațional de sănătate la COVID-19"\n\n'
                    'Articol de clasificat:\n'
                    '"Turnurile 5G răspândesc coronavirusul în populațiile din apropiere"\n\n'
                    'Acest articol susține narativul de dezinformare de mai sus, sau este o relatare factuală? FAKE sau REAL?')
    },
    {
        "role": "assistant",
        "content": "FAKE"
    },
    {
        "role": "user",
        "content": ('Narativ de dezinformare:\n'
                    '"Companiile farmaceutice ascund efectele secundare ale vaccinurilor"\n\n'
                    'Dovezi din știri false cunoscute:\n'
                    '- "Vaccinurile sunt periculoase și cauzează moartea" (scor: 0.80)\n'
                    '- "Vaccinurile COVID modifică ADN-ul uman" (scor: 0.45)\n\n'
                    'Context din știri factuale:\n'
                    '- "Autoritățile de sănătate recomandă vaccinarea împotriva COVID-19" (scor: 2.90)\n'
                    '  Temă: "Campanii de vaccinare în sănătatea publică"\n'
                    '- "Studiile clinice confirmă eficacitatea vaccinului" (scor: 2.40)\n'
                    '  Temă: "Dovezi științifice care susțin siguranța vaccinului"\n\n'
                    'Articol de clasificat:\n'
                    '"OMS recomandă doze de rapel pentru populațiile vulnerabile înainte de iarnă"\n\n'
                    'Acest articol susține narativul de dezinformare de mai sus, sau este o relatare factuală? FAKE sau REAL?')
    },
    {
        "role": "assistant",
        "content": "REAL"
    }
]

# ── Extragerea temei factuale (pentru știri reale) ──

SYS_PROMPT_NARRATIVE_TRUTH = """Extragi tema factuală comună dintr-o listă de afirmații.
Reguli:
- Răspunde DOAR cu o singură propoziție scurtă - tema factuală propriu-zisă.
- Tema trebuie să fie o afirmație factuală verificabilă sau un eveniment raportat în știrile reale.
- Fără explicații, fără analiză, fără liste, fără întrebări suplimentare.
- Dacă nu există o temă comună, răspunde exact cu: no narrative"""

USER_PROMPT_NARRATIVE_TRUTH = '''Afirmații:
{joined_statements}

Tema factuală comună din știri (o singură propoziție):'''

EXAMPLES_NARRATIVE_TRUTH = [
    {
        "role": "user",
        "content": "Afirmații:\n"
                   "OMS recomandă vaccinarea împotriva COVID-19 pentru toți adulții eligibili.\n"
                   "Studiile clinice arată că vaccinurile ARNm sunt 95% eficiente împotriva bolilor grave.\n\n"
                   "Tema factuală comună din știri (o singură propoziție):"
    },
    {
        "role": "assistant",
        "content": "Vaccinurile COVID-19 sunt eficiente și recomandate de autoritățile de sănătate"
    },
    {
        "role": "user",
        "content": "Afirmații:\n"
                   "Liderii UE au convenit asupra unui nou pachet de sancțiuni vizând importurile de petrol rusesc.\n"
                   "Consiliul European aprobă a șasea rundă de sancțiuni împotriva Rusiei.\n\n"
                   "Tema factuală comună din știri (o singură propoziție):"
    },
    {
        "role": "assistant",
        "content": "UE a impus sancțiuni Rusiei din cauza conflictului din Ucraina"
    },
    {
        "role": "user",
        "content": "Afirmații:\n"
                   "O brutărie locală câștigă premiul pentru cea mai bună pâine cu maia.\n"
                   "NASA confirmă descoperirea apei pe suprafața lui Marte.\n\n"
                   "Tema factuală comună din știri (o singură propoziție):"
    },
    {
        "role": "assistant",
        "content": "no narrative"
    }
]
