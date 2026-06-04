"""
Question Schema v2
==================
Improved question format with:
- Separate symptom tags vs condition tags
- Demographic filters (age, gender)
- Context metadata (duration, history, lifestyle)
- Multilingual support
- Backward-compatible with v1 questions

Schema:
{
    "id":                str,         # unique ID e.g. "Q001"
    "text":              str,         # English question text
    "translations":      dict,        # {"hi": "...", "gu": "...", "kn": "...", ...}
    "elicits":           str,         # the symptom/finding this probes

    # --- SEPARATED TAGS (the key design decision) ---
    "condition_tags":    list[str],   # conditions this helps discriminate e.g. ["influenza", "pneumonia"]
    "symptom_tags":      list[str],   # symptoms that make this question relevant e.g. ["cough", "fever"]
                                      # → engine uses this to check: "only ask if these symptoms are active"

    # --- DEMOGRAPHIC FILTERS ---
    "gender":            str|None,    # "male", "female", or None (ask anyone)
    "age_min":           int|None,    # minimum age (inclusive), None = no limit
    "age_max":           int|None,    # maximum age (inclusive), None = no limit

    # --- CONTEXT METADATA ---
    "context_tags":      list[str],   # free tags: "duration", "history", "lifestyle", "diet",
                                      #            "medication", "family_history", "travel", "occupation"
    "requires_symptoms": list[str],   # only ask if ALL of these symptoms are already active
                                      # e.g. ["fever"] means don't ask unless fever is confirmed
    "excludes_symptoms": list[str],   # skip this question if ANY of these are active
                                      # e.g. if patient already denied cough, don't ask about cough type

    # --- RESPONSE HANDLING ---
    "response_map":      dict,        # keyword → symptom activations (same as v1)
    "response_type":     str,         # "yes_no", "duration", "free_text", "choice"
}
"""

# ─────────────────────────────────────────────
# v1 → v2 CONVERTER
# ─────────────────────────────────────────────
def convert_v1_to_v2(v1_question: dict) -> dict:
    """
    Convert a v1 question (from current knowledge_base.py) to v2 format.
    v1 has: id, text, elicits, tags, response_map
    v2 adds: condition_tags, symptom_tags, gender, age_min/max, context_tags,
             requires_symptoms, excludes_symptoms, response_type, translations
    """
    # Detect response_type from response_map
    rmap = v1_question.get("response_map", {})
    if rmap.get("duration") == True:
        response_type = "duration"
    elif all(k in ("yes", "no") for k in rmap.keys() if k != "duration"):
        response_type = "yes_no"
    else:
        response_type = "free_text"

    # Infer context_tags from question text
    text_lower = v1_question.get("text", "").lower()
    context_tags = []
    if any(w in text_lower for w in ["how long", "how many days", "duration", "when did"]):
        context_tags.append("duration")
    if any(w in text_lower for w in ["history", "family", "anyone else"]):
        context_tags.append("history")
    if any(w in text_lower for w in ["smoke", "alcohol", "drink", "drug", "painkiller"]):
        context_tags.append("lifestyle")
    if any(w in text_lower for w in ["travel", "region"]):
        context_tags.append("travel")
    if any(w in text_lower for w in ["diet", "food", "eat"]):
        context_tags.append("diet")
    if any(w in text_lower for w in ["medication", "medicine", "ibuprofen", "aspirin"]):
        context_tags.append("medication")

    return {
        "id": v1_question["id"],
        "text": v1_question["text"],
        "translations": {},     # to be filled from multilingual corpus
        "elicits": v1_question["elicits"],

        "condition_tags": list(v1_question.get("tags", [])),  # v1 "tags" were conditions
        "symptom_tags": [],     # to be filled based on elicits

        "gender": None,
        "age_min": None,
        "age_max": None,

        "context_tags": context_tags,
        "requires_symptoms": [],
        "excludes_symptoms": [],

        "response_map": rmap,
        "response_type": response_type,
    }


def convert_multilingual_entry(symptom_key: str, entry: dict, question_id: str) -> dict:
    """
    Convert ONE entry from the multilingual corpus to v2 format.

    The multilingual corpus looks like:
    {
        "hi": "...", "en": "...", "gu": "...", "te": "...", "mr": "...", "kn": "...",
        "category": "urinary frequency",
        "symptom": "urinary frequency",
        "risk_factor": False
    }
    """
    # Build translations dict (exclude metadata fields)
    lang_keys = {"hi", "en", "gu", "te", "mr", "kn", "ta", "ml", "bn", "pa", "or"}
    translations = {}
    for key in lang_keys:
        if key in entry and key != "en":
            translations[key] = entry[key]

    en_text = entry.get("en", "")
    elicited_symptom = entry.get("symptom") or entry.get("category", "unknown")
    # Normalize: lowercase, replace spaces with underscores
    elicited_symptom = elicited_symptom.strip().lower().replace(" ", "_").replace(":", "_")

    # Detect response_type from text
    text_lower = en_text.lower()
    if any(w in text_lower for w in ["how long", "how many days", "since when"]):
        response_type = "duration"
    elif text_lower.startswith(("are you", "do you", "have you", "is there", "did you", "does ")):
        response_type = "yes_no"
    else:
        response_type = "free_text"

    # Detect context tags
    context_tags = []
    if any(w in text_lower for w in ["how long", "how many days", "since when", "duration"]):
        context_tags.append("duration")
    if any(w in text_lower for w in ["history", "family", "anyone else"]):
        context_tags.append("history")
    if any(w in text_lower for w in ["smoke", "alcohol", "drink", "drug"]):
        context_tags.append("lifestyle")
    if any(w in text_lower for w in ["travel", "region"]):
        context_tags.append("travel")
    if any(w in text_lower for w in ["diet", "food", "eat"]):
        context_tags.append("diet")
    if any(w in text_lower for w in ["medication", "medicine"]):
        context_tags.append("medication")
    if entry.get("risk_factor"):
        context_tags.append("risk_factor")

    # Build basic response_map for yes/no questions
    response_map = {}
    if response_type == "yes_no":
        response_map = {"yes": [elicited_symptom]}
    elif response_type == "duration":
        response_map = {"duration": True}

    return {
        "id": question_id,
        "text": en_text,
        "translations": translations,
        "elicits": elicited_symptom,

        "condition_tags": [],       # must be filled with domain knowledge
        "symptom_tags": [symptom_key.lower().replace(" ", "_")],

        "gender": None,
        "age_min": None,
        "age_max": None,

        "context_tags": context_tags,
        "requires_symptoms": [symptom_key.lower().replace(" ", "_")],
        "excludes_symptoms": [],

        "response_map": response_map,
        "response_type": response_type,
    }


def batch_convert_multilingual_corpus(corpus: dict, start_id: int = 200) -> list:
    """
    Convert an entire multilingual corpus dict.

    corpus format:
    {
        "excessive thirst": [ {entry1}, {entry2}, ... ],
        "fever": [ {entry1}, {entry2}, ... ],
        ...
    }

    Returns list of v2 question dicts.
    """
    questions = []
    qid = start_id
    for symptom_key, entries in corpus.items():
        for entry in entries:
            question_id = f"Q{qid:04d}"
            v2 = convert_multilingual_entry(symptom_key, entry, question_id)
            questions.append(v2)
            qid += 1
    return questions
