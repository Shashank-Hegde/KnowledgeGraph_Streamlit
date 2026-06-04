"""
Response Parser
===============
Parses patient responses to extract:
- New symptom mentions
- Yes/No answers
- Duration information
- Keyword matches from question response_maps

v2: Added negation detection — "no fever", "without headache", "there is no cough"
    are now correctly treated as DENIED symptoms, not confirmed ones.
"""

import re
from typing import Optional
from graph_engine import KnowledgeGraph


# Negation window: if any of these words appear within N words BEFORE the
# symptom mention, treat it as negated.
NEGATION_CUES = {"no", "not", "without", "don't", "dont", "doesn't", "doesnt",
                 "isn't", "isnt", "never", "none", "nor", "neither", "absent",
                 "deny", "denies", "negative"}


def _is_negated(text: str, match_start: int, window: int = 4) -> bool:
    """
    Check whether the symptom at position `match_start` in `text`
    is preceded (within `window` words) by a negation cue.

    Example:
        text = "yes but there is no fever"
        match_start = index of "fever"
        → "no" is 1 word before → returns True
    """
    # Get the portion of text BEFORE the match
    before = text[:match_start].strip()
    if not before:
        return False

    # Take the last `window` words before the symptom
    preceding_words = before.split()[-window:]
    for word in preceding_words:
        # Clean punctuation
        clean = word.strip(".,;:!?'\"")
        if clean in NEGATION_CUES:
            return True
    return False


class ResponseParser:

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def parse(
        self,
        raw_input: str,
        current_question: Optional[dict] = None,
    ) -> dict:
        """
        Parse a patient's response.

        Returns:
            {
                "is_null": bool,           # True if empty input (pressed enter)
                "is_yes": bool,
                "is_no": bool,
                "detected_symptoms": [],   # canonical symptom keys found (PRESENT)
                "negated_symptoms": [],    # canonical symptom keys found (NEGATED)
                "duration": None or str,   # e.g. "3 days", "1 week"
                "raw": str,
                "matched_findings": [],    # findings from response_map
            }
        """
        text = raw_input.strip().lower()
        result = {
            "is_null": len(text) == 0,
            "is_yes": False,
            "is_no": False,
            "detected_symptoms": [],
            "negated_symptoms": [],
            "duration": None,
            "raw": raw_input.strip(),
            "matched_findings": [],
        }

        if result["is_null"]:
            result["is_no"] = True  # null = no/skip
            return result

        # --- Yes/No detection ---
        yes_patterns = {"yes", "yeah", "ya", "yep", "haan", "ha", "correct", "right", "true", "sure"}
        no_patterns = {"no", "nah", "nope", "nahi", "none", "nothing"}
        # "not really" as a phrase
        not_really = "not really" in text

        words = set(text.split())
        if words.intersection(yes_patterns):
            result["is_yes"] = True
        if words.intersection(no_patterns) or not_really:
            result["is_no"] = True

        # --- Duration detection ---
        duration_match = re.search(
            r'(\d+)\s*(day|days|week|weeks|month|months|hour|hours|year|years|din|hafta|mahina)',
            text
        )
        if duration_match:
            result["duration"] = duration_match.group(0)

        # --- Symptom detection from aliases (WITH negation check) ---
        for alias, canonical in self.graph.aliases.items():
            # Find the alias in the text
            idx = text.find(alias)
            if idx == -1:
                continue

            if _is_negated(text, idx):
                # Symptom is mentioned but NEGATED
                if canonical not in result["negated_symptoms"]:
                    result["negated_symptoms"].append(canonical)
            else:
                if canonical not in result["detected_symptoms"]:
                    result["detected_symptoms"].append(canonical)

        # Also check direct symptom keys (e.g., "body_pain" → "body pain")
        for sym_key in self.graph.symptoms:
            display = sym_key.replace("_", " ")
            idx = text.find(display)
            if idx == -1:
                continue

            if _is_negated(text, idx):
                if sym_key not in result["negated_symptoms"]:
                    result["negated_symptoms"].append(sym_key)
            else:
                if sym_key not in result["detected_symptoms"]:
                    result["detected_symptoms"].append(sym_key)

        # Remove any symptom that appears in BOTH lists (negated wins — safer)
        overlap = set(result["detected_symptoms"]) & set(result["negated_symptoms"])
        if overlap:
            result["detected_symptoms"] = [s for s in result["detected_symptoms"] if s not in overlap]

        # --- Match against current question's response_map ---
        if current_question:
            rmap = current_question.get("response_map", {})
            for keyword, mapping in rmap.items():
                if keyword == "duration":
                    continue  # already handled
                if keyword in text:
                    # Check if this keyword is also negated
                    kidx = text.find(keyword)
                    if _is_negated(text, kidx):
                        continue  # skip negated keyword matches

                    if isinstance(mapping, list):
                        result["matched_findings"].extend(mapping)
                    elif isinstance(mapping, str):
                        result["matched_findings"].append("hint_" + mapping)

            # If user said yes and there's a "yes" in response_map
            if result["is_yes"] and "yes" in rmap:
                if isinstance(rmap["yes"], list):
                    for finding in rmap["yes"]:
                        if finding not in result["matched_findings"]:
                            result["matched_findings"].append(finding)

        # Deduplicate
        result["detected_symptoms"] = list(set(result["detected_symptoms"]))
        result["negated_symptoms"] = list(set(result["negated_symptoms"]))
        result["matched_findings"] = list(set(result["matched_findings"]))

        return result
