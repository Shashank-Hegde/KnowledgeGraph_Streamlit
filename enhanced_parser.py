"""
Enhanced Response Parser (v3 — LLM-integrated)
===============================================
Hybrid parser that routes between:
  - Fast path: regex + keyword matching (for "yes", "no", "3 days")
  - LLM path: structured extraction (for complex free-text responses)

The routing is automatic based on input complexity.
Falls back to fast path if LLM is unavailable.
"""

import re
from typing import Optional
from graph_engine import KnowledgeGraph
from llm_extractor import LLMExtractor
from entity_mapper import EntityMapper
from smart_resolver import SmartSymptomResolver


# Negation detection (reused from v2 parser)
NEGATION_CUES = {"no", "not", "without", "don't", "dont", "doesn't", "doesnt",
                 "isn't", "isnt", "never", "none", "nor", "neither", "absent",
                 "deny", "denies", "negative", "nahi", "nah"}


def _is_negated(text, match_start, window=4):
    before = text[:match_start].strip()
    if not before:
        return False
    preceding_words = before.split()[-window:]
    for word in preceding_words:
        clean = word.strip(".,;:!?'\"")
        if clean in NEGATION_CUES:
            return True
    return False


class EnhancedResponseParser:
    """
    Hybrid parser: fast path for simple inputs, LLM for complex ones.
    """

    def __init__(self, graph, llm_extractor=None):
        # type: (KnowledgeGraph, Optional[LLMExtractor]) -> None
        self.graph = graph
        self.llm = llm_extractor or LLMExtractor()
        self.mapper = EntityMapper(graph)
        self.smart = SmartSymptomResolver(graph)

    def parse(self, raw_input, current_question=None, is_initial=False):
        # type: (str, Optional[dict], bool) -> dict
        """
        Parse patient response using the appropriate strategy.

        Args:
            raw_input: patient's text
            current_question: the question that was asked (None for initial)
            is_initial: True for iteration 0 (chief complaint)

        Returns:
            {
                "is_null": bool,
                "is_yes": bool,
                "is_no": bool,
                "detected_symptoms": [],    # present symptoms (canonical keys)
                "negated_symptoms": [],     # absent symptoms
                "duration": None or {"value": int, "unit": str},
                "body_parts": [],           # ["chest", "lungs"]
                "body_part_categories": [], # ["cardiology", "pulmonology"]
                "medications_found": [],    # medication findings activated
                "history_found": [],        # history findings activated
                "metadata": {},             # severity, onset, LLM latency
                "matched_findings": [],     # from response_map keywords
                "raw": str,
                "parse_method": str,        # "llm" or "fast"
            }
        """
        text = raw_input.strip()

        # Empty input
        if not text:
            return self._empty_result(raw_input)

        # Decide: LLM or fast path?
        use_llm = (
            is_initial or  # always use LLM for chief complaint
            (self.llm.should_use_llm(text) and self.llm.is_available())
        )

        if use_llm:
            q_text = current_question.get("text", "") if current_question else None
            result = self._llm_parse(text, q_text, is_initial, current_question)
            if result is not None:
                return result
            # LLM failed — fall through to fast path

        return self._fast_parse(text, current_question)

    # ─────────────────────────────────────────
    # LLM PATH
    # ─────────────────────────────────────────
    def _llm_parse(self, text, question_text, is_initial, current_question):
        """Use LLM for structured extraction."""
        llm_output = self.llm.extract(text, question_text, is_initial)
        if llm_output is None:
            return None

        # Map LLM output → KG activations
        mapped = self.mapper.map_extraction(llm_output)
        if mapped is None:
            return None

        result = {
            "is_null": False,
            "is_yes": mapped.get("is_yes") or False,
            "is_no": mapped.get("is_no") or False,
            "detected_symptoms": [key for key, _ in mapped["activate"]],
            "negated_symptoms": mapped["negate"],
            "duration": mapped["duration"],
            "body_parts": mapped["body_parts"],
            "body_part_categories": mapped["body_part_categories"],
            "medications_found": [k for k, _ in mapped["activate"] if k.endswith("_use") or k.endswith("_medication")],
            "history_found": [k for k, _ in mapped["activate"] if k.startswith("family_") or k.endswith("_history")],
            "metadata": mapped.get("metadata", {}),
            "matched_findings": [],
            "raw": text,
            "parse_method": "llm",
            "_activation_strengths": {key: strength for key, strength in mapped["activate"]},
        }

        # Also run response_map matching from the current question
        if current_question:
            result["matched_findings"] = self._match_response_map(text, current_question)

        # Smart resolver safety net: catch symptoms LLM missed
        smart_results = self.smart.resolve(text)
        for sym, pri in smart_results:
            if sym not in result["detected_symptoms"] and sym not in result["negated_symptoms"]:
                result["detected_symptoms"].append(sym)

        # Fix yes/no if LLM didn't detect but keywords are present
        if not result["is_yes"] and not result["is_no"]:
            result["is_yes"], result["is_no"] = self._detect_yes_no(text)

        result["metadata"]["_llm_latency_ms"] = llm_output.get("_llm_latency_ms", 0)
        return result

    # ─────────────────────────────────────────
    # FAST PATH (regex + keywords, no LLM)
    # ─────────────────────────────────────────
    def _fast_parse(self, text, current_question):
        """Fast keyword/regex parser for simple inputs."""
        text_lower = text.lower()

        is_yes, is_no = self._detect_yes_no(text_lower)

        result = {
            "is_null": False,
            "is_yes": is_yes,
            "is_no": is_no,
            "detected_symptoms": [],
            "negated_symptoms": [],
            "duration": None,
            "body_parts": [],
            "body_part_categories": [],
            "medications_found": [],
            "history_found": [],
            "metadata": {},
            "matched_findings": [],
            "raw": text,
            "parse_method": "fast",
            "_activation_strengths": {},
        }

        # Duration detection
        duration_match = re.search(
            r'(\d+)\s*(day|days|week|weeks|month|months|hour|hours|year|years|din|hafta|mahina)',
            text_lower
        )
        if duration_match:
            value = int(duration_match.group(1))
            unit = duration_match.group(2)
            # Normalize unit
            if unit.startswith("day") or unit == "din":
                unit = "days"
            elif unit.startswith("week") or unit == "hafta":
                unit = "weeks"
            elif unit.startswith("month") or unit == "mahina":
                unit = "months"
            elif unit.startswith("year"):
                unit = "years"
            elif unit.startswith("hour"):
                unit = "hours"
            result["duration"] = {"value": value, "unit": unit}

        # Symptom detection with negation
        for alias, canonical in self.graph.aliases.items():
            idx = text_lower.find(alias)
            if idx == -1:
                continue
            if _is_negated(text_lower, idx):
                if canonical not in result["negated_symptoms"]:
                    result["negated_symptoms"].append(canonical)
            else:
                if canonical not in result["detected_symptoms"]:
                    result["detected_symptoms"].append(canonical)

        # Remove overlap (negated wins)
        overlap = set(result["detected_symptoms"]) & set(result["negated_symptoms"])
        if overlap:
            result["detected_symptoms"] = [s for s in result["detected_symptoms"] if s not in overlap]

        # Smart resolver: catches body-part combos, typos, regional terms
        # that alias matching misses (e.g., "pain in the leg" → body_pain)
        smart_results = self.smart.resolve(text_lower)
        for sym, pri in smart_results:
            if sym not in result["detected_symptoms"] and sym not in result["negated_symptoms"]:
                result["detected_symptoms"].append(sym)

        # Response map matching
        if current_question:
            result["matched_findings"] = self._match_response_map(text_lower, current_question)

        return result

    # ─────────────────────────────────────────
    # SHARED HELPERS
    # ─────────────────────────────────────────
    def _detect_yes_no(self, text):
        """Detect yes/no from text, including indirect patterns."""
        text_lower = text.lower().strip()
        # Fix common typos: remove isolated punctuation inside words
        text_clean = text_lower.replace(";", "").replace("'", "")
        words = set(text_clean.split())

        yes_patterns = {"yes", "yeah", "ya", "yep", "haan", "ha", "correct",
                       "right", "true", "sure", "sometimes", "occasionally",
                       "once", "twice", "definitely", "absolutely", "always"}
        no_patterns = {"no", "nah", "nope", "nahi", "none", "nothing", "never",
                      "not", "nope", "nahin", "nai"}

        # Indirect "no" phrases
        no_phrases = ["not really", "not at all", "don't think so", "seems ok",
                     "all ok", "seems fine", "all fine", "all good", "seems normal",
                     "not like that", "nothing like that", "no such", "haven't",
                     "doesn't seem", "i don't", "did not", "have not",
                     "not there", "isn't there", "is not there"]

        # Indirect "yes" phrases
        yes_phrases = ["it is there", "it is in", "there is", "i have", "i do",
                      "it seems", "a little", "a bit",
                      "last time", "last week", "last month",
                      "week ago", "weeks ago", "days ago", "month ago",
                      "months ago", "years ago", "year ago",
                      "once did", "used to", "i did", "went to",
                      "it comes", "it started", "it happens",
                      "in the side", "on the side", "in the back",
                      "for a while", "for some time", "since"]

        is_yes = bool(words.intersection(yes_patterns))
        is_no = bool(words.intersection(no_patterns))

        # Check phrases (phrases override single-word matches for disambiguation)
        if not is_no:
            for phrase in no_phrases:
                if phrase in text_lower:
                    is_no = True
                    is_yes = False  # "not there" should NOT be yes
                    break

        if not is_yes and not is_no:
            for phrase in yes_phrases:
                if phrase in text_lower:
                    is_yes = True
                    break

        # Handle conflict: if both yes and no detected, check which came first
        # "not there" → no wins. "yes but not much" → yes wins.
        if is_yes and is_no:
            # Check if response starts with negation
            first_words = text_lower.split()[:2]
            if any(w in no_patterns for w in first_words):
                is_yes = False  # "not there", "no really" → no
            elif any(w in yes_patterns for w in first_words):
                is_no = False   # "yes but...", "sometimes not" → yes

        return is_yes, is_no

    def _match_response_map(self, text, question):
        """Match keywords from question's response_map."""
        text_lower = text.lower()
        findings = []
        rmap = question.get("response_map", {})
        for keyword, mapping in rmap.items():
            if keyword == "duration":
                continue
            if keyword in text_lower:
                if _is_negated(text_lower, text_lower.find(keyword)):
                    continue
                if isinstance(mapping, list):
                    findings.extend(mapping)
                elif isinstance(mapping, str):
                    findings.append("hint_" + mapping)

        # Also handle "yes" key in response_map
        is_yes, _ = self._detect_yes_no(text_lower)
        if is_yes and "yes" in rmap:
            if isinstance(rmap["yes"], list):
                for f in rmap["yes"]:
                    if f not in findings:
                        findings.append(f)

        return list(set(findings))

    def _empty_result(self, raw):
        return {
            "is_null": True, "is_yes": False, "is_no": True,
            "detected_symptoms": [], "negated_symptoms": [],
            "duration": None, "body_parts": [], "body_part_categories": [],
            "medications_found": [], "history_found": [],
            "metadata": {}, "matched_findings": [],
            "raw": raw.strip(), "parse_method": "fast",
            "_activation_strengths": {},
        }
