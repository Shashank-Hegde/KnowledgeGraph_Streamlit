"""
Question Deduplication & Knowledge Accumulator
===============================================

Prevents asking the same question twice at scale, even when:
  - Two questions have different IDs but ask the same thing
  - Two questions have different wording but same category
  - The patient already volunteered the answer in a previous response

Three-layer dedup:

  Layer 1: CATEGORY DEDUP
    Every question has a normalized category (e.g., "cough_type", "nausea",
    "stress_impact"). Once a question with category X is asked, ALL other
    questions with category X are blocked.

  Layer 2: SYMPTOM DEDUP
    If a question's target symptom is already in the active or denied sets,
    skip it (already handled by the selector's _is_eligible check).

  Layer 3: KNOWLEDGE DEDUP (extra_medical_info)
    The LLM extracts rich medical details beyond yes/no — body parts,
    qualifiers, medications, history. If the answer to a question is
    already known from previously extracted info, auto-fill and skip.

Normalization rules for categories:
  - lowercase
  - strip whitespace
  - replace spaces/hyphens with underscores
  - collapse "impact: X" → "X_impact"
  - collapse "X with Y" → "X_Y"
  - remove articles (a, an, the)
  - remove "do you have", "is there", etc.
"""

import re
from typing import Optional, Set


# ─────────────────────────────────────────────
# 1. CATEGORY NORMALIZER
# ─────────────────────────────────────────────

def normalize_category(raw_category):
    """
    Normalize a question category to a canonical dedup key.

    Examples:
        "cough type"           → "cough_type"
        "Is the cough dry"     → "cough_dry"
        "stress impact: arm tremor" → "arm_tremor_stress_impact"
        "activity impact: sprain"   → "sprain_activity_impact"
        "nausea"               → "nausea"
        "Cough Type"           → "cough_type"
        "diet: thirst"         → "thirst_diet"
        "other_changes_with_thirst" → "thirst_other_changes"
    """
    if not raw_category:
        return "unknown"

    s = raw_category.strip().lower()

    # Handle "X impact: Y" or "X: Y" format → "Y_X"
    if ":" in s:
        parts = [p.strip() for p in s.split(":", 1)]
        # Reverse: "stress impact: arm tremor" → "arm_tremor_stress_impact"
        s = parts[1] + " " + parts[0]

    # Handle "X_with_Y" → "X_Y"
    s = re.sub(r'\bwith\b', '', s)
    s = re.sub(r'\bother_changes\b', 'other_changes', s)

    # Remove articles and filler words
    s = re.sub(r'\b(a|an|the|do|you|have|is|are|there|any|does|it)\b', '', s)

    # Replace spaces, hyphens, multiple underscores
    s = re.sub(r'[\s\-]+', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')

    return s or "unknown"


def category_from_question_text(question_text):
    """
    Auto-generate a category from question text when no explicit category exists.
    Uses the key medical noun phrases.

    "Is the cough dry or producing mucus/phlegm?" → "cough_dry_mucus"
    "Do you feel a burning sensation in your chest?" → "burning_chest"
    "Have you coughed up blood?" → "cough_blood"
    """
    text = question_text.lower().strip()

    # Remove question structure words
    text = re.sub(r'^(do you|have you|are you|is there|does the|is the|can you|did you)\s+', '', text)
    text = re.sub(r'\?$', '', text)

    # Remove filler words
    fillers = {'feel', 'any', 'also', 'recently', 'currently', 'ever', 'regularly',
               'frequently', 'often', 'usually', 'sometimes', 'along', 'with',
               'the', 'a', 'an', 'or', 'and', 'in', 'on', 'at', 'to', 'of',
               'your', 'you', 'if', 'yes', 'when', 'how', 'what', 'which',
               'been', 'had', 'has', 'have', 'does', 'did', 'are', 'is', 'was',
               'that', 'this', 'it', 'by', 'from', 'for', 'not', 'but'}

    words = [w for w in text.split() if w not in fillers and len(w) > 2]

    # Take first 3-4 meaningful words
    key_words = words[:4]
    if not key_words:
        return "unknown"

    return "_".join(key_words)


# ─────────────────────────────────────────────
# 2. EXTRA MEDICAL INFO ACCUMULATOR
# ─────────────────────────────────────────────

class ExtraMedicalInfo:
    """
    Accumulates ALL medical details extracted by the LLM across
    all iterations. This is the "what do we already know?" store.

    Structured as a dict of sets for fast lookup:
    {
        "symptoms": {"cough", "fever", "headache"},
        "qualifiers": {"productive_cough", "dry_cough", "high_fever"},
        "body_parts": {"chest", "lungs", "throat"},
        "medications": {"salbutamol", "ibuprofen"},
        "family_history": {"tuberculosis", "asthma"},
        "personal_history": {"diabetes", "hypertension"},
        "durations": {"cough:2_weeks", "fever:3_days"},
        "negated": {"headache", "nausea"},
        "raw_details": {"yellowish phlegm", "behind eyes pain"},
    }
    """

    def __init__(self):
        self.symptoms = set()
        self.qualifiers = set()
        self.body_parts = set()
        self.medications = set()
        self.family_history = set()
        self.personal_history = set()
        self.durations = set()
        self.negated = set()
        self.raw_details = set()

    def absorb_llm_output(self, llm_output):
        """Absorb everything from one LLM extraction pass."""
        if not llm_output:
            return

        for sym in llm_output.get("symptoms_present", []):
            name = sym.get("name", "").lower().strip() if isinstance(sym, dict) else str(sym).lower().strip()
            if name:
                self.symptoms.add(name)
            qualifier = (sym.get("qualifier", "") or "").lower().strip() if isinstance(sym, dict) else ""
            if qualifier and qualifier != "null":
                combined = "%s_%s" % (name, qualifier) if name else qualifier
                self.qualifiers.add(combined)
            detail = (sym.get("detail", "") or "").lower().strip() if isinstance(sym, dict) else ""
            if detail and detail != "null":
                self.raw_details.add(detail)

        for sym in llm_output.get("symptoms_absent", []):
            s = sym.lower().strip() if isinstance(sym, str) else str(sym).lower().strip()
            if s:
                self.negated.add(s)

        for bp in llm_output.get("body_parts", []):
            if bp:
                self.body_parts.add(bp.lower().strip())

        for med in llm_output.get("medications", []):
            if med:
                self.medications.add(med.lower().strip())

        for fh in llm_output.get("family_history", []):
            if fh:
                self.family_history.add(fh.lower().strip())

        for ph in llm_output.get("personal_history", []):
            if ph:
                self.personal_history.add(ph.lower().strip())

        dur = llm_output.get("duration", {})
        if dur and dur.get("value") is not None:
            key = "%s_%s" % (dur["value"], dur.get("unit", "days"))
            self.durations.add(key)

        # Absorb extra medical info (specific details like "yellowish phlegm")
        for info in llm_output.get("extra_medical_info", []):
            if info and isinstance(info, str):
                self.raw_details.add(info.lower().strip())

    def absorb_parsed_response(self, parsed):
        """Absorb from a parsed response (both LLM and fast-path)."""
        for sym in parsed.get("detected_symptoms", []):
            self.symptoms.add(sym)
        for sym in parsed.get("negated_symptoms", []):
            self.negated.add(sym)
        for bp in parsed.get("body_parts", []):
            self.body_parts.add(bp)
        for med in parsed.get("medications_found", []):
            self.medications.add(med)
        for hist in parsed.get("history_found", []):
            if "family" in hist:
                self.family_history.add(hist)
            else:
                self.personal_history.add(hist)
        for finding in parsed.get("matched_findings", []):
            if not finding.startswith("hint_"):
                self.qualifiers.add(finding)

    def is_known(self, symptom_key):
        """Check if we already know about this symptom (positive or negative)."""
        return symptom_key in self.symptoms or symptom_key in self.negated

    def has_qualifier(self, qualifier_key):
        """Check if we already have a specific qualifier (e.g., 'dry_cough')."""
        return qualifier_key in self.qualifiers

    def has_medication(self, med_key):
        """Check if we know about a specific medication."""
        return med_key in self.medications

    def has_family_history(self, condition):
        """Check if family history of a condition is known."""
        return condition in self.family_history

    def has_body_part(self, part):
        """Check if a body part has been mentioned."""
        return part in self.body_parts

    def summary(self):
        """Return a human-readable summary."""
        parts = []
        if self.symptoms:
            parts.append("Symptoms: %s" % ", ".join(sorted(self.symptoms)))
        if self.qualifiers:
            parts.append("Details: %s" % ", ".join(sorted(self.qualifiers)))
        if self.body_parts:
            parts.append("Body parts: %s" % ", ".join(sorted(self.body_parts)))
        if self.medications:
            parts.append("Medications: %s" % ", ".join(sorted(self.medications)))
        if self.negated:
            parts.append("Denied: %s" % ", ".join(sorted(self.negated)))
        return " | ".join(parts) if parts else "No info yet"


# ─────────────────────────────────────────────
# 3. QUESTION DEDUP GATE
# ─────────────────────────────────────────────

class QuestionDedupGate:
    """
    Central dedup controller. Call should_ask() before asking any question.

    Three layers of dedup:
      1. Category: same normalized category already asked? → SKIP
      2. Symptom: target symptom already known? → SKIP
      3. Knowledge: answer already in extra_medical_info? → AUTO-FILL + SKIP
    """

    def __init__(self, extra_info=None):
        # type: (Optional[ExtraMedicalInfo]) -> None
        self.asked_categories = set()      # normalized category keys
        self.asked_question_ids = set()    # question IDs
        self.asked_question_texts = set()  # normalized question texts
        self.asked_elicits = set()         # elicited symptom keys
        self.extra_info = extra_info or ExtraMedicalInfo()

    def register_question(self, question):
        """Mark a question as asked. Call after asking it."""
        self.asked_question_ids.add(question["id"])
        self.asked_question_texts.add(question["text"].strip().lower())
        self.asked_elicits.add(question["elicits"])

        # Register category
        cat = self._get_category(question)
        if cat:
            self.asked_categories.add(cat)

    def _block(self, question, reason):
        """
        Block a question AND register its ID so the selector never returns it again.
        This is the critical fix: without this, the selector keeps returning
        the same blocked question every iteration, wasting all 9 turns.
        """
        self.asked_question_ids.add(question["id"])
        self.asked_question_texts.add(question["text"].strip().lower())
        self.asked_elicits.add(question["elicits"])
        cat = self._get_category(question)
        if cat:
            self.asked_categories.add(cat)
        return False, reason, None

    def should_ask(self, question, spreader):
        """
        Check if this question should be asked.

        Returns:
            (should_ask: bool, reason: str, auto_fill: dict or None)

        IMPORTANT: When a question is blocked, it is ALWAYS registered
        in asked_question_ids so the selector stops returning it.
        """
        qid = question["id"]
        elicited = question["elicits"]

        # Layer 0: Already asked this exact question
        if qid in self.asked_question_ids:
            return False, "already_asked_id", None

        # Layer 0b: Already asked identical text
        if question["text"].strip().lower() in self.asked_question_texts:
            return self._block(question, "duplicate_text")

        # Layer 1: Category dedup
        cat = self._get_category(question)
        if cat and cat in self.asked_categories:
            return self._block(question, "category_%s_already_asked" % cat)

        # Layer 2: Elicited symptom already probed
        if elicited in self.asked_elicits:
            return self._block(question, "elicited_%s_already_probed" % elicited)

        # Layer 2b: Elicited symptom already in evidence
        if elicited in spreader.active_symptoms or elicited in spreader.negative_symptoms:
            return self._block(question, "elicited_%s_already_known" % elicited)

        # Layer 3: Knowledge dedup — check if answer is in extra_medical_info
        auto = self._check_auto_fill(question)
        if auto:
            # Auto-register so future questions with same category/text are blocked
            self.register_question(question)
            return False, "auto_filled_from_prior_info", auto

        return True, "ok", None

    def _get_category(self, question):
        """Extract normalized category from a question."""
        # Try explicit category field first
        raw_cat = question.get("category")
        if raw_cat:
            return normalize_category(raw_cat)

        # Try symptom field
        symptom = question.get("symptom")
        if symptom:
            return normalize_category(symptom)

        # Try elicits as fallback category
        elicits = question.get("elicits", "")
        if elicits:
            return normalize_category(elicits)

        # Last resort: generate from question text
        return category_from_question_text(question.get("text", ""))

    def _check_auto_fill(self, question):
        """
        Check if the answer to this question is already known from
        previously extracted medical info.

        Returns activation dict if auto-fillable, None otherwise.
        """
        elicited = question["elicits"]

        # If question asks about a medication and we already know about it
        med_tags = question.get("medication_tags", [])
        for med in med_tags:
            if self.extra_info.has_medication(med):
                return {"symptom": elicited, "present": True,
                        "reason": "medication '%s' already detected" % med}

        # If question asks about family history and we already know
        hist_tags = question.get("history_tags", [])
        for hist in hist_tags:
            # Check if any family history item matches
            for fh in self.extra_info.family_history:
                if hist.replace("family_", "") in fh or fh in hist:
                    return {"symptom": elicited, "present": True,
                            "reason": "family history '%s' already detected" % fh}

        # If question asks about a qualifier we already have
        rmap = question.get("response_map", {})
        for keyword, mapping in rmap.items():
            if keyword in ("yes", "no", "duration"):
                continue
            if isinstance(mapping, list):
                for finding in mapping:
                    if self.extra_info.has_qualifier(finding):
                        return {"symptom": elicited, "present": True,
                                "reason": "finding '%s' already known" % finding}

        # Check auto_activate_if field (v2 questions)
        auto_activate = question.get("auto_activate_if", {})
        for med_key in auto_activate.get("medications_detected", []):
            if self.extra_info.has_medication(med_key):
                return {"symptom": elicited, "present": True,
                        "reason": "medication '%s' triggers auto-activation" % med_key}
        for hist_key in auto_activate.get("history_detected", []):
            for ph in self.extra_info.personal_history:
                if hist_key in ph or ph in hist_key:
                    return {"symptom": elicited, "present": True,
                            "reason": "personal history '%s' triggers auto-activation" % ph}

        return None


# ─────────────────────────────────────────────
# 4. ENHANCED LLM PROMPT WITH EXTRA MEDICAL INFO
# ─────────────────────────────────────────────

ENHANCED_EXTRACTION_PROMPT = """You are a medical entity extractor. Extract structured information from the patient's response.

RULES:
- Extract ONLY what the patient explicitly said. Do NOT infer or assume.
- Do NOT diagnose. Do NOT suggest conditions.
- For qualifiers: use specific terms like "productive", "dry", "severe", "mild", "intermittent", "constant".
  Do NOT output the template "productive/dry/severe/mild/null" — pick the ONE that matches, or null.
- For extra_medical_info: extract specific details like "yellowish phlegm", "ring-shaped rash", "pain behind eyes".
- If something is not mentioned, use null or empty list.
- Output ONLY valid JSON. No explanation, no markdown, no preamble.

CONTEXT: The patient was asked: "{question}"
PATIENT RESPONSE: "{response}"

ALREADY KNOWN (do not re-extract):
{known_info}

Extract this JSON:
{{"symptoms_present":[{{"name":"str","qualifier":"str or null","detail":"str or null"}}],"symptoms_absent":["str"],"duration":{{"value":null,"unit":null}},"body_parts":[],"medications":[],"family_history":[],"personal_history":[],"extra_medical_info":[],"onset":null,"severity":null,"is_yes":null,"is_no":null}}

Output ONLY the JSON:"""

ENHANCED_INITIAL_PROMPT = """You are a medical entity extractor. The patient is describing their chief complaint.

RULES:
- Extract ONLY what the patient explicitly said. Do NOT infer.
- Do NOT diagnose.
- For qualifiers: pick ONE specific term (productive/dry/severe/mild/intermittent/constant) or null. Never output the template.
- extra_medical_info: any specific details like "yellowish phlegm", "worse at night", "radiating to arm".
- Output ONLY valid JSON. No explanation, no markdown.

PATIENT SAYS: "{response}"

Extract this JSON:
{{"symptoms_present":[{{"name":"str","qualifier":"str or null","detail":"str or null"}}],"symptoms_absent":["str"],"duration":{{"value":null,"unit":null}},"body_parts":[],"medications":[],"family_history":[],"personal_history":[],"extra_medical_info":[],"onset":null,"severity":null,"is_yes":null,"is_no":null}}

Output ONLY the JSON:"""
