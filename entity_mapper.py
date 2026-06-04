"""
Entity Mapper
=============
Converts LLM extraction output → Knowledge Graph activations.

The LLM outputs raw strings like "cough", "yellowish phlegm", "ibuprofen".
This module maps them to canonical KG node keys like "productive_cough",
"nsaid_use", "family_history_tb".

Three mapping strategies (tried in order):
  1. Exact alias match (fastest — O(1) dict lookup)
  2. Substring alias match (handles "bad cough" → "cough")
  3. Qualifier-aware mapping ("cough" + qualifier "productive" → "productive_cough")
"""

from typing import Optional


# ─────────────────────────────────────────────
# Medication → finding mapping
# ─────────────────────────────────────────────
MEDICATION_MAP = {
    # NSAIDs
    "ibuprofen": "nsaid_use", "aspirin": "nsaid_use", "diclofenac": "nsaid_use",
    "naproxen": "nsaid_use", "painkiller": "nsaid_use", "pain killer": "nsaid_use",
    "advil": "nsaid_use", "motrin": "nsaid_use", "brufen": "nsaid_use",

    # Inhalers / respiratory
    "inhaler": "inhaler_use", "salbutamol": "inhaler_use", "albuterol": "inhaler_use",
    "nebulizer": "inhaler_use", "asthalin": "inhaler_use", "ventolin": "inhaler_use",
    "budesonide": "steroid_use", "prednisolone": "steroid_use", "prednisone": "steroid_use",

    # Antacids / GI
    "antacid": "antacid_use", "omeprazole": "antacid_use", "pantoprazole": "antacid_use",
    "ranitidine": "antacid_use", "eno": "antacid_use", "gelusil": "antacid_use",
    "digene": "antacid_use",

    # Antihistamines
    "cetirizine": "antihistamine_use", "levocetirizine": "antihistamine_use",
    "allegra": "antihistamine_use", "fexofenadine": "antihistamine_use",

    # Cardiac
    "atenolol": "beta_blocker_use", "metoprolol": "beta_blocker_use",
    "amlodipine": "antihypertensive_use", "losartan": "antihypertensive_use",

    # Diabetes
    "metformin": "diabetes_medication", "insulin": "diabetes_medication",
    "glimepiride": "diabetes_medication",

    # Thyroid
    "thyroxine": "thyroid_medication", "levothyroxine": "thyroid_medication",
    "eltroxin": "thyroid_medication",
}

# ─────────────────────────────────────────────
# Family history → finding mapping
# ─────────────────────────────────────────────
FAMILY_HISTORY_MAP = {
    "tuberculosis": "family_history_tb", "tb": "family_history_tb",
    "asthma": "family_atopy", "allergy": "family_atopy", "allergies": "family_atopy",
    "eczema": "family_atopy",
    "diabetes": "family_diabetes",
    "heart disease": "family_cardiac", "heart attack": "family_cardiac",
    "cardiac": "family_cardiac", "bp": "family_cardiac",
    "hypertension": "family_cardiac",
    "cancer": "family_cancer",
    "arthritis": "family_autoimmune", "autoimmune": "family_autoimmune",
    "rheumatoid": "family_autoimmune",
    "thyroid": "family_thyroid",
}

# ─────────────────────────────────────────────
# Personal history → finding mapping
# ─────────────────────────────────────────────
PERSONAL_HISTORY_MAP = {
    "diabetes": "diabetes_history", "diabetic": "diabetes_history",
    "sugar": "diabetes_history",
    "hypertension": "hypertension_history", "bp": "hypertension_history",
    "high blood pressure": "hypertension_history",
    "asthma": "asthma_history",
    "smoking": "smoking_history", "smoker": "smoking_history",
    "smoke": "smoking_history",
    "alcohol": "alcohol_use", "drinking": "alcohol_use",
    "surgery": "surgery_history",
    "thyroid": "thyroid_history",
    "tb": "tb_history", "tuberculosis": "tb_history",
}

# ─────────────────────────────────────────────
# Symptom qualifier → finding mapping
# When LLM extracts symptom + qualifier, map to specific findings
# ─────────────────────────────────────────────
QUALIFIER_MAP = {
    ("cough", "productive"): "productive_cough",
    ("cough", "dry"): "dry_cough",
    ("cough", "blood"): "hemoptysis",
    ("cough", "bloody"): "hemoptysis",
    ("headache", "one-sided"): "unilateral_headache",
    ("headache", "unilateral"): "unilateral_headache",
    ("headache", "both sides"): "bilateral_headache",
    ("headache", "bilateral"): "bilateral_headache",
    ("headache", "throbbing"): "unilateral_headache",  # migraine pattern
    ("pain", "chest"): "chest_pain",
    ("pain", "abdominal"): "appendicitis_pain",
    ("pain", "joint"): "arthritis_pain",
    ("rash", "ring-shaped"): "ring_rash",
    ("rash", "circular"): "ring_rash",
    ("rash", "raised"): "raised_rash",
    ("rash", "itchy"): "itching",
    ("vision", "blurred"): "blurred_vision",
    ("vision", "sudden loss"): "sudden_vision_change",
    ("vision", "gradual"): "gradual_vision_change",
    ("breathing", "nocturnal"): "nocturnal_dyspnea",
    ("breathing", "night"): "nocturnal_dyspnea",
    ("breathing", "exertion"): "exertional_dyspnea",
}

# ─────────────────────────────────────────────
# Body part → relevant symptom category mapping
# ─────────────────────────────────────────────
BODY_PART_CATEGORIES = {
    "head": "neurology",
    "eyes": "ophthalmology",
    "ears": "ent",
    "nose": "ent",
    "throat": "ent",
    "neck": "ent",
    "chest": "cardiology",  # or pulmonology — both
    "lungs": "pulmonology",
    "heart": "cardiology",
    "abdomen": "gastro",
    "stomach": "gastro",
    "pelvis": "gastro",
    "back": "orthopedics",
    "spine": "orthopedics",
    "arms": "orthopedics",
    "legs": "orthopedics",
    "knee": "orthopedics",
    "ankle": "orthopedics",
    "wrist": "orthopedics",
    "fingers": "orthopedics",
    "skin": "dermatology",
    "nails": "dermatology",
}


class EntityMapper:
    """
    Maps LLM extraction output to Knowledge Graph activations.
    """

    def __init__(self, graph):
        """
        Args:
            graph: KnowledgeGraph instance (for alias lookup and edge checking)
        """
        self.graph = graph

    def map_extraction(self, llm_output):
        """
        Convert LLM extraction dict → list of KG activations.

        Returns:
            {
                "activate": [("symptom_key", strength), ...],
                "negate": ["symptom_key", ...],
                "duration": {"value": int, "unit": str} or None,
                "body_parts": ["chest", "lungs"],
                "body_part_categories": ["cardiology", "pulmonology"],
                "metadata": {extra context for the session},
                "is_yes": bool or None,
                "is_no": bool or None,
            }
        """
        if llm_output is None:
            return None

        result = {
            "activate": [],
            "negate": [],
            "duration": None,
            "body_parts": [],
            "body_part_categories": [],
            "metadata": {},
            "is_yes": llm_output.get("is_yes"),
            "is_no": llm_output.get("is_no"),
        }

        # 1. Map symptoms present
        for sym in llm_output.get("symptoms_present", []):
            name = sym.get("name", "").lower().strip()
            qualifier = (sym.get("qualifier") or "").lower().strip()
            detail = (sym.get("detail") or "").lower().strip()

            # Try qualifier-aware mapping first
            mapped = self._map_qualified_symptom(name, qualifier, detail)
            if mapped:
                for m in mapped:
                    result["activate"].append((m, 1.0))
            else:
                # Fall back to alias resolution
                canonical = self.graph.resolve_symptom(name)
                if canonical:
                    result["activate"].append((canonical, 1.0))
                else:
                    # Store as raw for potential future use
                    result["metadata"]["unresolved_symptom_%s" % name] = {
                        "qualifier": qualifier, "detail": detail
                    }

        # 2. Map symptoms absent (negated)
        for sym_name in llm_output.get("symptoms_absent", []):
            canonical = self.graph.resolve_symptom(sym_name.lower().strip())
            if canonical:
                result["negate"].append(canonical)

        # 3. Map duration
        dur = llm_output.get("duration", {})
        if dur and dur.get("value") is not None:
            result["duration"] = {
                "value": dur["value"],
                "unit": dur.get("unit", "days"),
            }

        # 4. Map medications → findings
        for med in llm_output.get("medications", []):
            finding = self._map_medication(med)
            if finding:
                result["activate"].append((finding, 1.0))
                result["metadata"]["medication_%s" % med] = finding

        # 5. Map family history → findings
        for hist in llm_output.get("family_history", []):
            finding = self._map_family_history(hist)
            if finding:
                result["activate"].append((finding, 1.0))

        # 6. Map personal history → findings
        for hist in llm_output.get("personal_history", []):
            finding = self._map_personal_history(hist)
            if finding:
                result["activate"].append((finding, 1.0))

        # 7. Map body parts
        for bp in llm_output.get("body_parts", []):
            bp_lower = bp.lower().strip()
            result["body_parts"].append(bp_lower)
            cat = BODY_PART_CATEGORIES.get(bp_lower)
            if cat and cat not in result["body_part_categories"]:
                result["body_part_categories"].append(cat)

        # 8. Severity and onset metadata
        if llm_output.get("severity"):
            result["metadata"]["severity"] = llm_output["severity"]
        if llm_output.get("onset"):
            result["metadata"]["onset"] = llm_output["onset"]
            # Map onset to findings
            onset = llm_output["onset"].lower()
            if onset in ("sudden", "acute"):
                result["activate"].append(("acute_onset", 0.5))
            elif onset in ("gradual", "slow", "progressive"):
                result["activate"].append(("insidious_onset", 0.5))

        # Deduplicate
        seen_activate = set()
        deduped = []
        for key, strength in result["activate"]:
            if key not in seen_activate:
                seen_activate.add(key)
                deduped.append((key, strength))
        result["activate"] = deduped
        result["negate"] = list(set(result["negate"]))

        return result

    def _map_qualified_symptom(self, name, qualifier, detail):
        """Map symptom + qualifier to specific findings."""
        mapped = []

        # Check qualifier map
        for (syn_name, syn_qual), finding in QUALIFIER_MAP.items():
            if syn_name in name and (syn_qual in qualifier or syn_qual in detail):
                mapped.append(finding)

        # If qualifier mentions body parts, try to map
        combined = "%s %s %s" % (name, qualifier, detail)
        if "phlegm" in combined or "mucus" in combined or "sputum" in combined:
            mapped.append("productive_cough")
        if "blood" in combined and "cough" in name:
            mapped.append("hemoptysis")
        if "behind eyes" in combined or "retro-orbital" in combined:
            mapped.append("dengue_headache")

        return mapped if mapped else None

    def _map_medication(self, med_name):
        """Map medication name to a KG finding."""
        med = med_name.lower().strip()
        # Exact match
        if med in MEDICATION_MAP:
            return MEDICATION_MAP[med]
        # Substring match
        for alias, finding in MEDICATION_MAP.items():
            if alias in med or med in alias:
                return finding
        return None

    def _map_family_history(self, history_item):
        """Map family history item to a KG finding."""
        h = history_item.lower().strip()
        if h in FAMILY_HISTORY_MAP:
            return FAMILY_HISTORY_MAP[h]
        for alias, finding in FAMILY_HISTORY_MAP.items():
            if alias in h or h in alias:
                return finding
        return None

    def _map_personal_history(self, history_item):
        """Map personal history to a KG finding."""
        h = history_item.lower().strip()
        if h in PERSONAL_HISTORY_MAP:
            return PERSONAL_HISTORY_MAP[h]
        for alias, finding in PERSONAL_HISTORY_MAP.items():
            if alias in h or h in alias:
                return finding
        return None
