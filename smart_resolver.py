"""
Smart Symptom Resolver
======================
Handles inputs that the simple alias dict misses:

1. BODY-PART-AWARE resolution:
   "pain in upper body"  → chest_pain or body_pain
   "swelling in feet"    → swelling
   "itching on skin"     → itching
   "pain in joints"      → arthritis_pain

2. FUZZY matching:
   "sme kind of pain"    → pain (handles typos)
   "coughing"            → cough
   "breathlessnes"       → asthma_symptoms

3. MULTI-SYMPTOM extraction + prioritization:
   "cough, headache, sneezing" → [("cough", HIGH), ("headache", MEDIUM), ("sneezing", LOW)]
   Returns symptoms sorted by clinical priority so the engine
   allocates more questions to critical ones.
"""

import re
from typing import Optional


# ─────────────────────────────────────────────
# 1. GENERIC SYMPTOM + BODY PART → CANONICAL MAP
# ─────────────────────────────────────────────
# When patient says "pain in X", map (pain, X) → specific symptom

BODY_PART_SYMPTOM_MAP = {
    # (generic_symptom, body_part_keyword) → canonical symptom
    # Pain
    ("pain", "chest"): "chest_pain",
    ("pain", "heart"): "chest_pain",
    ("pain", "upper body"): "chest_pain",
    ("pain", "upper part"): "chest_pain",
    ("pain", "abdomen"): "appendicitis_pain",
    ("pain", "stomach"): "acidity",
    ("pain", "belly"): "appendicitis_pain",
    ("pain", "lower right"): "appendicitis_pain",
    ("pain", "head"): "headache",
    ("pain", "forehead"): "headache",
    ("pain", "temple"): "headache",
    ("pain", "joint"): "arthritis_pain",
    ("pain", "knee"): "arthritis_pain",
    ("pain", "hip"): "arthritis_pain",
    ("pain", "wrist"): "arthritis_pain",
    ("pain", "finger"): "arthritis_pain",
    ("pain", "back"): "body_pain",
    ("pain", "body"): "body_pain",
    ("pain", "muscle"): "body_pain",
    ("pain", "throat"): "sore_throat",
    ("pain", "neck"): "body_pain",
    ("pain", "eye"): "blurred_vision",
    ("pain", "behind eye"): "headache",
    ("pain", "ear"): "sore_throat",  # referred pain

    # Swelling
    ("swelling", "feet"): "swelling",
    ("swelling", "ankle"): "swelling",
    ("swelling", "leg"): "swelling",
    ("swelling", "face"): "swelling",
    ("swelling", "neck"): "swelling",
    ("swelling", "joint"): "arthritis_pain",

    # Itching
    ("itching", "skin"): "itching",
    ("itching", "body"): "itching",
    ("itching", "eye"): "allergy",
    ("itching", "nose"): "allergy",

    # Burning
    ("burning", "chest"): "acidity",
    ("burning", "stomach"): "acidity",
    ("burning", "throat"): "sore_throat",
    ("burning", "skin"): "rash",
    ("burning", "eye"): "blurred_vision",
    ("burning", "urine"): "nausea",  # UTI proxy — limited conditions

    # Stiffness
    ("stiffness", "joint"): "arthritis_pain",
    ("stiffness", "knee"): "arthritis_pain",
    ("stiffness", "back"): "body_pain",
    ("stiffness", "neck"): "body_pain",

    # Numbness
    ("numbness", "arm"): "balance_problem",
    ("numbness", "leg"): "balance_problem",
    ("numbness", "face"): "balance_problem",
    ("numbness", "hand"): "balance_problem",
    ("numbness", "finger"): "balance_problem",
}

# Generic symptom words that need a body part to resolve
GENERIC_SYMPTOM_WORDS = {
    "pain", "ache", "aching", "hurts", "hurting", "sore", "soreness",
    "swelling", "swollen", "swelled",
    "itching", "itchy", "itch",
    "burning", "burn",
    "stiffness", "stiff",
    "numbness", "numb", "tingling",
}

# Default mapping when body part isn't recognized
GENERIC_DEFAULTS = {
    "pain": "body_pain",
    "ache": "body_pain",
    "aching": "body_pain",
    "hurts": "body_pain",
    "hurting": "body_pain",
    "sore": "sore_throat",
    "soreness": "body_pain",
    "swelling": "swelling",
    "swollen": "swelling",
    "itching": "itching",
    "itchy": "itching",
    "itch": "itching",
    "burning": "acidity",
    "stiffness": "arthritis_pain",
    "stiff": "arthritis_pain",
    "numbness": "balance_problem",
    "numb": "balance_problem",
    "tingling": "balance_problem",
}

# ─────────────────────────────────────────────
# 2. FUZZY ALIAS EXPANSION
# ─────────────────────────────────────────────
# Common misspellings and variations

FUZZY_ALIASES = {
    # Typos
    "coughing": "cough", "cof": "cough", "koff": "cough",
    "fiver": "fever", "fevar": "fever", "bukhar": "fever",
    "headach": "headache", "hedache": "headache",
    "breathlessnes": "asthma_symptoms", "breathlessness": "asthma_symptoms",
    "whezing": "asthma_symptoms", "weezing": "asthma_symptoms",
    "vomitting": "nausea", "vomiting": "nausea", "puking": "nausea",
    "diarrhoea": "bloating", "diarrhea": "bloating", "loose motion": "bloating",
    "loose motions": "bloating",
    "giddiness": "balance_problem", "giddy": "balance_problem",
    "snooring": "broken_voice", "hoarse voice": "broken_voice",
    "pimple": "acne", "zits": "acne",
    "hives": "rash", "welts": "rash",
    "tirednes": "fatigue", "tiredness": "fatigue", "lethargy": "fatigue",
    "sleeplessness": "anxiety", "insomnia": "anxiety",
    "palpitation": "palpitations", "heart racing": "palpitations",
    "runny nose": "cold", "stuffy nose": "cold", "blocked nose": "cold",
    "sneezing": "cold",
    "stomach pain": "acidity", "tummy pain": "acidity",
    "gas": "bloating", "flatulence": "bloating",
    "weight gain": "weight_loss",  # same symptom node, different direction
    "hair loss": "brittle_nails",  # both point to thyroid/iron deficiency
    "rashes": "rash", "spots": "rash",
    "bruise": "bruises", "marks": "bruises",

    # Hindi/regional common terms
    "bukhar": "fever", "buhkaar": "fever",
    "khansi": "cough", "khasi": "cough",
    "sir dard": "headache", "sar dard": "headache",
    "ulti": "nausea", "ji machlana": "nausea",
    "dast": "bloating", "pet dard": "acidity",
    "sans": "asthma_symptoms", "dum": "asthma_symptoms",
    "chakkar": "balance_problem",
    "khujli": "itching",
    "sujan": "swelling", "soojan": "swelling",
    "thakan": "fatigue",
    "ghabrahat": "anxiety", "bechain": "anxiety",
}


# ─────────────────────────────────────────────
# 3. SYMPTOM CLINICAL PRIORITY
# ─────────────────────────────────────────────
# Used for multi-symptom proportional questioning.
# Higher priority = more questions allocated.

SYMPTOM_PRIORITY = {
    # Critical — always investigate
    "chest_pain": 10, "palpitations": 10, "balance_problem": 9,
    "blood_in_stool": 9, "blurred_vision": 8,

    # High — common chief complaints
    "fever": 8, "cough": 7, "headache": 7, "asthma_symptoms": 8,
    "body_pain": 6, "acidity": 6, "nausea": 6, "arthritis_pain": 6,

    # Medium
    "rash": 5, "sore_throat": 5, "bloating": 5, "fatigue": 5,
    "anxiety": 5, "swelling": 5, "itching": 5, "bruises": 5,
    "weight_loss": 5, "broken_voice": 4, "blister": 4,

    # Low — supporting symptoms
    "cold": 3, "chills": 3, "brittle_nails": 3, "acne": 3,
    "allergy": 4,
}

DEFAULT_PRIORITY = 4


# ─────────────────────────────────────────────
# 4. THE SMART RESOLVER
# ─────────────────────────────────────────────

class SmartSymptomResolver:
    """
    Resolves patient free-text to canonical symptom keys using:
    1. Exact alias match (from graph.aliases)
    2. Fuzzy alias match (typos, regional terms)
    3. Body-part-aware resolution ("pain in chest" → chest_pain)
    4. Generic default ("pain" without body part → body_pain)
    """

    def __init__(self, graph):
        self.graph = graph
        # Merge graph aliases with fuzzy aliases
        self._all_aliases = dict(graph.aliases)
        self._all_aliases.update(FUZZY_ALIASES)

    def resolve(self, text):
        """
        Resolve free-text to a list of (symptom_key, priority) tuples.
        Handles single symptoms, body-part combos, and multi-symptom inputs.

        Returns: [(symptom_key, priority), ...]
        """
        text = text.strip().lower()
        if not text:
            return []

        results = []
        seen = set()

        # Strategy 1: Split on commas/and and resolve each part
        # Includes Hindi "aur", Kannada "mattu", etc.
        parts = re.split(r'[,;]|\band\b|\balso\b|\bwith\b|\baur\b|\bya\b|\bmattu\b|\baani\b', text)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            resolved = self._resolve_single(part)
            for sym, pri in resolved:
                if sym not in seen:
                    seen.add(sym)
                    results.append((sym, pri))

        # Strategy 2: If nothing found, try the whole text as one chunk
        if not results:
            resolved = self._resolve_single(text)
            for sym, pri in resolved:
                if sym not in seen:
                    seen.add(sym)
                    results.append((sym, pri))

        # Sort by priority (highest first)
        results.sort(key=lambda x: -x[1])
        return results

    def _resolve_single(self, text):
        """Resolve a single phrase to symptom(s)."""
        results = []

        # Try exact alias match first
        canonical = self._alias_lookup(text)
        if canonical:
            pri = SYMPTOM_PRIORITY.get(canonical, DEFAULT_PRIORITY)
            results.append((canonical, pri))
            return results

        # Try body-part-aware resolution
        bp_result = self._body_part_resolve(text)
        if bp_result:
            pri = SYMPTOM_PRIORITY.get(bp_result, DEFAULT_PRIORITY)
            results.append((bp_result, pri))
            return results

        # Try matching individual words
        words = text.split()
        for word in words:
            canonical = self._alias_lookup(word)
            if canonical and canonical not in [r[0] for r in results]:
                pri = SYMPTOM_PRIORITY.get(canonical, DEFAULT_PRIORITY)
                results.append((canonical, pri))

        return results

    def _alias_lookup(self, text):
        """Look up text in all aliases (exact + substring)."""
        text = text.strip().lower()

        # Exact match
        if text in self._all_aliases:
            return self._all_aliases[text]

        # Exact match in graph symptoms
        if text in self.graph.symptoms:
            return text

        # Substring match (text contains an alias)
        for alias, canonical in self._all_aliases.items():
            if len(alias) >= 3 and alias in text:
                return canonical

        return None

    def _body_part_resolve(self, text):
        """
        Resolve "pain in upper body" → chest_pain using body-part map.
        """
        text = text.lower()

        # Find which generic symptom word is present
        found_generic = None
        for word in GENERIC_SYMPTOM_WORDS:
            if word in text:
                found_generic = word
                break

        if not found_generic:
            return None

        # Normalize to base form
        base_generic = found_generic
        for base in ("pain", "swelling", "itching", "burning", "stiffness", "numbness"):
            if found_generic.startswith(base[:4]):
                base_generic = base
                break

        # Find which body part is mentioned
        for (generic, body_part), symptom in BODY_PART_SYMPTOM_MAP.items():
            if generic == base_generic and body_part in text:
                return symptom

        # No body part found — use generic default
        return GENERIC_DEFAULTS.get(base_generic)

    def prioritize_symptoms(self, symptom_list):
        """
        Given a list of symptom keys, return them sorted by clinical priority
        with question allocation counts.

        For 9 max questions across N symptoms:
          - Highest priority symptom: gets ~50% of questions
          - Second: ~30%
          - Third+: ~20% split

        Returns: [(symptom, priority, question_allocation), ...]
        """
        if not symptom_list:
            return []

        scored = []
        for sym in symptom_list:
            pri = SYMPTOM_PRIORITY.get(sym, DEFAULT_PRIORITY)
            scored.append((sym, pri))

        scored.sort(key=lambda x: -x[1])

        # Allocate questions (out of 9)
        n = len(scored)
        allocations = []
        if n == 1:
            allocations = [9]
        elif n == 2:
            allocations = [5, 4]
        elif n == 3:
            allocations = [4, 3, 2]
        else:
            # First gets 4, second gets 3, rest split 2
            allocations = [4, 3]
            remaining = 2
            for i in range(2, n):
                alloc = max(1, remaining // (n - i))
                allocations.append(alloc)
                remaining -= alloc

        result = []
        for i, (sym, pri) in enumerate(scored):
            alloc = allocations[i] if i < len(allocations) else 1
            result.append((sym, pri, alloc))

        return result
