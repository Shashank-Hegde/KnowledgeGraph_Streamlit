"""
LLM Medical Entity Extractor
=============================
Calls a local Ollama model to extract structured medical information
from free-text patient responses.

The LLM extracts:
  - Symptoms present and absent (with negation)
  - Duration of symptoms
  - Body parts mentioned
  - Medications currently taken
  - Family and personal medical history
  - Onset pattern (sudden vs gradual)
  - Severity indicators
  - Yes/No intent

The LLM NEVER diagnoses. It only extracts what the patient said.

Requires: Ollama running locally with phi3:mini (or any model)
  ollama pull phi3:mini
  ollama serve  (runs on localhost:11434)
"""

import json
import re
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "phi3:mini"
TIMEOUT_SECONDS = 10
MIN_WORDS_FOR_LLM = 4  # don't call LLM for "yes", "no", "3 days"

# ─────────────────────────────────────────────
# The extraction prompt (fixed, narrow, no diagnosis)
# ─────────────────────────────────────────────
EXTRACTION_PROMPT = """You are a medical entity extractor. Extract structured information from the patient's response.

RULES:
- Extract ONLY what the patient explicitly said. Do NOT infer or assume.
- Do NOT diagnose. Do NOT suggest conditions.
- If something is not mentioned, leave the field empty or null.
- Output ONLY valid JSON. No explanation, no markdown, no preamble.

CONTEXT: The patient was asked: "{question}"
PATIENT RESPONSE: "{response}"

Extract this JSON structure:
{{
  "symptoms_present": [
    {{"name": "symptom_name", "qualifier": "productive/dry/severe/mild/null", "detail": "any extra detail or null"}}
  ],
  "symptoms_absent": ["symptom_name"],
  "duration": {{"value": null, "unit": null}},
  "body_parts": [],
  "medications": [],
  "family_history": [],
  "personal_history": [],
  "onset": null,
  "severity": null,
  "is_yes": null,
  "is_no": null
}}

Output ONLY the JSON:"""


INITIAL_SYMPTOM_PROMPT = """You are a medical entity extractor. The patient is describing their chief complaint at the start of a clinical visit.

RULES:
- Extract ONLY what the patient explicitly said. Do NOT infer or assume.
- Do NOT diagnose. Do NOT suggest conditions.
- Output ONLY valid JSON. No explanation, no markdown.

PATIENT SAYS: "{response}"

Extract this JSON structure:
{{
  "symptoms_present": [
    {{"name": "symptom_name", "qualifier": "productive/dry/severe/mild/null", "detail": "any extra detail or null"}}
  ],
  "symptoms_absent": ["symptom_name"],
  "duration": {{"value": null, "unit": null}},
  "body_parts": [],
  "medications": [],
  "family_history": [],
  "personal_history": [],
  "onset": null,
  "severity": null,
  "is_yes": null,
  "is_no": null
}}

Output ONLY the JSON:"""


# ─────────────────────────────────────────────
# LLM Client
# ─────────────────────────────────────────────

class LLMExtractor:
    """
    Calls a local Ollama model to extract medical entities from patient text.
    Falls back gracefully if Ollama is not running.
    """

    def __init__(self, model=None, base_url=None, timeout=None):
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url or OLLAMA_URL
        self.timeout = timeout or TIMEOUT_SECONDS
        self._available = None  # lazy check

    def is_available(self):
        """Check if the Ollama server is reachable."""
        if self._available is not None:
            return self._available
        try:
            import requests
            resp = requests.get(
                self.base_url.replace("/api/generate", "/api/tags"),
                timeout=3
            )
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def should_use_llm(self, text):
        """Decide whether this input needs LLM extraction or simple parsing."""
        text = text.strip()
        if not text:
            return False

        # Single word responses: yes, no, haan, nahi
        if len(text.split()) < MIN_WORDS_FOR_LLM:
            # Exception: short responses with medical terms
            medical_hints = ["pain", "ache", "blood", "swelling", "fever",
                            "cough", "rash", "dizzy", "nausea", "vomit",
                            "inhaler", "tablet", "medicine", "doctor"]
            if not any(h in text.lower() for h in medical_hints):
                return False

        return True

    def extract(self, patient_text, question_text=None, is_initial=False, known_info_summary=""):
        """
        Extract medical entities from patient text.

        Args:
            patient_text: what the patient said
            question_text: what question was asked (None for iteration 0)
            is_initial: True for iteration 0 (chief complaint)
            known_info_summary: string summary of already-known info (for dedup)

        Returns:
            dict with extracted entities, or None if LLM unavailable
        """
        if not self.is_available():
            return None

        # Build prompt — use enhanced versions if known_info is available
        if known_info_summary:
            from question_dedup import ENHANCED_INITIAL_PROMPT, ENHANCED_EXTRACTION_PROMPT
            if is_initial:
                prompt = ENHANCED_INITIAL_PROMPT.format(response=patient_text)
            else:
                q = question_text or "general follow-up"
                prompt = ENHANCED_EXTRACTION_PROMPT.format(
                    question=q, response=patient_text, known_info=known_info_summary)
        else:
            if is_initial:
                prompt = INITIAL_SYMPTOM_PROMPT.format(response=patient_text)
            else:
                q = question_text or "general follow-up"
                prompt = EXTRACTION_PROMPT.format(question=q, response=patient_text)

        # Call Ollama
        try:
            import requests
            start = time.time()

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,  # zero = fully deterministic
                    "num_predict": 1024, # more room for complete JSON
                    "top_p": 0.9,
                }
            }

            resp = requests.post(self.base_url, json=payload, timeout=self.timeout)

            elapsed = time.time() - start
            logger.debug("LLM extraction took %.2fs", elapsed)

            if resp.status_code != 200:
                logger.warning("Ollama returned status %d", resp.status_code)
                return None

            raw_output = resp.json().get("response", "")
            return self._parse_llm_output(raw_output, elapsed)

        except ImportError:
            logger.warning("'requests' library not installed. Run: pip install requests")
            return None
        except Exception as e:
            logger.warning("LLM extraction failed: %s", str(e))
            return None

    def _parse_llm_output(self, raw, elapsed):
        """Parse the LLM's JSON output, handling common formatting issues."""
        # Strip markdown fences
        text = raw.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        # Find the JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            logger.warning("No JSON found in LLM output: %s", text[:200])
            return None

        json_str = text[start:end+1]

        try:
            data = json.loads(json_str)
            data["_llm_latency_ms"] = round(elapsed * 1000)
            return self._normalize(data)
        except json.JSONDecodeError as e:
            logger.warning("JSON parse error: %s | Raw: %s", str(e), json_str[:200])
            return None

    def _normalize(self, data):
        """Ensure all expected fields exist with correct types."""
        defaults = {
            "symptoms_present": [],
            "symptoms_absent": [],
            "duration": {"value": None, "unit": None},
            "body_parts": [],
            "medications": [],
            "family_history": [],
            "personal_history": [],
            "extra_medical_info": [],
            "onset": None,
            "severity": None,
            "is_yes": None,
            "is_no": None,
        }
        for key, default in defaults.items():
            if key not in data:
                data[key] = default
            elif data[key] is None:
                data[key] = default

        # Normalize symptoms_present to list of dicts
        normalized_symptoms = []
        for s in data.get("symptoms_present", []):
            if isinstance(s, str):
                normalized_symptoms.append({"name": s, "qualifier": None, "detail": None})
            elif isinstance(s, dict):
                normalized_symptoms.append({
                    "name": s.get("name", "unknown"),
                    "qualifier": s.get("qualifier"),
                    "detail": s.get("detail"),
                })
        data["symptoms_present"] = normalized_symptoms

        # Normalize symptoms_absent to list of strings
        data["symptoms_absent"] = [
            s if isinstance(s, str) else s.get("name", "unknown")
            for s in data.get("symptoms_absent", [])
        ]

        # Normalize duration
        dur = data.get("duration")
        if isinstance(dur, dict):
            data["duration"] = {
                "value": dur.get("value"),
                "unit": dur.get("unit"),
            }
        elif dur is None:
            data["duration"] = {"value": None, "unit": None}

        # Lowercase everything for matching
        data["body_parts"] = [b.lower().strip() for b in data.get("body_parts", []) if b]
        data["medications"] = [m.lower().strip() for m in data.get("medications", []) if m]
        data["family_history"] = [f.lower().strip() for f in data.get("family_history", []) if f]
        data["personal_history"] = [p.lower().strip() for p in data.get("personal_history", []) if p]
        data["extra_medical_info"] = [e.lower().strip() for e in data.get("extra_medical_info", []) if e]

        return data
