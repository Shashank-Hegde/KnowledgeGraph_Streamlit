"""
Remote LLM Client
==================
Calls Ollama running on DGX Spark via the exposed tunnel (ngrok/SSH).
Drop-in replacement for the local LLMExtractor — same interface, remote execution.
"""

import json
import re
import time
import requests
from typing import Optional


EXTRACTION_PROMPT = """You are a medical entity extractor. Extract structured information from the patient's response.

RULES:
- Extract ONLY what the patient explicitly said. Do NOT infer or assume.
- Do NOT diagnose. Do NOT suggest conditions.
- For qualifiers: pick ONE specific term (productive/dry/severe/mild) or null. Never output the template.
- If something is not mentioned, use null or empty list.
- Output ONLY valid JSON. No explanation, no markdown, no preamble.

CONTEXT: The patient was asked: "{question}"
PATIENT RESPONSE: "{response}"

Extract this JSON:
{{"symptoms_present":[{{"name":"str","qualifier":"str or null","detail":"str or null"}}],"symptoms_absent":["str"],"duration":{{"value":null,"unit":null}},"body_parts":[],"medications":[],"family_history":[],"personal_history":[],"extra_medical_info":[],"onset":null,"severity":null,"is_yes":null,"is_no":null}}

Output ONLY the JSON:"""


INITIAL_PROMPT = """You are a medical entity extractor. The patient is describing their chief complaint.

RULES:
- Extract ONLY what the patient explicitly said. Do NOT infer.
- Do NOT diagnose.
- For qualifiers: pick ONE specific term or null. Never output the template.
- Output ONLY valid JSON. No explanation, no markdown.

PATIENT SAYS: "{response}"

Extract this JSON:
{{"symptoms_present":[{{"name":"str","qualifier":"str or null","detail":"str or null"}}],"symptoms_absent":["str"],"duration":{{"value":null,"unit":null}},"body_parts":[],"medications":[],"family_history":[],"personal_history":[],"extra_medical_info":[],"onset":null,"severity":null,"is_yes":null,"is_no":null}}

Output ONLY the JSON:"""


MIN_WORDS_FOR_LLM = 4


class RemoteLLMExtractor:
    """Calls Ollama on DGX via HTTP tunnel."""

    def __init__(self, url=None, model=None, auth_user=None, auth_password=None, timeout=15):
        self.url = url or "http://localhost:11434/api/generate"
        self.model = model or "phi3:mini"
        self.auth = None
        if auth_user and auth_password:
            self.auth = (auth_user, auth_password)
        self.timeout = timeout
        self._available = None
        # Required for ngrok tunnels — without this, ngrok returns an HTML interstitial
        self.headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        }

    def is_available(self):
        """Check if the remote Ollama is reachable."""
        # Don't cache indefinitely — recheck every call in case ngrok restarted
        try:
            test_url = self.url.replace("/api/generate", "/api/tags")
            resp = requests.get(
                test_url,
                auth=self.auth,
                headers=self.headers,
                timeout=5,
            )
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def should_use_llm(self, text):
        """Decide whether this input needs LLM extraction."""
        text = text.strip()
        if not text:
            return False
        if len(text.split()) < MIN_WORDS_FOR_LLM:
            medical_hints = ["pain", "ache", "blood", "swelling", "fever",
                            "cough", "rash", "dizzy", "nausea", "vomit",
                            "inhaler", "tablet", "medicine"]
            if not any(h in text.lower() for h in medical_hints):
                return False
        return True

    def extract(self, patient_text, question_text=None, is_initial=False, known_info_summary=""):
        """
        Extract medical entities via remote Ollama.
        Same interface as local LLMExtractor.
        """
        if not self.is_available():
            return None

        if is_initial:
            prompt = INITIAL_PROMPT.format(response=patient_text)
        else:
            q = question_text or "general follow-up"
            prompt = EXTRACTION_PROMPT.format(question=q, response=patient_text)

        try:
            start = time.time()
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 1024,
                    "top_p": 0.9,
                }
            }

            resp = requests.post(
                self.url, json=payload,
                auth=self.auth,
                headers=self.headers,
                timeout=self.timeout,
            )
            elapsed = time.time() - start

            if resp.status_code != 200:
                return None

            raw_output = resp.json().get("response", "")
            return self._parse_output(raw_output, elapsed)

        except Exception as e:
            return None

    def _parse_output(self, raw, elapsed):
        """Parse the LLM's JSON output."""
        text = raw.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'^```\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            return None

        json_str = text[start:end+1]

        try:
            data = json.loads(json_str)
            data["_llm_latency_ms"] = round(elapsed * 1000)
            return self._normalize(data)
        except json.JSONDecodeError:
            return None

    def _normalize(self, data):
        """Ensure all expected fields exist."""
        defaults = {
            "symptoms_present": [], "symptoms_absent": [],
            "duration": {"value": None, "unit": None},
            "body_parts": [], "medications": [],
            "family_history": [], "personal_history": [],
            "extra_medical_info": [],
            "onset": None, "severity": None,
            "is_yes": None, "is_no": None,
        }
        for key, default in defaults.items():
            if key not in data or data[key] is None:
                data[key] = default

        # Normalize symptoms_present
        normalized = []
        for s in data.get("symptoms_present", []):
            if isinstance(s, str):
                normalized.append({"name": s, "qualifier": None, "detail": None})
            elif isinstance(s, dict):
                normalized.append({
                    "name": s.get("name", "unknown"),
                    "qualifier": s.get("qualifier"),
                    "detail": s.get("detail"),
                })
        data["symptoms_present"] = normalized

        data["symptoms_absent"] = [
            s if isinstance(s, str) else s.get("name", "")
            for s in data.get("symptoms_absent", [])
        ]

        dur = data.get("duration")
        if isinstance(dur, dict):
            data["duration"] = {"value": dur.get("value"), "unit": dur.get("unit")}
        else:
            data["duration"] = {"value": None, "unit": None}

        data["body_parts"] = [b.lower().strip() for b in data.get("body_parts", []) if b]
        data["medications"] = [m.lower().strip() for m in data.get("medications", []) if m]
        data["family_history"] = [f.lower().strip() for f in data.get("family_history", []) if f]
        data["personal_history"] = [p.lower().strip() for p in data.get("personal_history", []) if p]
        data["extra_medical_info"] = [e.lower().strip() for e in data.get("extra_medical_info", []) if e]

        return data