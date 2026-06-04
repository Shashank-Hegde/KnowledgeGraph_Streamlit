# LLM-Enhanced Triage Engine — DGX Spark Setup & Architecture
# =============================================================

# ┌──────────────────────────────────────────────────────────────┐
# │                    ARCHITECTURE OVERVIEW                      │
# │                                                              │
# │  Patient Input (free text, multilingual)                     │
# │        │                                                     │
# │        ▼                                                     │
# │  ┌─────────────┐    ┌──────────────┐                        │
# │  │ Fast Parser  │───▸│ LLM Extractor│  (Phi-3 via Ollama)   │
# │  │ (regex,      │    │ (structured  │                        │
# │  │  yes/no)     │    │  JSON out)   │                        │
# │  └──────┬───────┘    └──────┬───────┘                        │
# │         │  simple input     │  complex input (4+ words)      │
# │         └────────┬──────────┘                                │
# │                  ▼                                           │
# │         ┌────────────────┐                                   │
# │         │ Entity Mapper  │  LLM output → KG node keys        │
# │         └────────┬───────┘                                   │
# │                  ▼                                           │
# │         ┌────────────────┐                                   │
# │         │ Knowledge Graph│  activation spreading + softmax   │
# │         │ (with PRIMARY  │  initial symptom at 2.0x boost    │
# │         │  symptom boost)│                                   │
# │         └────────┬───────┘                                   │
# │                  ▼                                           │
# │         ┌────────────────┐                                   │
# │         │ Question Select│  IG + relevance + body_part +     │
# │         │ (enhanced)     │  medication + history filtering   │
# │         └────────┬───────┘                                   │
# │                  ▼                                           │
# │         ┌────────────────┐                                   │
# │         │ Clinician RLHF │  feedback → weight calibration    │
# │         └────────────────┘                                   │
# └──────────────────────────────────────────────────────────────┘

# ═══════════════════════════════════════════════════════════════
# PART 1: DGX SPARK ENVIRONMENT SETUP
# ═══════════════════════════════════════════════════════════════
#
# The DGX Spark has a Grace CPU + Blackwell GPU.
# We use Ollama to serve the LLM locally — no internet needed in production.
#
# STEP 1: SSH into your DGX Spark
#   ssh user@dgx-spark-ip
#
# STEP 2: Create a virtual environment
#   cd /home/user/projects
#   python3 -m venv triage_venv
#   source triage_venv/bin/activate
#
# STEP 3: Install Ollama (if not already installed)
#   curl -fsSL https://ollama.com/install.sh | sh
#
# STEP 4: Pull the model (one-time download, ~2.3GB)
#   ollama pull phi3:mini
#
#   Alternative models (pick ONE):
#     ollama pull gemma2:2b        # smaller, faster, decent JSON
#     ollama pull mistral:7b       # best quality, needs more VRAM
#     ollama pull llama3.2:3b      # good balance of speed + quality
#
# STEP 5: Verify Ollama is running
#   ollama list                    # should show phi3:mini
#   curl http://localhost:11434/api/generate -d '{"model":"phi3:mini","prompt":"hello"}'
#
# STEP 6: Install Python dependencies
#   pip install requests           # for Ollama API calls
#   # No other dependencies needed — everything else is stdlib
#
# STEP 7: Copy triage engine files into the project directory
#   cp /path/to/triage_engine/*.py /home/user/projects/triage_engine/
#
# STEP 8: Test the system
#   cd /home/user/projects/triage_engine
#   python3 test_llm_extraction.py     # tests LLM extraction
#   python3 triage_engine_v2.py        # runs the full engine
#
# ═══════════════════════════════════════════════════════════════
# PART 2: HOW THE LLM LAYER WORKS
# ═══════════════════════════════════════════════════════════════
#
# The LLM is NOT a decision maker. It is a STRUCTURED EXTRACTOR.
#
# Input to LLM:
#   "I've been coughing up yellowish phlegm for about 2 weeks,
#    and my mother had TB. I take salbutamol inhaler daily."
#
# LLM Output (structured JSON):
#   {
#     "symptoms_present": [
#       {"name": "cough", "qualifier": "productive", "detail": "yellowish phlegm"},
#       {"name": "breathing_difficulty", "qualifier": null, "detail": "uses inhaler"}
#     ],
#     "symptoms_absent": [],
#     "duration": {"value": 2, "unit": "weeks"},
#     "body_parts": ["chest", "lungs"],
#     "medications": ["salbutamol", "inhaler"],
#     "family_history": ["tuberculosis"],
#     "personal_history": [],
#     "onset": null,
#     "severity": "moderate",
#     "is_yes": true,
#     "is_no": false
#   }
#
# Entity Mapper then converts this to KG activations:
#   "cough" + "productive" → activate "productive_cough"
#   "salbutamol" / "inhaler" → activate "inhaler_use" (new finding)
#   "mother had TB" → activate "family_history_tb"
#   duration 2 weeks → store metadata + chronic boost
#
# The KG then spreads activation as before, but now with
# MUCH richer input from a single patient response.
#
# ═══════════════════════════════════════════════════════════════
# PART 3: WHEN TO CALL THE LLM vs SKIP IT
# ═══════════════════════════════════════════════════════════════
#
# LLM calls cost 200-500ms on GPU. Not every response needs it.
#
# ALWAYS use LLM:
#   - Iteration 0 (initial symptom) — richest input
#   - Response with 4+ words — likely contains extractable info
#   - Response mentions body parts, medications, or family
#
# NEVER use LLM:
#   - Empty input (just pressed Enter)
#   - Single word: "yes", "no", "nope", "haan"
#   - Pure number: "3" (ambiguous without context anyway)
#
# USE FAST PARSER ONLY:
#   - "3 days", "2 weeks" — regex handles duration fine
#   - "yes", "no", "ya", "nahi" — keyword set match
#
# The system automatically decides which path to take.
