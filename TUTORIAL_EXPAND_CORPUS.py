"""
HOW TO EXPAND THE TRIAGE CORPUS
================================
A practical guide for doctors and engineers.

This tutorial covers:
  1. Adding a new CONDITION (disease)
  2. Adding a new SYMPTOM
  3. Adding new QUESTIONS
  4. Connecting them with EDGES (weights)
  5. Common OPD conditions to add

Everything goes in ONE file: knowledge_base.py
No other files need to change.

═══════════════════════════════════════════════════════════
EXAMPLE: Adding "Common Cold" as a more specific condition
═══════════════════════════════════════════════════════════

Currently, URI (Upper Respiratory Infection) is the catch-all.
Let's add "Common Cold (Rhinitis)" as a separate, milder condition.

STEP 1: Add the CONDITION to the CONDITIONS dict
─────────────────────────────────────────────────
Open knowledge_base.py, find the CONDITIONS dict, and add:

    "common_cold": {
        "id": "CON051",                     # next available ID
        "severity": "low",                  # low/medium/high/critical
        "route_to": "general_medicine",     # which department
        "display": "Common Cold (Rhinitis)" # what the doctor sees
    },

Rules for conditions:
  - Key must be lowercase_with_underscores (e.g., "common_cold")
  - ID must be unique (CON051, CON052, ...)
  - severity: "low" = self-care, "medium" = OPD visit, "high" = urgent, "critical" = emergency
  - route_to: the department name (general_medicine, pulmonology, ent, cardiology, etc.)
  - display: human-readable name shown to the doctor


STEP 2: Add EDGES from existing symptoms to this condition
──────────────────────────────────────────────────────────
Find the SYMPTOM_CONDITION_EDGES list and add:

    # Common Cold edges
    ("cold",        "common_cold",    0.80),   # runny nose strongly suggests cold
    ("sore_throat", "common_cold",    0.50),   # sore throat moderately suggests cold
    ("cough",       "common_cold",    0.40),   # cough is weaker signal
    ("fever",       "common_cold",    0.25),   # only mild fever in a cold
    ("headache",    "common_cold",    0.20),   # mild headache possible

Edge weight guide:
    0.80 - 1.00 = very strong association (runny nose → cold)
    0.50 - 0.79 = moderate association (sore throat → cold)
    0.30 - 0.49 = weak but relevant (cough → cold)
    0.10 - 0.29 = possible but unlikely (headache → cold)
    
    Start with 0.50 if unsure — the doctor feedback loop will adjust.
    
    TIP: Don't need to be precise! Set weights to 0.50 for everything,
    then let 20-30 doctor corrections calibrate them automatically.


STEP 3: Add QUESTIONS for this condition (if needed)
────────────────────────────────────────────────────
Most conditions don't need NEW questions — existing questions already cover
the symptoms. But if you want condition-specific follow-ups, add to QUESTIONS:

    {
        "id": "Q113",                                      # next available
        "text": "Is the cold mainly just a runny nose with no other symptoms?",
        "elicits": "isolated_rhinitis",                    # what this question probes
        "tags": ["common_cold", "allergic_rhinitis"],      # conditions it helps distinguish
        "response_map": {
            "yes": ["isolated_rhinitis"],                  # if yes, activate this finding
            "just nose": ["isolated_rhinitis"],
        },
        
        # OPTIONAL v2 fields (add any, all, or none):
        "category": "cold_severity",                       # dedup key
        "symptom_tags": ["cold"],                          # only relevant when cold is active
        "condition_tags": ["common_cold", "allergic_rhinitis"],
        "requires_symptoms": ["cold"],                     # only ask if runny nose confirmed
        "gender": None,                                    # ask anyone
        "age_min": None,
        "age_max": None,
        
        # OPTIONAL language translations:
        "hi": "क्या सर्दी मुख्य रूप से सिर्फ नाक बहना है, कोई और लक्षण नहीं?",
        "kn": "ಶೀತವು ಮುಖ್ಯವಾಗಿ ಮೂಗು ಸೋರುವಿಕೆ ಮಾತ್ರವೇ?",
    },

Question field reference:
    id          : "Q113" (unique, sequential)
    text        : the English question text
    elicits     : the symptom/finding this probes (one string)
    tags        : list of condition keys this helps discriminate (REQUIRED)
    response_map: dict of keyword → list of findings to activate (REQUIRED)
    
    Everything else is OPTIONAL. Add what you have, skip what you don't.


STEP 4: Add EXTRA FINDING EDGES (if you added new findings)
───────────────────────────────────────────────────────────
If your question's response_map creates new findings (like "isolated_rhinitis"),
add edges from that finding to conditions in EXTRA_FINDING_EDGES:

    ("isolated_rhinitis", "common_cold",       0.70),
    ("isolated_rhinitis", "allergic_rhinitis",  0.50),


STEP 5: Add ALIASES (if new symptoms have alternate names)
─────────────────────────────────────────────────────────
In SYMPTOM_ALIASES, add any way a patient might say this symptom:

    "runny nose": "cold",
    "nose running": "cold",
    "naak bahna": "cold",    # Hindi
    "moogu soruvu": "cold",  # Kannada

That's it. Save knowledge_base.py and run: python3 triage_engine_v2.py


═══════════════════════════════════════════════════════════
COMMON OPD CONDITIONS TO ADD (copy-paste ready)
═══════════════════════════════════════════════════════════

Below are conditions commonly seen in Indian OPDs that are NOT
yet in the knowledge base. A doctor should review and adjust weights.

──── GENERAL / FEVER ────

# 1. Seasonal Flu (distinct from Influenza)
"seasonal_flu": {"id": "CON051", "severity": "low", "route_to": "general_medicine", "display": "Seasonal Flu"},
    Edges: ("fever", "seasonal_flu", 0.60), ("cold", "seasonal_flu", 0.70),
           ("cough", "seasonal_flu", 0.55), ("sore_throat", "seasonal_flu", 0.50),
           ("body_pain", "seasonal_flu", 0.45), ("headache", "seasonal_flu", 0.30)

# 2. Heat Exhaustion / Dehydration
"dehydration": {"id": "CON052", "severity": "medium", "route_to": "general_medicine", "display": "Dehydration / Heat Exhaustion"},
    Edges: ("headache", "dehydration", 0.40), ("nausea", "dehydration", 0.45),
           ("fatigue", "dehydration", 0.50), ("balance_problem", "dehydration", 0.35)

# 3. Food Poisoning / Gastroenteritis
"gastroenteritis": {"id": "CON053", "severity": "medium", "route_to": "gastroenterology", "display": "Gastroenteritis / Food Poisoning"},
    Edges: ("nausea", "gastroenteritis", 0.80), ("bloating", "gastroenteritis", 0.60),
           ("fever", "gastroenteritis", 0.40), ("body_pain", "gastroenteritis", 0.30),
           ("acidity", "gastroenteritis", 0.35)

──── GASTRIC (very common in Indian OPDs) ────

# 4. Functional Dyspepsia (the "normal gastric issue")
"dyspepsia": {"id": "CON054", "severity": "low", "route_to": "gastroenterology", "display": "Functional Dyspepsia"},
    Edges: ("acidity", "dyspepsia", 0.70), ("bloating", "dyspepsia", 0.65),
           ("nausea", "dyspepsia", 0.45)

# 5. Constipation
"constipation": {"id": "CON055", "severity": "low", "route_to": "gastroenterology", "display": "Constipation"},
    Edges: ("bloating", "constipation", 0.60), ("acidity", "constipation", 0.30)
    New symptom needed: "constipation_symptom" with alias "constipation"

# 6. Acid Reflux (distinct from GERD — milder)
"acid_reflux": {"id": "CON056", "severity": "low", "route_to": "general_medicine", "display": "Acid Reflux (mild)"},
    Edges: ("acidity", "acid_reflux", 0.75), ("sore_throat", "acid_reflux", 0.25)

──── ENT ────

# 7. Sinusitis
"sinusitis": {"id": "CON057", "severity": "low", "route_to": "ent", "display": "Sinusitis"},
    Edges: ("cold", "sinusitis", 0.55), ("headache", "sinusitis", 0.50),
           ("fever", "sinusitis", 0.25)
    New finding: ("facial_pain", "sinusitis", 0.70)

# 8. Otitis Media (ear infection)
"otitis_media": {"id": "CON058", "severity": "medium", "route_to": "ent", "display": "Otitis Media (Ear Infection)"},
    Edges: ("fever", "otitis_media", 0.35), ("sore_throat", "otitis_media", 0.25)
    New finding: ("ear_symptoms", "otitis_media", 0.75)

──── MUSCULOSKELETAL (very common in elderly OPD) ────

# 9. Muscle Strain
"muscle_strain": {"id": "CON059", "severity": "low", "route_to": "orthopedics", "display": "Muscle Strain"},
    Edges: ("body_pain", "muscle_strain", 0.60), ("arthritis_pain", "muscle_strain", 0.30)

# 10. Lower Back Pain
"lower_back_pain": {"id": "CON060", "severity": "low", "route_to": "orthopedics", "display": "Lower Back Pain"},
    Edges: ("body_pain", "lower_back_pain", 0.45), ("arthritis_pain", "lower_back_pain", 0.40)

──── SKIN ────

# 11. Scabies
"scabies": {"id": "CON061", "severity": "low", "route_to": "dermatology", "display": "Scabies"},
    Edges: ("itching", "scabies", 0.70), ("rash", "scabies", 0.50)

# 12. Heat Rash / Prickly Heat
"heat_rash": {"id": "CON062", "severity": "low", "route_to": "dermatology", "display": "Heat Rash / Prickly Heat"},
    Edges: ("itching", "heat_rash", 0.55), ("rash", "heat_rash", 0.60)

──── URINARY (needs new symptoms) ────

# 13. UTI
"uti": {"id": "CON063", "severity": "medium", "route_to": "general_medicine", "display": "Urinary Tract Infection"},
    New symptom: "urinary_symptoms" with aliases "burning urine", "frequent urination"
    Edges: ("urinary_symptoms", "uti", 0.80), ("fever", "uti", 0.35)
    Gender note: add question with "gender": "female" (more common)

──── EYES ────

# 14. Conjunctivitis
"conjunctivitis": {"id": "CON064", "severity": "low", "route_to": "ophthalmology", "display": "Conjunctivitis (Pink Eye)"},
    Edges: ("blurred_vision", "conjunctivitis", 0.20)
    New symptom: "eye_redness" with aliases "red eye", "pink eye", "eye itching"
    Edges: ("eye_redness", "conjunctivitis", 0.80)


═══════════════════════════════════════════════════════════
ADDING A NEW SYMPTOM (full example: "burning urination")
═══════════════════════════════════════════════════════════

STEP A: Add to SYMPTOMS dict:

    "urinary_symptoms": {
        "id": "SYM031",
        "category": "urinary",
        "display": "Burning Urination / Frequent Urination"
    },

STEP B: Add to SYMPTOM_ALIASES:

    "burning urination": "urinary_symptoms",
    "frequent urination": "urinary_symptoms",
    "urine burning": "urinary_symptoms",
    "peshab mein jalan": "urinary_symptoms",   # Hindi
    "baar baar peshab": "urinary_symptoms",     # Hindi

STEP C: Add edges to conditions:

    ("urinary_symptoms", "uti",           0.80),
    ("urinary_symptoms", "diabetes_history", 0.30),  # diabetes can cause frequent urination

STEP D: Add questions (optional):

    {
        "id": "Q114",
        "text": "Is there a burning sensation when you urinate?",
        "elicits": "urinary_burning",
        "tags": ["uti"],
        "response_map": {"yes": ["urinary_burning"], "burning": ["urinary_burning"]},
        "symptom_tags": ["urinary_symptoms"],
        "requires_symptoms": ["urinary_symptoms"],
        "hi": "क्या पेशाब करते समय जलन होती है?",
    },

STEP E: Add finding edge:

    ("urinary_burning", "uti", 0.75),


═══════════════════════════════════════════════════════════
CHECKLIST: What to change for each new addition
═══════════════════════════════════════════════════════════

Adding a new CONDITION:
  [ ] Add to CONDITIONS dict (id, severity, route_to, display)
  [ ] Add edges in SYMPTOM_CONDITION_EDGES (symptom → condition, weight)
  [ ] Optional: add condition-specific questions to QUESTIONS list

Adding a new SYMPTOM:
  [ ] Add to SYMPTOMS dict (id, category, display)
  [ ] Add aliases to SYMPTOM_ALIASES (all ways a patient might say it)
  [ ] Add edges to SYMPTOM_CONDITION_EDGES
  [ ] Optional: add follow-up questions

Adding new QUESTIONS:
  [ ] Add to QUESTIONS list with unique ID
  [ ] Must have: id, text, elicits, tags, response_map
  [ ] If response_map creates new findings → add edges to EXTRA_FINDING_EDGES
  [ ] Optional: add translations (hi, kn, gu, te, mr)
  [ ] Optional: add v2 fields (category, symptom_tags, gender, age_min/max, etc.)

REMEMBER:
  - Weights don't need to be perfect — start at 0.50 if unsure
  - The doctor feedback loop (RLHF) will calibrate them over 20-30 sessions
  - No other files need to change — only knowledge_base.py
  - Run python3 test_triage.py after changes to verify nothing broke
  - Delete calibrated_weights.json to reset to new baseline weights
"""
