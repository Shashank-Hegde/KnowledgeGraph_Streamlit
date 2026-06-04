"""
Knowledge Base for Clinical Triage Engine
==========================================
Contains:
- Symptom definitions with metadata
- Condition (disease) definitions with severity and routing
- Symptom-Condition edges with weights
- Follow-up question corpus (~100 questions) linked to symptoms/findings
"""

# ─────────────────────────────────────────────
# 1. SYMPTOM DEFINITIONS
# ─────────────────────────────────────────────
SYMPTOMS = {
    "acidity":          {"id": "SYM001", "category": "gastro",       "display": "Acidity / Heartburn"},
    "acne":             {"id": "SYM002", "category": "dermatology",  "display": "Acne"},
    "allergy":          {"id": "SYM003", "category": "immunology",   "display": "Allergic Reaction"},
    "anxiety":          {"id": "SYM004", "category": "psychiatry",   "display": "Anxiety"},
    "appendicitis_pain":{"id": "SYM005", "category": "gastro",       "display": "Lower Right Abdominal Pain"},
    "arthritis_pain":   {"id": "SYM006", "category": "orthopedics",  "display": "Joint Pain / Stiffness"},
    "asthma_symptoms":  {"id": "SYM007", "category": "pulmonology",  "display": "Wheezing / Breathlessness"},
    "balance_problem":  {"id": "SYM008", "category": "neurology",    "display": "Balance Problem / Dizziness"},
    "blister":          {"id": "SYM009", "category": "dermatology",  "display": "Blisters on Skin"},
    "bloating":         {"id": "SYM010", "category": "gastro",       "display": "Bloating / Abdominal Distension"},
    "blood_in_stool":   {"id": "SYM011", "category": "gastro",       "display": "Blood in Stool"},
    "blurred_vision":   {"id": "SYM012", "category": "ophthalmology","display": "Blurred Vision"},
    "brittle_nails":    {"id": "SYM013", "category": "dermatology",  "display": "Brittle / Discolored Nails"},
    "broken_voice":     {"id": "SYM014", "category": "ent",          "display": "Hoarseness / Broken Voice"},
    "bruises":          {"id": "SYM015", "category": "hematology",   "display": "Easy Bruising"},
    "chills":           {"id": "SYM016", "category": "general",      "display": "Chills / Rigors"},
    "cold":             {"id": "SYM017", "category": "general",      "display": "Common Cold / Runny Nose"},
    "fever":            {"id": "SYM018", "category": "general",       "display": "Fever"},
    "headache":         {"id": "SYM019", "category": "neurology",    "display": "Headache"},
    "fatigue":          {"id": "SYM020", "category": "general",      "display": "Fatigue / Tiredness"},
    "nausea":           {"id": "SYM021", "category": "gastro",       "display": "Nausea / Vomiting"},
    "cough":            {"id": "SYM022", "category": "pulmonology",  "display": "Cough"},
    "chest_pain":       {"id": "SYM023", "category": "cardiology",   "display": "Chest Pain"},
    "rash":             {"id": "SYM024", "category": "dermatology",  "display": "Skin Rash"},
    "sore_throat":      {"id": "SYM025", "category": "ent",          "display": "Sore Throat"},
    "body_pain":        {"id": "SYM026", "category": "general",      "display": "Body Pain / Myalgia"},
    "weight_loss":      {"id": "SYM027", "category": "general",      "display": "Unexplained Weight Loss"},
    "palpitations":     {"id": "SYM028", "category": "cardiology",   "display": "Heart Palpitations"},
    "swelling":         {"id": "SYM029", "category": "general",      "display": "Swelling / Edema"},
    "itching":          {"id": "SYM030", "category": "dermatology",  "display": "Itching / Pruritus"},
}


# ─────────────────────────────────────────────
# 2. CONDITION DEFINITIONS
# ─────────────────────────────────────────────
CONDITIONS = {
    "GERD":             {"id": "CON001", "severity": "low",    "route_to": "gastroenterology", "display": "Gastroesophageal Reflux Disease"},
    "peptic_ulcer":     {"id": "CON002", "severity": "medium", "route_to": "gastroenterology", "display": "Peptic Ulcer"},
    "acne_vulgaris":    {"id": "CON003", "severity": "low",    "route_to": "dermatology",      "display": "Acne Vulgaris"},
    "hormonal_acne":    {"id": "CON004", "severity": "low",    "route_to": "dermatology",      "display": "Hormonal Acne"},
    "allergic_rhinitis": {"id": "CON005", "severity": "low",   "route_to": "immunology",       "display": "Allergic Rhinitis"},
    "anaphylaxis":      {"id": "CON006", "severity": "critical","route_to": "emergency",       "display": "Anaphylaxis"},
    "urticaria":        {"id": "CON007", "severity": "low",    "route_to": "dermatology",      "display": "Urticaria (Hives)"},
    "GAD":              {"id": "CON008", "severity": "medium", "route_to": "psychiatry",        "display": "Generalized Anxiety Disorder"},
    "panic_disorder":   {"id": "CON009", "severity": "medium", "route_to": "psychiatry",        "display": "Panic Disorder"},
    "appendicitis":     {"id": "CON010", "severity": "high",   "route_to": "surgery",           "display": "Acute Appendicitis"},
    "rheumatoid_arthritis": {"id": "CON011", "severity": "medium","route_to": "rheumatology",   "display": "Rheumatoid Arthritis"},
    "osteoarthritis":   {"id": "CON012", "severity": "medium", "route_to": "orthopedics",       "display": "Osteoarthritis"},
    "bronchial_asthma": {"id": "CON013", "severity": "medium", "route_to": "pulmonology",       "display": "Bronchial Asthma"},
    "COPD":             {"id": "CON014", "severity": "high",   "route_to": "pulmonology",       "display": "COPD"},
    "vertigo":          {"id": "CON015", "severity": "medium", "route_to": "ent",               "display": "BPPV / Vertigo"},
    "stroke":           {"id": "CON016", "severity": "critical","route_to": "emergency",        "display": "Stroke / TIA"},
    "chickenpox":       {"id": "CON017", "severity": "medium", "route_to": "general_medicine",   "display": "Chickenpox (Varicella)"},
    "herpes_simplex":   {"id": "CON018", "severity": "low",    "route_to": "dermatology",       "display": "Herpes Simplex"},
    "IBS":              {"id": "CON019", "severity": "low",    "route_to": "gastroenterology",   "display": "Irritable Bowel Syndrome"},
    "IBD":              {"id": "CON020", "severity": "high",   "route_to": "gastroenterology",   "display": "Inflammatory Bowel Disease"},
    "colorectal_issue": {"id": "CON021", "severity": "high",   "route_to": "gastroenterology",   "display": "Colorectal Condition"},
    "diabetic_retinopathy": {"id": "CON022", "severity": "high","route_to": "ophthalmology",    "display": "Diabetic Retinopathy"},
    "glaucoma":         {"id": "CON023", "severity": "high",   "route_to": "ophthalmology",      "display": "Glaucoma"},
    "iron_deficiency":  {"id": "CON024", "severity": "medium", "route_to": "general_medicine",   "display": "Iron Deficiency Anemia"},
    "thyroid_disorder": {"id": "CON025", "severity": "medium", "route_to": "endocrinology",      "display": "Thyroid Disorder"},
    "laryngitis":       {"id": "CON026", "severity": "low",    "route_to": "ent",                "display": "Laryngitis"},
    "vocal_cord_issue": {"id": "CON027", "severity": "medium", "route_to": "ent",                "display": "Vocal Cord Dysfunction"},
    "thrombocytopenia": {"id": "CON028", "severity": "high",   "route_to": "hematology",         "display": "Thrombocytopenia"},
    "leukemia":         {"id": "CON029", "severity": "critical","route_to": "oncology",          "display": "Leukemia (suspect)"},
    "malaria":          {"id": "CON030", "severity": "high",   "route_to": "general_medicine",   "display": "Malaria"},
    "dengue":           {"id": "CON031", "severity": "high",   "route_to": "general_medicine",   "display": "Dengue Fever"},
    "typhoid":          {"id": "CON032", "severity": "high",   "route_to": "general_medicine",   "display": "Typhoid Fever"},
    "viral_fever":      {"id": "CON033", "severity": "low",    "route_to": "general_medicine",   "display": "Viral Fever"},
    "influenza":        {"id": "CON034", "severity": "medium", "route_to": "general_medicine",   "display": "Influenza"},
    "migraine":         {"id": "CON035", "severity": "medium", "route_to": "neurology",          "display": "Migraine"},
    "tension_headache": {"id": "CON036", "severity": "low",    "route_to": "general_medicine",   "display": "Tension Headache"},
    "pneumonia":        {"id": "CON037", "severity": "high",   "route_to": "pulmonology",        "display": "Pneumonia"},
    "tuberculosis":     {"id": "CON038", "severity": "high",   "route_to": "pulmonology",        "display": "Tuberculosis"},
    "URI":              {"id": "CON039", "severity": "low",    "route_to": "general_medicine",   "display": "Upper Respiratory Infection"},
    "angina":           {"id": "CON040", "severity": "high",   "route_to": "cardiology",         "display": "Angina Pectoris"},
    "MI":               {"id": "CON041", "severity": "critical","route_to": "emergency",         "display": "Myocardial Infarction (suspect)"},
    "contact_dermatitis":{"id": "CON042", "severity": "low",   "route_to": "dermatology",        "display": "Contact Dermatitis"},
    "eczema":           {"id": "CON043", "severity": "low",    "route_to": "dermatology",        "display": "Eczema"},
    "pharyngitis":      {"id": "CON044", "severity": "low",    "route_to": "ent",                "display": "Pharyngitis"},
    "tonsillitis":      {"id": "CON045", "severity": "medium", "route_to": "ent",                "display": "Tonsillitis"},
    "cardiac_arrhythmia":{"id": "CON046", "severity": "high",  "route_to": "cardiology",         "display": "Cardiac Arrhythmia"},
    "hyperthyroidism":  {"id": "CON047", "severity": "medium", "route_to": "endocrinology",      "display": "Hyperthyroidism"},
    "fungal_infection": {"id": "CON048", "severity": "low",    "route_to": "dermatology",        "display": "Fungal Skin Infection"},
    "rabies_exposure":  {"id": "CON049", "severity": "critical","route_to": "emergency",         "display": "Rabies Exposure (suspect)"},
    "cholesterol_high": {"id": "CON050", "severity": "medium", "route_to": "cardiology",         "display": "Hyperlipidemia"},
}


# ─────────────────────────────────────────────
# 3. SYMPTOM → CONDITION EDGES  (weight 0.0–1.0)
# ─────────────────────────────────────────────
SYMPTOM_CONDITION_EDGES = [
    # --- Acidity / GI ---
    ("acidity",     "GERD",           0.85),
    ("acidity",     "peptic_ulcer",   0.60),
    ("acidity",     "IBS",            0.30),
    ("bloating",    "IBS",            0.70),
    ("bloating",    "GERD",           0.40),
    ("bloating",    "IBD",            0.35),
    ("nausea",      "GERD",           0.50),
    ("nausea",      "peptic_ulcer",   0.55),
    ("nausea",      "appendicitis",   0.60),
    ("nausea",      "typhoid",        0.40),
    ("blood_in_stool", "IBD",         0.80),
    ("blood_in_stool", "colorectal_issue", 0.70),
    ("blood_in_stool", "peptic_ulcer",0.40),

    # --- Dermatology ---
    ("acne",        "acne_vulgaris",  0.80),
    ("acne",        "hormonal_acne",  0.65),
    ("blister",     "chickenpox",     0.70),
    ("blister",     "herpes_simplex", 0.60),
    ("blister",     "contact_dermatitis", 0.40),
    ("rash",        "chickenpox",     0.65),
    ("rash",        "urticaria",      0.60),
    ("rash",        "contact_dermatitis", 0.55),
    ("rash",        "eczema",         0.50),
    ("rash",        "dengue",         0.35),
    ("itching",     "eczema",         0.70),
    ("itching",     "urticaria",      0.65),
    ("itching",     "contact_dermatitis", 0.55),
    ("itching",     "fungal_infection", 0.60),
    ("itching",     "allergic_rhinitis", 0.30),
    ("brittle_nails", "iron_deficiency", 0.65),
    ("brittle_nails", "thyroid_disorder", 0.55),
    ("brittle_nails", "fungal_infection", 0.50),

    # --- Allergy / Immunology ---
    ("allergy",     "allergic_rhinitis", 0.75),
    ("allergy",     "urticaria",      0.60),
    ("allergy",     "anaphylaxis",    0.40),
    ("allergy",     "contact_dermatitis", 0.35),
    ("swelling",    "anaphylaxis",    0.55),
    ("swelling",    "urticaria",      0.40),

    # --- Psychiatry ---
    ("anxiety",     "GAD",            0.80),
    ("anxiety",     "panic_disorder", 0.65),
    ("anxiety",     "hyperthyroidism",0.30),
    ("palpitations","panic_disorder", 0.55),
    ("palpitations","cardiac_arrhythmia", 0.65),
    ("palpitations","hyperthyroidism",0.50),
    ("palpitations","angina",         0.35),

    # --- Surgical / Appendicitis ---
    ("appendicitis_pain", "appendicitis", 0.90),
    ("appendicitis_pain", "IBD",      0.25),

    # --- Orthopedics / Rheumatology ---
    ("arthritis_pain", "rheumatoid_arthritis", 0.70),
    ("arthritis_pain", "osteoarthritis",       0.65),

    # --- Pulmonology ---
    ("asthma_symptoms", "bronchial_asthma", 0.85),
    ("asthma_symptoms", "COPD",       0.50),
    ("cough",       "bronchial_asthma", 0.40),
    ("cough",       "COPD",           0.45),
    ("cough",       "pneumonia",      0.55),
    ("cough",       "tuberculosis",   0.50),
    ("cough",       "URI",            0.60),
    ("cough",       "influenza",      0.45),
    ("chest_pain",  "pneumonia",      0.45),
    ("chest_pain",  "angina",         0.70),
    ("chest_pain",  "MI",             0.65),
    ("chest_pain",  "GERD",           0.30),

    # --- Neurology ---
    ("balance_problem", "vertigo",    0.80),
    ("balance_problem", "stroke",     0.45),
    ("headache",    "migraine",       0.65),
    ("headache",    "tension_headache", 0.60),
    ("headache",    "stroke",         0.25),
    ("headache",    "typhoid",        0.30),
    ("headache",    "dengue",         0.35),
    ("headache",    "malaria",        0.35),
    ("blurred_vision", "diabetic_retinopathy", 0.60),
    ("blurred_vision", "glaucoma",    0.55),
    ("blurred_vision", "migraine",    0.35),
    ("blurred_vision", "stroke",      0.30),

    # --- Ophthalmology ---

    # --- Hematology ---
    ("bruises",     "thrombocytopenia", 0.70),
    ("bruises",     "leukemia",       0.35),
    ("bruises",     "iron_deficiency", 0.30),
    ("bruises",     "dengue",         0.40),

    # --- ENT ---
    ("broken_voice","laryngitis",     0.75),
    ("broken_voice","vocal_cord_issue", 0.60),
    ("broken_voice","thyroid_disorder", 0.30),
    ("sore_throat", "pharyngitis",    0.70),
    ("sore_throat", "tonsillitis",    0.65),
    ("sore_throat", "URI",            0.50),
    ("sore_throat", "influenza",      0.40),

    # --- General / Fever cluster ---
    ("fever",       "malaria",        0.65),
    ("fever",       "dengue",         0.70),
    ("fever",       "typhoid",        0.60),
    ("fever",       "viral_fever",    0.75),
    ("fever",       "influenza",      0.65),
    ("fever",       "pneumonia",      0.50),
    ("fever",       "chickenpox",     0.45),
    ("fever",       "appendicitis",   0.35),
    ("fever",       "URI",            0.45),
    ("chills",      "malaria",        0.75),
    ("chills",      "dengue",         0.55),
    ("chills",      "typhoid",        0.50),
    ("chills",      "influenza",      0.55),
    ("chills",      "viral_fever",    0.50),
    ("chills",      "pneumonia",      0.45),
    ("cold",        "URI",            0.80),
    ("cold",        "allergic_rhinitis", 0.50),
    ("cold",        "influenza",      0.45),
    ("cold",        "viral_fever",    0.40),
    ("body_pain",   "viral_fever",    0.60),
    ("body_pain",   "dengue",         0.65),
    ("body_pain",   "malaria",        0.55),
    ("body_pain",   "influenza",      0.60),
    ("body_pain",   "typhoid",        0.40),
    ("fatigue",     "iron_deficiency",0.60),
    ("fatigue",     "thyroid_disorder", 0.55),
    ("fatigue",     "viral_fever",    0.40),
    ("fatigue",     "dengue",         0.35),
    ("fatigue",     "tuberculosis",   0.45),
    ("fatigue",     "GAD",            0.30),
    ("weight_loss", "thyroid_disorder", 0.55),
    ("weight_loss", "tuberculosis",   0.60),
    ("weight_loss", "IBD",            0.40),
    ("weight_loss", "leukemia",       0.35),
    ("weight_loss", "hyperthyroidism",0.50),
]


# ─────────────────────────────────────────────
# 4. FOLLOW-UP QUESTIONS CORPUS  (~100 questions)
# ─────────────────────────────────────────────
# Each question:
#   - "id":       unique question ID
#   - "text":     the question to ask the patient
#   - "elicits":  the symptom/finding this question probes
#   - "tags":     list of conditions this is most relevant to
#   - "response_map": maps keywords in patient response to new symptom activations
#
# When the patient answers, we check:
#   1. If they mention a known symptom → activate it
#   2. If they give a duration → store as metadata for severity
#   3. If they press enter (null) → treat as "no" for this question's elicited symptom

QUESTIONS = [
    # ── FEVER CLUSTER (Q001–Q012) ──
    {
        "id": "Q001", "text": "How many days have you had fever?",
        "elicits": "fever", "tags": ["malaria", "dengue", "typhoid", "viral_fever"],
        "response_map": {"duration": True},  # captures duration
    },
    {
        "id": "Q002", "text": "Is the fever continuous, or does it come and go?",
        "elicits": "fever_pattern", "tags": ["malaria", "typhoid", "viral_fever"],
        "response_map": {"intermittent": "malaria", "continuous": "typhoid", "comes and goes": "malaria"},
    },
    {
        "id": "Q003", "text": "Do you have chills or shivering along with the fever?",
        "elicits": "chills", "tags": ["malaria", "dengue", "influenza"],
        "response_map": {"yes": ["chills"], "shivering": ["chills"]},
    },
    {
        "id": "Q004", "text": "Are you experiencing body pain or muscle aches?",
        "elicits": "body_pain", "tags": ["dengue", "viral_fever", "influenza", "malaria"],
        "response_map": {"yes": ["body_pain"], "muscle": ["body_pain"], "ache": ["body_pain"]},
    },
    {
        "id": "Q005", "text": "Have you noticed any rash or red spots on your skin?",
        "elicits": "rash", "tags": ["dengue", "chickenpox", "urticaria"],
        "response_map": {"yes": ["rash"], "spots": ["rash"], "red": ["rash"]},
    },
    {
        "id": "Q006", "text": "Do you have a headache? If yes, where exactly?",
        "elicits": "headache", "tags": ["dengue", "malaria", "migraine", "typhoid"],
        "response_map": {"yes": ["headache"], "behind eyes": ["headache", "dengue_headache"], "forehead": ["headache"]},
    },
    {
        "id": "Q007", "text": "Have you traveled to a different region in the last 2 weeks?",
        "elicits": "travel_history", "tags": ["malaria", "dengue", "typhoid"],
        "response_map": {"yes": ["travel_history"]},
    },
    {
        "id": "Q008", "text": "Is anyone else at home or work also sick with similar symptoms?",
        "elicits": "contact_history", "tags": ["viral_fever", "influenza", "chickenpox", "URI"],
        "response_map": {"yes": ["contact_history"]},
    },
    {
        "id": "Q009", "text": "Do you have a cough along with the fever?",
        "elicits": "cough", "tags": ["pneumonia", "tuberculosis", "influenza", "URI"],
        "response_map": {"yes": ["cough"], "dry": ["cough"], "phlegm": ["cough", "productive_cough"]},
    },
    {
        "id": "Q010", "text": "Have you been experiencing night sweats?",
        "elicits": "night_sweats", "tags": ["tuberculosis", "leukemia", "malaria"],
        "response_map": {"yes": ["night_sweats"]},
    },
    {
        "id": "Q011", "text": "Have you noticed any bleeding from gums or nose?",
        "elicits": "bleeding_gums", "tags": ["dengue", "thrombocytopenia", "leukemia"],
        "response_map": {"yes": ["bruises", "bleeding_gums"], "gums": ["bleeding_gums"], "nose": ["bleeding_gums"]},
    },
    {
        "id": "Q012", "text": "Do you feel nauseous, or have you vomited?",
        "elicits": "nausea", "tags": ["dengue", "appendicitis", "GERD", "typhoid"],
        "response_map": {"yes": ["nausea"], "vomit": ["nausea"]},
    },

    # ── RESPIRATORY (Q013–Q022) ──
    # Q013 removed — replaced by Q108 (v2 format with requires_symptoms)
    {
        "id": "Q014", "text": "How long have you had the cough?",
        "elicits": "cough", "tags": ["tuberculosis", "COPD", "bronchial_asthma"],
        "response_map": {"duration": True},
    },
    {
        "id": "Q015", "text": "Do you feel short of breath or difficulty breathing?",
        "elicits": "asthma_symptoms", "tags": ["bronchial_asthma", "COPD", "pneumonia", "MI"],
        "response_map": {"yes": ["asthma_symptoms"], "breathless": ["asthma_symptoms"]},
    },
    {
        "id": "Q016", "text": "Do you hear a whistling or wheezing sound when breathing?",
        "elicits": "wheezing", "tags": ["bronchial_asthma", "COPD"],
        "response_map": {"yes": ["asthma_symptoms"], "wheeze": ["asthma_symptoms"]},
    },
    {
        "id": "Q017", "text": "Does the breathlessness worsen at night or early morning?",
        "elicits": "nocturnal_dyspnea", "tags": ["bronchial_asthma"],
        "response_map": {"yes": ["nocturnal_dyspnea"], "night": ["nocturnal_dyspnea"], "morning": ["nocturnal_dyspnea"]},
    },
    {
        "id": "Q018", "text": "Have you coughed up blood or blood-streaked mucus?",
        "elicits": "hemoptysis", "tags": ["tuberculosis", "pneumonia", "leukemia"],
        "response_map": {"yes": ["hemoptysis"], "blood": ["hemoptysis"]},
    },
    {
        "id": "Q019", "text": "Do you have chest pain when coughing or breathing deeply?",
        "elicits": "chest_pain", "tags": ["pneumonia", "angina", "MI"],
        "response_map": {"yes": ["chest_pain"]},
    },
    {
        "id": "Q020", "text": "Do you smoke or have you ever smoked regularly?",
        "elicits": "smoking_history", "tags": ["COPD", "pneumonia", "tuberculosis"],
        "response_map": {"yes": ["smoking_history"], "smoke": ["smoking_history"]},
    },
    {
        "id": "Q021", "text": "Do you have a sore throat or pain when swallowing?",
        "elicits": "sore_throat", "tags": ["pharyngitis", "tonsillitis", "URI", "influenza"],
        "response_map": {"yes": ["sore_throat"], "swallow": ["sore_throat"]},
    },
    {
        "id": "Q022", "text": "Is your nose blocked or runny?",
        "elicits": "cold", "tags": ["URI", "allergic_rhinitis", "influenza"],
        "response_map": {"yes": ["cold"], "blocked": ["cold"], "runny": ["cold"]},
    },

    # ── GASTRO (Q023–Q034) ──
    {
        "id": "Q023", "text": "Do you feel a burning sensation in your chest or upper stomach?",
        "elicits": "acidity", "tags": ["GERD", "peptic_ulcer"],
        "response_map": {"yes": ["acidity"], "burning": ["acidity"]},
    },
    {
        "id": "Q024", "text": "Does the burning feeling get worse after eating or when lying down?",
        "elicits": "postprandial_burn", "tags": ["GERD"],
        "response_map": {"yes": ["postprandial_burn"], "after eating": ["postprandial_burn"], "lying": ["postprandial_burn"]},
    },
    {
        "id": "Q025", "text": "Do you feel bloated or excessively full after meals?",
        "elicits": "bloating", "tags": ["IBS", "GERD", "IBD"],
        "response_map": {"yes": ["bloating"], "full": ["bloating"]},
    },
    {
        "id": "Q026", "text": "Have your bowel habits changed recently (diarrhea or constipation)?",
        "elicits": "bowel_change", "tags": ["IBS", "IBD", "colorectal_issue", "typhoid"],
        "response_map": {"yes": ["bowel_change"], "diarrhea": ["bowel_change"], "constipation": ["bowel_change"], "loose": ["bowel_change"]},
    },
    {
        "id": "Q027", "text": "Have you noticed blood or dark color in your stool?",
        "elicits": "blood_in_stool", "tags": ["IBD", "colorectal_issue", "peptic_ulcer"],
        "response_map": {"yes": ["blood_in_stool"], "blood": ["blood_in_stool"], "dark": ["blood_in_stool"], "black": ["blood_in_stool"]},
    },
    {
        "id": "Q028", "text": "Do you have pain in your lower right abdomen?",
        "elicits": "appendicitis_pain", "tags": ["appendicitis"],
        "response_map": {"yes": ["appendicitis_pain"], "right side": ["appendicitis_pain"]},
    },
    {
        "id": "Q029", "text": "Does the abdominal pain get worse when you press and then release?",
        "elicits": "rebound_tenderness", "tags": ["appendicitis"],
        "response_map": {"yes": ["rebound_tenderness"]},
    },
    {
        "id": "Q030", "text": "Have you lost your appetite recently?",
        "elicits": "appetite_loss", "tags": ["typhoid", "tuberculosis", "appendicitis", "leukemia"],
        "response_map": {"yes": ["appetite_loss"]},
    },
    {
        "id": "Q031", "text": "Have you had any unintentional weight loss in recent weeks?",
        "elicits": "weight_loss", "tags": ["tuberculosis", "thyroid_disorder", "IBD", "leukemia"],
        "response_map": {"yes": ["weight_loss"]},
    },
    {
        "id": "Q032", "text": "Do you frequently use painkillers like ibuprofen or aspirin?",
        "elicits": "nsaid_use", "tags": ["peptic_ulcer", "GERD"],
        "response_map": {"yes": ["nsaid_use"], "ibuprofen": ["nsaid_use"], "aspirin": ["nsaid_use"]},
    },
    {
        "id": "Q033", "text": "Do you consume alcohol regularly?",
        "elicits": "alcohol_use", "tags": ["peptic_ulcer", "GERD"],
        "response_map": {"yes": ["alcohol_use"]},
    },
    {
        "id": "Q034", "text": "Do you feel pain in your abdomen that is relieved after passing stool?",
        "elicits": "pain_relieved_defecation", "tags": ["IBS"],
        "response_map": {"yes": ["pain_relieved_defecation"]},
    },

    # ── DERMATOLOGY (Q035–Q046) ──
    {
        "id": "Q035", "text": "Where on your body are the skin breakouts or pimples?",
        "elicits": "acne_location", "tags": ["acne_vulgaris", "hormonal_acne"],
        "response_map": {"face": ["acne_face"], "jawline": ["acne_jawline", "hormonal_acne_sign"], "back": ["acne_back"], "chest": ["acne_chest"]},
    },
    {
        "id": "Q036", "text": "Do the breakouts get worse around your menstrual cycle?",
        "elicits": "cyclic_acne", "tags": ["hormonal_acne"],
        "response_map": {"yes": ["cyclic_acne"]},
    },
    {
        "id": "Q037", "text": "Are the blisters filled with clear fluid or pus?",
        "elicits": "blister_type", "tags": ["chickenpox", "herpes_simplex", "contact_dermatitis"],
        "response_map": {"clear": ["vesicle_clear"], "pus": ["vesicle_pus"], "fluid": ["vesicle_clear"]},
    },
    {
        "id": "Q038", "text": "Did the blisters start on your trunk/chest and spread outward?",
        "elicits": "centrifugal_spread", "tags": ["chickenpox"],
        "response_map": {"yes": ["centrifugal_spread"]},
    },
    {
        "id": "Q039", "text": "Do you have itching along with the rash?",
        "elicits": "itching", "tags": ["eczema", "urticaria", "chickenpox", "fungal_infection"],
        "response_map": {"yes": ["itching"], "itch": ["itching"]},
    },
    {
        "id": "Q040", "text": "Did you come in contact with any new soap, detergent, or cosmetic recently?",
        "elicits": "irritant_exposure", "tags": ["contact_dermatitis"],
        "response_map": {"yes": ["irritant_exposure"]},
    },
    {
        "id": "Q041", "text": "Is the rash raised (bumpy) or flat?",
        "elicits": "rash_morphology", "tags": ["urticaria", "eczema", "contact_dermatitis"],
        "response_map": {"raised": ["raised_rash"], "bumpy": ["raised_rash"], "flat": ["flat_rash"]},
    },
    {
        "id": "Q042", "text": "Does the rash appear in the same area each time, or move around?",
        "elicits": "rash_distribution", "tags": ["urticaria", "eczema"],
        "response_map": {"same": ["fixed_rash"], "moves": ["migratory_rash"]},
    },
    {
        "id": "Q043", "text": "Is the affected skin area warm to the touch?",
        "elicits": "skin_warmth", "tags": ["contact_dermatitis", "fungal_infection"],
        "response_map": {"yes": ["skin_warmth"]},
    },
    {
        "id": "Q044", "text": "Do your nails appear discolored, yellow, or thickened?",
        "elicits": "brittle_nails", "tags": ["fungal_infection", "iron_deficiency", "thyroid_disorder"],
        "response_map": {"yes": ["brittle_nails"], "yellow": ["brittle_nails"], "thick": ["brittle_nails"]},
    },
    {
        "id": "Q045", "text": "Is the itching worse at night?",
        "elicits": "nocturnal_itch", "tags": ["eczema", "fungal_infection"],
        "response_map": {"yes": ["nocturnal_itch"], "night": ["nocturnal_itch"]},
    },
    {
        "id": "Q046", "text": "Is the rash ring-shaped or circular?",
        "elicits": "ring_rash", "tags": ["fungal_infection"],
        "response_map": {"yes": ["ring_rash"], "ring": ["ring_rash"], "circular": ["ring_rash"]},
    },

    # ── CARDIOLOGY (Q047–Q055) ──
    {
        "id": "Q047", "text": "Where exactly do you feel chest pain? Can you point to it?",
        "elicits": "chest_pain_location", "tags": ["angina", "MI", "GERD"],
        "response_map": {"center": ["central_chest_pain"], "left": ["left_chest_pain"], "right": ["right_chest_pain"]},
    },
    {
        "id": "Q048", "text": "Does the chest pain radiate to your arm, jaw, or neck?",
        "elicits": "radiating_pain", "tags": ["angina", "MI"],
        "response_map": {"yes": ["radiating_pain"], "arm": ["radiating_pain"], "jaw": ["radiating_pain"]},
    },
    {
        "id": "Q049", "text": "Does the chest pain get worse with physical exertion?",
        "elicits": "exertional_pain", "tags": ["angina", "MI"],
        "response_map": {"yes": ["exertional_pain"]},
    },
    {
        "id": "Q050", "text": "Do you feel your heart beating fast, irregularly, or skipping beats?",
        "elicits": "palpitations", "tags": ["cardiac_arrhythmia", "panic_disorder", "hyperthyroidism"],
        "response_map": {"yes": ["palpitations"], "fast": ["palpitations"], "skip": ["palpitations"], "irregular": ["palpitations"]},
    },
    {
        "id": "Q051", "text": "Do you feel dizzy or lightheaded along with the palpitations?",
        "elicits": "balance_problem", "tags": ["cardiac_arrhythmia", "vertigo"],
        "response_map": {"yes": ["balance_problem"], "dizzy": ["balance_problem"]},
    },
    {
        "id": "Q052", "text": "Have you fainted or nearly fainted recently?",
        "elicits": "syncope", "tags": ["cardiac_arrhythmia", "stroke", "MI"],
        "response_map": {"yes": ["syncope"]},
    },
    {
        "id": "Q053", "text": "Do you have swelling in your feet or ankles?",
        "elicits": "swelling", "tags": ["cardiac_arrhythmia", "cholesterol_high"],
        "response_map": {"yes": ["swelling"], "feet": ["swelling"], "ankles": ["swelling"]},
    },
    {
        "id": "Q054", "text": "Do you have a history of high blood pressure?",
        "elicits": "hypertension_history", "tags": ["angina", "MI", "stroke", "cholesterol_high"],
        "response_map": {"yes": ["hypertension_history"]},
    },
    {
        "id": "Q055", "text": "Do you have diabetes?",
        "elicits": "diabetes_history", "tags": ["diabetic_retinopathy", "angina", "MI", "cholesterol_high"],
        "response_map": {"yes": ["diabetes_history"]},
    },

    # ── NEUROLOGY (Q056–Q063) ──
    {
        "id": "Q056", "text": "Is the headache on one side or both sides?",
        "elicits": "headache_laterality", "tags": ["migraine", "tension_headache"],
        "response_map": {"one": ["unilateral_headache"], "both": ["bilateral_headache"]},
    },
    {
        "id": "Q057", "text": "Do you feel sensitive to light or sound during the headache?",
        "elicits": "photophobia", "tags": ["migraine"],
        "response_map": {"yes": ["photophobia"], "light": ["photophobia"], "sound": ["photophobia"]},
    },
    {
        "id": "Q058", "text": "Do you see flashing lights or zigzag lines before the headache starts?",
        "elicits": "aura", "tags": ["migraine"],
        "response_map": {"yes": ["aura"], "flash": ["aura"], "zigzag": ["aura"]},
    },
    {
        "id": "Q059", "text": "Do you feel the room is spinning around you?",
        "elicits": "balance_problem", "tags": ["vertigo", "stroke"],
        "response_map": {"yes": ["balance_problem"], "spinning": ["balance_problem"]},
    },
    {
        "id": "Q060", "text": "Have you noticed sudden weakness or numbness on one side of your body?",
        "elicits": "hemiparesis", "tags": ["stroke"],
        "response_map": {"yes": ["hemiparesis"]},
    },
    {
        "id": "Q061", "text": "Have you had difficulty speaking or understanding speech suddenly?",
        "elicits": "speech_difficulty", "tags": ["stroke"],
        "response_map": {"yes": ["speech_difficulty"]},
    },
    {
        "id": "Q062", "text": "Is your vision blurred or have you experienced sudden vision loss?",
        "elicits": "blurred_vision", "tags": ["glaucoma", "diabetic_retinopathy", "stroke", "migraine"],
        "response_map": {"yes": ["blurred_vision"], "blurred": ["blurred_vision"], "loss": ["blurred_vision"]},
    },
    {
        "id": "Q063", "text": "Do you feel pressure behind your eyes?",
        "elicits": "eye_pressure", "tags": ["glaucoma", "migraine"],
        "response_map": {"yes": ["eye_pressure"], "pressure": ["eye_pressure"]},
    },

    # ── ENT (Q064–Q070) ──
    {
        "id": "Q064", "text": "Has your voice been hoarse or changed recently?",
        "elicits": "broken_voice", "tags": ["laryngitis", "vocal_cord_issue", "thyroid_disorder"],
        "response_map": {"yes": ["broken_voice"], "hoarse": ["broken_voice"]},
    },
    {
        "id": "Q065", "text": "How long has the voice change lasted?",
        "elicits": "broken_voice", "tags": ["laryngitis", "vocal_cord_issue"],
        "response_map": {"duration": True},
    },
    {
        "id": "Q066", "text": "Does your throat feel scratchy or painful?",
        "elicits": "sore_throat", "tags": ["pharyngitis", "tonsillitis", "laryngitis"],
        "response_map": {"yes": ["sore_throat"], "pain": ["sore_throat"], "scratchy": ["sore_throat"]},
    },
    {
        "id": "Q067", "text": "Do you have difficulty swallowing food or liquids?",
        "elicits": "dysphagia", "tags": ["tonsillitis", "pharyngitis", "thyroid_disorder"],
        "response_map": {"yes": ["dysphagia"]},
    },
    {
        "id": "Q068", "text": "Do you have ear pain or feel fullness in your ears?",
        "elicits": "ear_symptoms", "tags": ["pharyngitis", "tonsillitis", "vertigo"],
        "response_map": {"yes": ["ear_symptoms"]},
    },
    {
        "id": "Q069", "text": "Do you frequently sneeze, especially in mornings or around dust?",
        "elicits": "sneezing", "tags": ["allergic_rhinitis", "URI"],
        "response_map": {"yes": ["sneezing"], "dust": ["sneezing"], "morning": ["sneezing"]},
    },
    {
        "id": "Q070", "text": "Do you have watery or itchy eyes along with the sneezing?",
        "elicits": "watery_eyes", "tags": ["allergic_rhinitis"],
        "response_map": {"yes": ["watery_eyes"], "itchy": ["watery_eyes"]},
    },

    # ── PSYCHIATRY / ANXIETY (Q071–Q078) ──
    {
        "id": "Q071", "text": "Do you often feel excessively worried or on edge?",
        "elicits": "anxiety", "tags": ["GAD", "panic_disorder"],
        "response_map": {"yes": ["anxiety"], "worried": ["anxiety"], "edge": ["anxiety"]},
    },
    {
        "id": "Q072", "text": "Do you have difficulty sleeping or staying asleep?",
        "elicits": "insomnia", "tags": ["GAD", "hyperthyroidism"],
        "response_map": {"yes": ["insomnia"]},
    },
    {
        "id": "Q073", "text": "Have you experienced sudden episodes of intense fear or panic?",
        "elicits": "panic_attack", "tags": ["panic_disorder"],
        "response_map": {"yes": ["panic_attack"]},
    },
    {
        "id": "Q074", "text": "During these episodes, do you feel chest tightness or difficulty breathing?",
        "elicits": "panic_somatic", "tags": ["panic_disorder", "MI"],
        "response_map": {"yes": ["chest_pain", "asthma_symptoms"]},
    },
    {
        "id": "Q075", "text": "Do you feel your hands trembling or shaking?",
        "elicits": "tremor", "tags": ["hyperthyroidism", "anxiety", "panic_disorder"],
        "response_map": {"yes": ["tremor"]},
    },
    {
        "id": "Q076", "text": "Have you been feeling unusually hot or intolerant to heat?",
        "elicits": "heat_intolerance", "tags": ["hyperthyroidism"],
        "response_map": {"yes": ["heat_intolerance"]},
    },
    {
        "id": "Q077", "text": "Have you noticed a swelling in the front of your neck?",
        "elicits": "neck_swelling", "tags": ["thyroid_disorder", "hyperthyroidism"],
        "response_map": {"yes": ["neck_swelling"]},
    },
    {
        "id": "Q078", "text": "Have you been having frequent loose stools despite not having an infection?",
        "elicits": "hyperthyroid_diarrhea", "tags": ["hyperthyroidism"],
        "response_map": {"yes": ["bowel_change"]},
    },

    # ── HEMATOLOGY / GENERAL (Q079–Q086) ──
    {
        "id": "Q079", "text": "Do you bruise easily even from minor bumps?",
        "elicits": "bruises", "tags": ["thrombocytopenia", "leukemia", "dengue"],
        "response_map": {"yes": ["bruises"]},
    },
    {
        "id": "Q080", "text": "Have you noticed tiny red or purple dots on your skin (petechiae)?",
        "elicits": "petechiae", "tags": ["thrombocytopenia", "dengue", "leukemia"],
        "response_map": {"yes": ["petechiae"]},
    },
    {
        "id": "Q081", "text": "Do you feel extremely tired even after resting?",
        "elicits": "fatigue", "tags": ["iron_deficiency", "thyroid_disorder", "leukemia"],
        "response_map": {"yes": ["fatigue"], "tired": ["fatigue"]},
    },
    {
        "id": "Q082", "text": "Do you feel breathless on climbing stairs or during mild activity?",
        "elicits": "exertional_dyspnea", "tags": ["iron_deficiency", "angina", "COPD"],
        "response_map": {"yes": ["exertional_dyspnea"]},
    },
    {
        "id": "Q083", "text": "Do you crave unusual things like ice, clay, or chalk?",
        "elicits": "pica", "tags": ["iron_deficiency"],
        "response_map": {"yes": ["pica"]},
    },
    {
        "id": "Q084", "text": "Are your eyes or skin looking pale or yellowish?",
        "elicits": "pallor", "tags": ["iron_deficiency", "leukemia", "malaria"],
        "response_map": {"yes": ["pallor"], "pale": ["pallor"], "yellow": ["pallor"]},
    },
    {
        "id": "Q085", "text": "Have you had recurrent infections or fevers in recent months?",
        "elicits": "recurrent_infections", "tags": ["leukemia", "iron_deficiency"],
        "response_map": {"yes": ["recurrent_infections"]},
    },
    {
        "id": "Q086", "text": "Have you noticed swollen lymph nodes (lumps in neck, armpit, or groin)?",
        "elicits": "lymphadenopathy", "tags": ["leukemia", "tuberculosis", "tonsillitis"],
        "response_map": {"yes": ["lymphadenopathy"]},
    },

    # ── ALLERGY SPECIFIC (Q087–Q092) ──
    {
        "id": "Q087", "text": "Do you know what triggered the allergic reaction (food, dust, medication)?",
        "elicits": "allergen_trigger", "tags": ["allergic_rhinitis", "urticaria", "anaphylaxis"],
        "response_map": {"food": ["food_allergy"], "dust": ["dust_allergy"], "medicine": ["drug_allergy"], "medication": ["drug_allergy"]},
    },
    {
        "id": "Q088", "text": "Are you having any difficulty breathing or swelling of lips/tongue?",
        "elicits": "airway_compromise", "tags": ["anaphylaxis"],
        "response_map": {"yes": ["airway_compromise"], "swelling": ["airway_compromise"], "lips": ["airway_compromise"]},
    },
    {
        "id": "Q089", "text": "Have you used any new medication in the last few days?",
        "elicits": "new_medication", "tags": ["urticaria", "anaphylaxis", "contact_dermatitis"],
        "response_map": {"yes": ["new_medication"]},
    },
    {
        "id": "Q090", "text": "Do your allergy symptoms occur at a specific season or year-round?",
        "elicits": "seasonal_pattern", "tags": ["allergic_rhinitis"],
        "response_map": {"season": ["seasonal_allergy"], "year": ["perennial_allergy"]},
    },
    {
        "id": "Q091", "text": "Do you have a family history of allergies or asthma?",
        "elicits": "family_atopy", "tags": ["allergic_rhinitis", "bronchial_asthma", "eczema"],
        "response_map": {"yes": ["family_atopy"]},
    },
    {
        "id": "Q092", "text": "Have you been bitten or scratched by an animal recently?",
        "elicits": "animal_bite", "tags": ["rabies_exposure"],
        "response_map": {"yes": ["animal_bite"], "dog": ["animal_bite"], "cat": ["animal_bite"], "bite": ["animal_bite"]},
    },

    # ── ORTHOPEDICS / JOINTS (Q093–Q098) ──
    {
        "id": "Q093", "text": "Which joints are affected? Large joints (knees, hips) or small joints (fingers, wrists)?",
        "elicits": "joint_type", "tags": ["rheumatoid_arthritis", "osteoarthritis"],
        "response_map": {"small": ["small_joint_pain"], "fingers": ["small_joint_pain"], "large": ["large_joint_pain"], "knee": ["large_joint_pain"]},
    },
    {
        "id": "Q094", "text": "Is the joint stiffness worse in the morning? How long does it last?",
        "elicits": "morning_stiffness", "tags": ["rheumatoid_arthritis", "osteoarthritis"],
        "response_map": {"yes": ["morning_stiffness"], "duration": True},
    },
    {
        "id": "Q095", "text": "Are the affected joints swollen, warm, or red?",
        "elicits": "joint_inflammation", "tags": ["rheumatoid_arthritis"],
        "response_map": {"yes": ["joint_inflammation"], "swollen": ["joint_inflammation"], "warm": ["joint_inflammation"]},
    },
    {
        "id": "Q096", "text": "Does the joint pain get worse with activity and better with rest?",
        "elicits": "mechanical_pain", "tags": ["osteoarthritis"],
        "response_map": {"yes": ["mechanical_pain"]},
    },
    {
        "id": "Q097", "text": "Are the symptoms on both sides of the body symmetrically?",
        "elicits": "symmetrical_symptoms", "tags": ["rheumatoid_arthritis"],
        "response_map": {"yes": ["symmetrical_symptoms"]},
    },
    {
        "id": "Q098", "text": "Do you have a family history of arthritis or autoimmune conditions?",
        "elicits": "family_autoimmune", "tags": ["rheumatoid_arthritis"],
        "response_map": {"yes": ["family_autoimmune"]},
    },

    # ── OPHTHALMOLOGY (Q099–Q102) ──
    {
        "id": "Q099", "text": "Is the blurred vision in one eye or both?",
        "elicits": "vision_laterality", "tags": ["glaucoma", "diabetic_retinopathy", "stroke"],
        "response_map": {"one": ["unilateral_vision_loss"], "both": ["bilateral_vision_loss"]},
    },
    {
        "id": "Q100", "text": "Did the vision change come on suddenly or gradually?",
        "elicits": "vision_onset", "tags": ["stroke", "glaucoma", "diabetic_retinopathy"],
        "response_map": {"sudden": ["sudden_vision_change"], "gradual": ["gradual_vision_change"]},
    },
    {
        "id": "Q101", "text": "Do you see halos around lights, especially at night?",
        "elicits": "halos", "tags": ["glaucoma"],
        "response_map": {"yes": ["halos"]},
    },
    {
        "id": "Q102", "text": "Do you see floaters or dark spots in your vision?",
        "elicits": "floaters", "tags": ["diabetic_retinopathy"],
        "response_map": {"yes": ["floaters"], "spots": ["floaters"]},
    },

    # ── v2 FORMAT QUESTIONS: DEMOGRAPHIC-FILTERED (Q103–Q112) ──

    # Female-only questions
    {
        "id": "Q103", "text": "Do the breakouts get worse around your menstrual cycle?",
        "elicits": "cyclic_acne",
        "condition_tags": ["hormonal_acne"],
        "symptom_tags": ["acne"],
        "gender": "female", "age_min": 12, "age_max": 55,
        "requires_symptoms": ["acne"],
        "excludes_symptoms": [],
        "context_tags": ["hormonal"],
        "response_map": {"yes": ["cyclic_acne"]},
        "response_type": "yes_no",
    },
    {
        "id": "Q104", "text": "Are your periods regular? Any recent changes in your cycle?",
        "elicits": "menstrual_irregularity",
        "condition_tags": ["thyroid_disorder", "hormonal_acne", "iron_deficiency"],
        "symptom_tags": ["fatigue", "weight_loss", "acne"],
        "gender": "female", "age_min": 12, "age_max": 55,
        "requires_symptoms": [],
        "excludes_symptoms": [],
        "context_tags": ["hormonal", "history"],
        "response_map": {"yes": ["menstrual_irregularity"], "irregular": ["menstrual_irregularity"],
                         "heavy": ["menstrual_irregularity", "iron_deficiency_sign"]},
        "response_type": "yes_no",
    },
    {
        "id": "Q105", "text": "Is there any chance you could be pregnant?",
        "elicits": "pregnancy_possible",
        "condition_tags": [],
        "symptom_tags": ["nausea", "fatigue", "bloating"],
        "gender": "female", "age_min": 14, "age_max": 50,
        "requires_symptoms": [],
        "excludes_symptoms": [],
        "context_tags": ["screening"],
        "response_map": {"yes": ["pregnancy_possible"]},
        "response_type": "yes_no",
    },

    # Age-gated questions
    {
        "id": "Q106", "text": "Have you had all your routine vaccinations (MMR, chickenpox, etc.)?",
        "elicits": "vaccination_history",
        "condition_tags": ["chickenpox", "viral_fever"],
        "symptom_tags": ["rash", "blister", "fever"],
        "gender": None, "age_min": 1, "age_max": 18,
        "requires_symptoms": [],
        "excludes_symptoms": [],
        "context_tags": ["history"],
        "response_map": {"yes": ["vaccinated"], "no": ["unvaccinated"]},
        "response_type": "yes_no",
    },
    {
        "id": "Q107", "text": "Do you have difficulty holding urine or frequent urination at night?",
        "elicits": "urinary_symptoms",
        "condition_tags": ["cholesterol_high"],
        "symptom_tags": ["fatigue", "weight_loss"],
        "gender": "male", "age_min": 50, "age_max": None,
        "requires_symptoms": [],
        "excludes_symptoms": [],
        "context_tags": ["screening"],
        "response_map": {"yes": ["urinary_symptoms"]},
        "response_type": "yes_no",
    },

    # Context-requires examples
    {
        "id": "Q108", "text": "Is the cough dry or producing mucus/phlegm?",
        "elicits": "cough_type",
        "condition_tags": ["pneumonia", "tuberculosis", "bronchial_asthma", "URI"],
        "symptom_tags": ["cough"],
        "gender": None, "age_min": None, "age_max": None,
        "requires_symptoms": ["cough"],
        "excludes_symptoms": [],
        "context_tags": [],
        "response_map": {"dry": ["dry_cough"], "mucus": ["productive_cough"],
                         "phlegm": ["productive_cough"], "green": ["productive_cough"]},
        "response_type": "free_text",
    },
    {
        "id": "Q109", "text": "Does the chest pain get worse when lying down or after heavy meals?",
        "elicits": "positional_chest_pain",
        "condition_tags": ["GERD", "angina"],
        "symptom_tags": ["chest_pain", "acidity"],
        "gender": None, "age_min": None, "age_max": None,
        "requires_symptoms": ["chest_pain"],
        "excludes_symptoms": [],
        "context_tags": ["positional"],
        "response_map": {"yes": ["positional_chest_pain"], "lying": ["positional_chest_pain"],
                         "meal": ["postprandial_burn"]},
        "response_type": "yes_no",
    },
    {
        "id": "Q110", "text": "Did the joint pain start after an injury or gradually over time?",
        "elicits": "joint_onset",
        "condition_tags": ["osteoarthritis", "rheumatoid_arthritis"],
        "symptom_tags": ["arthritis_pain"],
        "gender": None, "age_min": None, "age_max": None,
        "requires_symptoms": ["arthritis_pain"],
        "excludes_symptoms": [],
        "context_tags": ["onset", "history"],
        "response_map": {"injury": ["traumatic_onset"], "gradual": ["insidious_onset"],
                         "sudden": ["acute_onset"]},
        "response_type": "free_text",
    },
    {
        "id": "Q111", "text": "Do you have a family history of diabetes or heart disease?",
        "elicits": "family_metabolic_history",
        "condition_tags": ["angina", "MI", "diabetic_retinopathy", "cholesterol_high"],
        "symptom_tags": ["chest_pain", "palpitations", "blurred_vision"],
        "gender": None, "age_min": 30, "age_max": None,
        "requires_symptoms": [],
        "excludes_symptoms": [],
        "context_tags": ["family_history"],
        "response_map": {"yes": ["family_metabolic_history"], "diabetes": ["family_diabetes"],
                         "heart": ["family_cardiac"]},
        "response_type": "yes_no",
    },
    {
        "id": "Q112", "text": "Are you currently breastfeeding?",
        "elicits": "breastfeeding",
        "condition_tags": [],
        "symptom_tags": ["fatigue", "body_pain"],
        "gender": "female", "age_min": 18, "age_max": 45,
        "requires_symptoms": [],
        "excludes_symptoms": [],
        "context_tags": ["screening"],
        "response_map": {"yes": ["breastfeeding"]},
        "response_type": "yes_no",
    },
]


# ─────────────────────────────────────────────
# 5. EXTRA EDGES: finding → condition (from question responses)
#    These are symptoms/findings that only emerge via follow-up
# ─────────────────────────────────────────────
EXTRA_FINDING_EDGES = [
    # Fever pattern findings
    ("fever_pattern",       "malaria",          0.40),
    ("fever_pattern",       "typhoid",          0.40),
    ("night_sweats",        "tuberculosis",     0.70),
    ("night_sweats",        "leukemia",         0.50),
    ("night_sweats",        "malaria",          0.35),
    ("travel_history",      "malaria",          0.50),
    ("travel_history",      "dengue",           0.50),
    ("travel_history",      "typhoid",          0.40),
    ("contact_history",     "viral_fever",      0.50),
    ("contact_history",     "influenza",        0.45),
    ("contact_history",     "chickenpox",       0.55),
    ("bleeding_gums",       "dengue",           0.70),
    ("bleeding_gums",       "thrombocytopenia", 0.65),
    ("bleeding_gums",       "leukemia",         0.40),

    # Respiratory findings
    ("dry_cough",           "bronchial_asthma", 0.50),
    ("productive_cough",    "pneumonia",        0.60),
    ("productive_cough",    "tuberculosis",     0.55),
    ("productive_cough",    "COPD",             0.50),
    ("nocturnal_dyspnea",   "bronchial_asthma", 0.65),
    ("hemoptysis",          "tuberculosis",     0.75),
    ("hemoptysis",          "pneumonia",        0.40),
    ("hemoptysis",          "leukemia",         0.30),
    ("smoking_history",     "COPD",             0.60),
    ("smoking_history",     "pneumonia",        0.30),

    # GI findings
    ("postprandial_burn",   "GERD",             0.70),
    ("bowel_change",        "IBS",              0.55),
    ("bowel_change",        "IBD",              0.50),
    ("bowel_change",        "colorectal_issue", 0.35),
    ("rebound_tenderness",  "appendicitis",     0.80),
    ("appetite_loss",       "typhoid",          0.40),
    ("appetite_loss",       "tuberculosis",     0.45),
    ("appetite_loss",       "leukemia",         0.35),
    ("nsaid_use",           "peptic_ulcer",     0.60),
    ("nsaid_use",           "GERD",             0.40),
    ("alcohol_use",         "peptic_ulcer",     0.45),
    ("pain_relieved_defecation", "IBS",         0.65),

    # Derm findings
    ("acne_jawline",        "hormonal_acne",    0.60),
    ("hormonal_acne_sign",  "hormonal_acne",    0.65),
    ("cyclic_acne",         "hormonal_acne",    0.70),
    ("vesicle_clear",       "chickenpox",       0.55),
    ("vesicle_clear",       "herpes_simplex",   0.50),
    ("centrifugal_spread",  "chickenpox",       0.70),
    ("irritant_exposure",   "contact_dermatitis", 0.65),
    ("raised_rash",         "urticaria",        0.55),
    ("nocturnal_itch",      "eczema",           0.55),
    ("ring_rash",           "fungal_infection", 0.75),
    ("skin_warmth",         "contact_dermatitis", 0.40),
    ("new_medication",      "urticaria",        0.45),
    ("new_medication",      "anaphylaxis",      0.30),

    # Cardiac findings
    ("central_chest_pain",  "angina",           0.55),
    ("central_chest_pain",  "MI",               0.60),
    ("left_chest_pain",     "angina",           0.50),
    ("left_chest_pain",     "MI",               0.55),
    ("radiating_pain",      "angina",           0.70),
    ("radiating_pain",      "MI",               0.75),
    ("exertional_pain",     "angina",           0.70),
    ("exertional_pain",     "MI",               0.55),
    ("syncope",             "cardiac_arrhythmia", 0.60),
    ("syncope",             "stroke",           0.40),
    ("hypertension_history","angina",           0.40),
    ("hypertension_history","MI",               0.45),
    ("hypertension_history","stroke",           0.50),
    ("diabetes_history",    "diabetic_retinopathy", 0.70),
    ("diabetes_history",    "angina",           0.35),
    ("diabetes_history",    "MI",               0.40),

    # Neuro findings
    ("unilateral_headache", "migraine",         0.60),
    ("bilateral_headache",  "tension_headache", 0.55),
    ("photophobia",         "migraine",         0.65),
    ("aura",                "migraine",         0.75),
    ("hemiparesis",         "stroke",           0.85),
    ("speech_difficulty",   "stroke",           0.80),

    # Eye findings
    ("sudden_vision_change","stroke",           0.50),
    ("sudden_vision_change","glaucoma",         0.45),
    ("gradual_vision_change","diabetic_retinopathy", 0.55),
    ("halos",               "glaucoma",         0.65),
    ("floaters",            "diabetic_retinopathy", 0.60),
    ("eye_pressure",        "glaucoma",         0.60),
    ("eye_pressure",        "migraine",         0.35),

    # Psych / endocrine findings
    ("insomnia",            "GAD",              0.50),
    ("insomnia",            "hyperthyroidism",  0.40),
    ("panic_attack",        "panic_disorder",   0.80),
    ("tremor",              "hyperthyroidism",  0.55),
    ("tremor",              "panic_disorder",   0.35),
    ("heat_intolerance",    "hyperthyroidism",  0.70),
    ("neck_swelling",       "thyroid_disorder", 0.65),
    ("neck_swelling",       "hyperthyroidism",  0.60),

    # Hematology findings
    ("petechiae",           "thrombocytopenia", 0.75),
    ("petechiae",           "dengue",           0.65),
    ("petechiae",           "leukemia",         0.45),
    ("exertional_dyspnea",  "iron_deficiency",  0.55),
    ("exertional_dyspnea",  "angina",           0.40),
    ("pica",                "iron_deficiency",  0.70),
    ("pallor",              "iron_deficiency",  0.60),
    ("pallor",              "leukemia",         0.40),
    ("recurrent_infections","leukemia",         0.50),
    ("lymphadenopathy",     "leukemia",         0.55),
    ("lymphadenopathy",     "tuberculosis",     0.50),

    # Allergy findings
    ("airway_compromise",   "anaphylaxis",      0.90),
    ("food_allergy",        "anaphylaxis",      0.35),
    ("food_allergy",        "urticaria",        0.40),
    ("drug_allergy",        "anaphylaxis",      0.45),
    ("dust_allergy",        "allergic_rhinitis", 0.55),
    ("seasonal_allergy",    "allergic_rhinitis", 0.60),
    ("family_atopy",        "allergic_rhinitis", 0.35),
    ("family_atopy",        "bronchial_asthma", 0.40),
    ("family_atopy",        "eczema",           0.30),
    ("animal_bite",         "rabies_exposure",  0.85),
    ("sneezing",            "allergic_rhinitis", 0.60),
    ("sneezing",            "URI",              0.40),
    ("watery_eyes",         "allergic_rhinitis", 0.55),

    # Joint findings
    ("small_joint_pain",    "rheumatoid_arthritis", 0.65),
    ("large_joint_pain",    "osteoarthritis",   0.60),
    ("morning_stiffness",   "rheumatoid_arthritis", 0.60),
    ("joint_inflammation",  "rheumatoid_arthritis", 0.65),
    ("mechanical_pain",     "osteoarthritis",   0.60),
    ("symmetrical_symptoms","rheumatoid_arthritis", 0.55),
    ("family_autoimmune",   "rheumatoid_arthritis", 0.35),

    # ENT findings
    ("dysphagia",           "tonsillitis",      0.50),
    ("dysphagia",           "thyroid_disorder",  0.35),
    ("ear_symptoms",        "pharyngitis",      0.30),
    ("ear_symptoms",        "vertigo",          0.40),

    # dengue-specific headache
    ("dengue_headache",     "dengue",           0.55),

    # v2 demographic-filtered question findings
    ("menstrual_irregularity", "thyroid_disorder",  0.45),
    ("menstrual_irregularity", "iron_deficiency",   0.40),
    ("menstrual_irregularity", "hormonal_acne",     0.35),
    ("iron_deficiency_sign",   "iron_deficiency",   0.50),
    ("unvaccinated",           "chickenpox",        0.40),
    ("positional_chest_pain",  "GERD",              0.60),
    ("insidious_onset",        "osteoarthritis",    0.50),
    ("insidious_onset",        "rheumatoid_arthritis", 0.40),
    ("traumatic_onset",        "osteoarthritis",    0.30),
    ("acute_onset",            "rheumatoid_arthritis", 0.35),
    ("family_metabolic_history", "angina",          0.30),
    ("family_metabolic_history", "MI",              0.30),
    ("family_metabolic_history", "cholesterol_high", 0.35),
    ("family_diabetes",        "diabetic_retinopathy", 0.45),
    ("family_cardiac",         "angina",            0.35),
    ("family_cardiac",         "MI",                0.35),

    # LLM-extracted medication findings → conditions
    ("inhaler_use",            "bronchial_asthma",  0.75),
    ("inhaler_use",            "COPD",              0.50),
    ("steroid_use",            "bronchial_asthma",  0.45),
    ("steroid_use",            "rheumatoid_arthritis", 0.30),
    ("antacid_use",            "GERD",              0.55),
    ("antacid_use",            "peptic_ulcer",      0.45),
    ("antihistamine_use",      "allergic_rhinitis",  0.50),
    ("antihistamine_use",      "urticaria",          0.40),
    ("beta_blocker_use",       "angina",             0.40),
    ("beta_blocker_use",       "cardiac_arrhythmia", 0.35),
    ("antihypertensive_use",   "angina",             0.30),
    ("antihypertensive_use",   "cholesterol_high",   0.25),
    ("diabetes_medication",    "diabetic_retinopathy", 0.55),
    ("thyroid_medication",     "thyroid_disorder",    0.65),
    ("thyroid_medication",     "hyperthyroidism",     0.45),

    # LLM-extracted history findings → conditions
    ("family_history_tb",      "tuberculosis",       0.50),
    ("family_thyroid",         "thyroid_disorder",    0.40),
    ("family_thyroid",         "hyperthyroidism",     0.35),
    ("family_cancer",          "leukemia",            0.20),
    ("diabetes_history",       "diabetic_retinopathy", 0.70),
    ("diabetes_history",       "angina",              0.35),
    ("diabetes_history",       "MI",                  0.40),
    ("asthma_history",         "bronchial_asthma",    0.70),
    ("tb_history",             "tuberculosis",        0.55),
    ("thyroid_history",        "thyroid_disorder",     0.60),
    ("thyroid_history",        "hyperthyroidism",      0.45),
    ("surgery_history",        "appendicitis",         0.10),
]


# ─────────────────────────────────────────────
# 6. SYMPTOM ALIASES  (user input → canonical symptom key)
# ─────────────────────────────────────────────
SYMPTOM_ALIASES = {
    # Direct matches from original list
    "acidity": "acidity", "heartburn": "acidity", "acid reflux": "acidity",
    "acne": "acne", "pimples": "acne", "breakout": "acne",
    "allergy": "allergy", "allergic": "allergy", "allergic reaction": "allergy",
    "rabies": "animal_bite",
    "anxiety": "anxiety", "anxious": "anxiety", "nervous": "anxiety", "worry": "anxiety",
    "appendicitis": "appendicitis_pain",
    "arthritis": "arthritis_pain", "joint pain": "arthritis_pain", "joint stiffness": "arthritis_pain",
    "asthma": "asthma_symptoms", "wheezing": "asthma_symptoms", "breathless": "asthma_symptoms",
    "balance problem": "balance_problem", "dizzy": "balance_problem", "dizziness": "balance_problem", "vertigo": "balance_problem",
    "blister": "blister", "blisters": "blister",
    "bloating": "bloating", "bloated": "bloating",
    "blood in stool": "blood_in_stool", "bloody stool": "blood_in_stool",
    "blurred vision": "blurred_vision", "vision problem": "blurred_vision",
    "brittle nails": "brittle_nails",
    "broken voice": "broken_voice", "hoarse": "broken_voice", "hoarseness": "broken_voice",
    "bruises": "bruises", "bruising": "bruises",
    "chills": "chills", "shivering": "chills", "rigors": "chills",
    "cold": "cold", "runny nose": "cold", "common cold": "cold",
    "cholesterol": "weight_loss",  # maps loosely; cholesterol_high is a condition
    "fever": "fever", "temperature": "fever",
    "headache": "headache", "head pain": "headache",
    "fatigue": "fatigue", "tired": "fatigue", "tiredness": "fatigue", "exhausted": "fatigue",
    "nausea": "nausea", "vomiting": "nausea", "vomit": "nausea",
    "cough": "cough", "coughing": "cough",
    "chest pain": "chest_pain",
    "rash": "rash", "skin rash": "rash",
    "sore throat": "sore_throat", "throat pain": "sore_throat",
    "body pain": "body_pain", "body ache": "body_pain", "myalgia": "body_pain",
    "weight loss": "weight_loss",
    "palpitations": "palpitations", "heart racing": "palpitations",
    "swelling": "swelling", "edema": "swelling",
    "itching": "itching", "itch": "itching", "itchy": "itching",
    "chickenpox": "blister",  # maps to blister + rash symptoms
}
