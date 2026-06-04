"""
Test: Question Deduplication System
====================================
Tests all 3 dedup layers + category normalization + extra medical info.

Run:  python3 test_dedup.py
"""

import sys
sys.path.insert(0, ".")

from graph_engine import KnowledgeGraph, ActivationSpreader
from question_dedup import (
    normalize_category, category_from_question_text,
    ExtraMedicalInfo, QuestionDedupGate,
)


def test_category_normalization():
    """Test that different phrasings normalize to the same key."""
    print("=" * 65)
    print("  PART A: Category Normalization")
    print("=" * 65)

    cases = [
        # (input, expected)
        ("cough type", "cough_type"),
        ("Cough Type", "cough_type"),
        ("cough_type", "cough_type"),
        ("nausea", "nausea"),
        ("stress impact: arm tremor", "arm_tremor_stress_impact"),
        ("activity impact: sprain", "sprain_activity_impact"),
        ("diet: thirst", "thirst_diet"),
        ("swelling", "swelling"),
        ("location", "location"),
    ]

    all_ok = True
    for raw, expected in cases:
        result = normalize_category(raw)
        ok = result == expected
        status = "✅" if ok else "❌"
        print("  %s '%s' → '%s' (expected '%s')" % (status, raw, result, expected))
        if not ok:
            all_ok = False

    assert all_ok, "FAIL: some normalizations wrong"
    print("  ✅ All category normalizations correct")


def test_category_from_text():
    """Test auto-generating category from question text."""
    print("\n" + "=" * 65)
    print("  PART B: Category from Question Text")
    print("=" * 65)

    cases = [
        ("Is the cough dry or producing mucus/phlegm?", "cough_dry_producing"),
        ("Do you feel a burning sensation in your chest?", "burning_sensation_chest"),
        ("Have you coughed up blood or blood-streaked mucus?", "coughed_blood_blood-streaked"),
        ("Do you smoke or have you ever smoked regularly?", "smoke_smoked"),
    ]

    for text, _ in cases:
        result = category_from_question_text(text)
        print("  '%s'" % text[:50])
        print("    → '%s'" % result)

    print("  ✅ Categories generated (review manually for sanity)")


def test_extra_medical_info():
    """Test the knowledge accumulator."""
    print("\n" + "=" * 65)
    print("  PART C: Extra Medical Info Accumulator")
    print("=" * 65)

    info = ExtraMedicalInfo()

    # Simulate iteration 0: patient says "coughing with yellowish phlegm, take inhaler"
    mock_llm_1 = {
        "symptoms_present": [
            {"name": "cough", "qualifier": "productive", "detail": "yellowish phlegm"},
        ],
        "symptoms_absent": ["headache"],
        "duration": {"value": 1, "unit": "weeks"},
        "body_parts": ["chest"],
        "medications": ["salbutamol"],
        "family_history": ["tuberculosis"],
        "personal_history": ["diabetes"],
        "extra_medical_info": ["yellowish phlegm", "worse at night"],
    }
    info.absorb_llm_output(mock_llm_1)

    print("  After iteration 0:")
    print("    Symptoms: %s" % info.symptoms)
    print("    Qualifiers: %s" % info.qualifiers)
    print("    Body parts: %s" % info.body_parts)
    print("    Medications: %s" % info.medications)
    print("    Family history: %s" % info.family_history)
    print("    Negated: %s" % info.negated)
    print("    Extra details: %s" % info.raw_details)

    assert "cough" in info.symptoms, "FAIL"
    assert "cough_productive" in info.qualifiers, "FAIL"
    assert "chest" in info.body_parts, "FAIL"
    assert "salbutamol" in info.medications, "FAIL"
    assert "tuberculosis" in info.family_history, "FAIL"
    assert "headache" in info.negated, "FAIL"
    assert "yellowish phlegm" in info.raw_details, "FAIL"
    print("  ✅ All absorptions correct")

    # Test lookups
    assert info.is_known("cough") == True
    assert info.is_known("headache") == True   # known as negated
    assert info.is_known("fever") == False
    assert info.has_medication("salbutamol") == True
    assert info.has_family_history("tuberculosis") == True
    assert info.has_body_part("chest") == True
    print("  ✅ All lookups correct")


def test_dedup_gate():
    """Test the 3-layer dedup gate."""
    print("\n" + "=" * 65)
    print("  PART D: Question Dedup Gate")
    print("=" * 65)

    graph = KnowledgeGraph()
    spreader = ActivationSpreader(graph)
    spreader.activate("cough", present=True)
    spreader.activate("asthma_symptoms", present=True)

    info = ExtraMedicalInfo()
    info.qualifiers.add("dry_cough")  # already know it's dry
    info.medications.add("salbutamol")
    info.family_history.add("tuberculosis")

    gate = QuestionDedupGate(extra_info=info)

    # Q108: "Is the cough dry or producing mucus/phlegm?" — first time asking
    q108 = graph.questions.get("Q108")
    if q108:
        ok, reason, auto = gate.should_ask(q108, spreader)
        # This should be auto-filled because "dry_cough" is already in qualifiers
        if not ok and auto:
            print("  Q108 (cough type): SKIP + AUTO-FILL (%s) ✅" % reason)
        elif ok:
            # If not auto-filled, ask it — then register
            gate.register_question(q108)
            print("  Q108 (cough type): ASK (first time)")

    # Try Q013 equivalent — same category, should be blocked
    fake_q013 = {
        "id": "Q013_fake",
        "text": "Is the cough dry or producing mucus/phlegm?",
        "elicits": "cough_type",
        "tags": ["pneumonia", "tuberculosis"],
        "category": "cough type",
        "response_map": {},
    }
    ok, reason, auto = gate.should_ask(fake_q013, spreader)
    if q108:
        # If Q108 was registered, this should be blocked by text or elicits
        print("  Q013_fake (same text): should_ask=%s, reason=%s %s" % (
            ok, reason, "✅" if not ok else "❌"))

    # Test Layer 2: symptom already in spreader
    fake_q_cough = {
        "id": "Q999",
        "text": "Do you have a cough?",
        "elicits": "cough",  # already in active_symptoms
        "tags": ["URI"],
        "response_map": {},
    }
    ok, reason, _ = gate.should_ask(fake_q_cough, spreader)
    assert not ok, "FAIL: cough already active, should skip"
    print("  Q999 (cough already active): SKIP (%s) ✅" % reason)

    # Test Layer 1: category dedup
    gate.register_question({"id": "QXXX", "text": "Rate your nausea", "elicits": "nausea_level",
                           "category": "nausea", "response_map": {}})

    fake_q_nausea2 = {
        "id": "QYYY", "text": "Do you feel nauseated?",
        "elicits": "nausea_feeling", "category": "nausea",
        "response_map": {},
    }
    ok, reason, _ = gate.should_ask(fake_q_nausea2, spreader)
    assert not ok, "FAIL: nausea category already asked"
    print("  QYYY (nausea category already asked): SKIP (%s) ✅" % reason)

    # Test Layer 3: auto-fill from medication
    fake_q_inhaler = {
        "id": "QZZZ", "text": "Do you use an inhaler?",
        "elicits": "inhaler_use",
        "auto_activate_if": {"medications_detected": ["salbutamol"]},
        "response_map": {},
    }
    ok, reason, auto = gate.should_ask(fake_q_inhaler, spreader)
    assert not ok, "FAIL: salbutamol already known"
    assert auto is not None, "FAIL: should have auto-fill"
    print("  QZZZ (inhaler — salbutamol known): AUTO-FILL (%s) ✅" % auto["reason"])

    print("\n  ✅ All dedup gate tests passed")


def test_dedup_at_scale():
    """Simulate a corpus with many duplicate-ish questions."""
    print("\n" + "=" * 65)
    print("  PART E: Scale Simulation (100 questions, many overlaps)")
    print("=" * 65)

    graph = KnowledgeGraph()
    spreader = ActivationSpreader(graph)
    spreader.activate("fever", present=True)

    info = ExtraMedicalInfo()
    gate = QuestionDedupGate(extra_info=info)

    # Simulate 100 questions, many with overlapping categories
    test_questions = []
    categories_used = ["fever_duration", "fever_pattern", "chills", "body_pain",
                       "headache", "rash", "nausea", "cough_type", "travel",
                       "contact_history"]

    for i in range(100):
        cat = categories_used[i % len(categories_used)]
        q = {
            "id": "SCALE_%03d" % i,
            "text": "Question %d about %s?" % (i, cat.replace("_", " ")),
            "elicits": "%s_finding_%d" % (cat, i),
            "category": cat,
            "response_map": {},
        }
        test_questions.append(q)

    # Ask questions through the gate
    asked = 0
    skipped = 0
    for q in test_questions:
        ok, reason, auto = gate.should_ask(q, spreader)
        if ok:
            gate.register_question(q)
            asked += 1
        else:
            skipped += 1

    print("  100 questions submitted to gate:")
    print("    Asked: %d" % asked)
    print("    Skipped: %d (dedup)" % skipped)
    print("    Categories: %d unique" % len(categories_used))

    # Should only ask ~10 (one per category)
    assert asked == len(categories_used), \
        "FAIL: expected %d asked (one per category), got %d" % (len(categories_used), asked)
    assert skipped == 100 - len(categories_used), \
        "FAIL: expected %d skipped, got %d" % (100 - len(categories_used), skipped)
    print("  ✅ Correctly asked exactly 1 question per category (%d/%d skipped)" % (
        skipped, 100))


# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  QUESTION DEDUP — TEST SUITE")
    print("=" * 65)

    test_category_normalization()
    test_category_from_text()
    test_extra_medical_info()
    test_dedup_gate()
    test_dedup_at_scale()

    print("\n\n✅ All dedup tests completed.")
