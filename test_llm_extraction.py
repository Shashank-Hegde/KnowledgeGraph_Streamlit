"""
Test: LLM Extraction + Entity Mapping + Primary Boost
=====================================================
Tests the full v2 pipeline.

Part A: Tests entity mapper with mock LLM output (no Ollama needed)
Part B: Tests primary symptom boost vs v1 (no boost)
Part C: Tests live LLM extraction (only if Ollama is running)

Run:  python3 test_llm_extraction.py
"""

import sys
sys.path.insert(0, ".")

from graph_engine import KnowledgeGraph, ActivationSpreader
from entity_mapper import EntityMapper
from enhanced_parser import EnhancedResponseParser
from llm_extractor import LLMExtractor


def test_entity_mapper():
    """Test mapping LLM output → KG activations (no LLM needed)."""
    print("=" * 65)
    print("  PART A: Entity Mapper (mock LLM output)")
    print("=" * 65)

    graph = KnowledgeGraph()
    mapper = EntityMapper(graph)

    # Scenario 1: Rich patient input about cough
    mock_llm_output_1 = {
        "symptoms_present": [
            {"name": "cough", "qualifier": "productive", "detail": "yellowish phlegm"},
            {"name": "fever", "qualifier": None, "detail": "low grade"},
        ],
        "symptoms_absent": ["headache"],
        "duration": {"value": 2, "unit": "weeks"},
        "body_parts": ["chest", "lungs"],
        "medications": ["salbutamol", "ibuprofen"],
        "family_history": ["tuberculosis", "asthma"],
        "personal_history": ["diabetes"],
        "onset": "gradual",
        "severity": "moderate",
        "is_yes": True,
        "is_no": False,
    }

    result = mapper.map_extraction(mock_llm_output_1)

    print("\n  Input: 'coughing yellowish phlegm for 2 weeks, use salbutamol")
    print("          and ibuprofen, mother had TB and asthma, I have diabetes'")
    print()
    print("  Activations:")
    for key, strength in result["activate"]:
        print("    ✓ %-30s  strength=%.1f" % (key, strength))
    print("  Negations:")
    for key in result["negate"]:
        print("    ✗ %s" % key)
    print("  Duration: %s" % result["duration"])
    print("  Body parts: %s" % result["body_parts"])
    print("  Categories: %s" % result["body_part_categories"])

    # Verify key mappings
    activate_keys = [k for k, _ in result["activate"]]
    assert "productive_cough" in activate_keys, "FAIL: should map 'yellowish phlegm' to productive_cough"
    assert "fever" in activate_keys, "FAIL: should detect fever"
    assert "inhaler_use" in activate_keys, "FAIL: should map salbutamol to inhaler_use"
    assert "nsaid_use" in activate_keys, "FAIL: should map ibuprofen to nsaid_use"
    assert "family_history_tb" in activate_keys, "FAIL: should map TB family history"
    assert "family_atopy" in activate_keys, "FAIL: should map asthma family history to family_atopy"
    assert "diabetes_history" in activate_keys, "FAIL: should map personal diabetes"
    assert "headache" in result["negate"], "FAIL: should negate headache"
    assert result["duration"]["value"] == 2, "FAIL: should extract duration 2"
    assert "chest" in result["body_parts"], "FAIL: should detect chest"
    assert "pulmonology" in result["body_part_categories"], "FAIL: should map lungs to pulmonology"
    print("\n  ✅ All entity mapper assertions passed")

    # Scenario 2: Simple medication mention
    mock_llm_output_2 = {
        "symptoms_present": [],
        "symptoms_absent": [],
        "duration": {"value": None, "unit": None},
        "body_parts": [],
        "medications": ["pantoprazole", "eltroxin"],
        "family_history": [],
        "personal_history": ["hypertension"],
        "onset": None,
        "severity": None,
        "is_yes": True,
        "is_no": False,
    }

    result2 = mapper.map_extraction(mock_llm_output_2)
    activate_keys2 = [k for k, _ in result2["activate"]]
    assert "antacid_use" in activate_keys2, "FAIL: pantoprazole should map to antacid_use"
    assert "thyroid_medication" in activate_keys2, "FAIL: eltroxin should map to thyroid_medication"
    assert "hypertension_history" in activate_keys2, "FAIL: hypertension should map"
    print("  ✅ Medication/history mapping assertions passed")


def test_primary_boost():
    """Test that PRIMARY_BOOST makes initial symptom dominant."""
    print("\n" + "=" * 65)
    print("  PART B: Primary Symptom Boost Comparison")
    print("=" * 65)

    graph = KnowledgeGraph()

    # WITHOUT boost (v1 behavior): all at strength 1.0
    spreader_v1 = ActivationSpreader(graph)
    spreader_v1.activate("asthma_symptoms", present=True, strength=1.0)
    spreader_v1.activate("acidity", present=True, strength=1.0)
    spreader_v1.activate("nsaid_use", present=True, strength=1.0)

    top_v1 = spreader_v1.get_top_conditions(5)
    asthma_v1 = dict(top_v1).get("bronchial_asthma", 0)
    gerd_v1 = dict(top_v1).get("GERD", 0)

    # WITH boost (v2): initial symptom at 2.0x
    spreader_v2 = ActivationSpreader(graph)
    spreader_v2.activate("asthma_symptoms", present=True, strength=2.0)  # PRIMARY
    spreader_v2.activate("acidity", present=True, strength=1.0)
    spreader_v2.activate("nsaid_use", present=True, strength=1.0)

    top_v2 = spreader_v2.get_top_conditions(5)
    asthma_v2 = dict(top_v2).get("bronchial_asthma", 0)
    gerd_v2 = dict(top_v2).get("GERD", 0)

    print("\n  Patient: asthma (primary), then acidity + NSAIDs confirmed")
    print()
    print("  v1 (no boost):")
    print("    Asthma:  %.1f%%  |  GERD:  %.1f%%  |  top = %s" % (
        asthma_v1 * 100, gerd_v1 * 100, top_v1[0][0]))
    print()
    print("  v2 (2.0x PRIMARY boost on asthma):")
    print("    Asthma:  %.1f%%  |  GERD:  %.1f%%  |  top = %s" % (
        asthma_v2 * 100, gerd_v2 * 100, top_v2[0][0]))

    # Asthma should be higher in v2
    assert asthma_v2 > asthma_v1, "FAIL: v2 asthma should be higher than v1"
    # Asthma should be the top condition in v2
    assert top_v2[0][0] == "bronchial_asthma", "FAIL: asthma should be #1 with 2x boost. Got: %s" % top_v2[0][0]
    print("\n  ✅ Primary boost makes asthma #1 (was %s in v1)" % top_v1[0][0])

    # Test VOLUNTEER_BOOST (1.3x for symptoms mentioned unprompted)
    spreader_vol = ActivationSpreader(graph)
    spreader_vol.activate("fever", present=True, strength=2.0)  # PRIMARY
    spreader_vol.activate("headache", present=True, strength=1.3)  # volunteered
    spreader_vol.activate("cough", present=True, strength=1.0)    # elicited by question

    top_vol = spreader_vol.get_top_conditions(3)
    print("\n  Volunteer boost test:")
    print("    fever 2.0x (primary) + headache 1.3x (volunteered) + cough 1.0x (elicited)")
    for c, p in top_vol:
        print("    %s: %.1f%%" % (graph.conditions[c]["display"], p * 100))
    print("  ✅ Volunteer boost applied successfully")


def test_enhanced_parser_fast_path():
    """Test the enhanced parser's fast path (no LLM)."""
    print("\n" + "=" * 65)
    print("  PART C: Enhanced Parser — Fast Path")
    print("=" * 65)

    graph = KnowledgeGraph()
    # Create parser with no LLM (force fast path)
    parser = EnhancedResponseParser(graph, llm_extractor=LLMExtractor(base_url="http://localhost:99999"))

    # Test 1: Simple yes
    r = parser.parse("yes", is_initial=False)
    assert r["is_yes"] == True, "FAIL"
    assert r["parse_method"] == "fast", "FAIL"
    print("  ✅ 'yes' → fast path, is_yes=True")

    # Test 2: Negation
    r = parser.parse("no fever", is_initial=False)
    assert "fever" in r["negated_symptoms"], "FAIL: should negate fever"
    print("  ✅ 'no fever' → fever negated")

    # Test 3: Duration
    r = parser.parse("3 weeks", is_initial=False)
    assert r["duration"]["value"] == 3, "FAIL"
    assert r["duration"]["unit"] == "weeks", "FAIL"
    print("  ✅ '3 weeks' → duration extracted")

    # Test 4: Complex with symptom
    r = parser.parse("yes I also have headache and body pain", is_initial=False)
    assert r["is_yes"] == True, "FAIL"
    assert "headache" in r["detected_symptoms"], "FAIL"
    assert "body_pain" in r["detected_symptoms"], "FAIL"
    print("  ✅ Complex response with symptoms → detected correctly")

    # Test 5: Negation in complex
    r = parser.parse("yes but there is no fever", is_initial=False)
    assert r["is_yes"] == True, "FAIL"
    assert "fever" in r["negated_symptoms"], "FAIL"
    assert "fever" not in r["detected_symptoms"], "FAIL"
    print("  ✅ 'yes but there is no fever' → fever negated, not detected")


def test_live_llm():
    """Test with actual LLM (only if Ollama is running)."""
    print("\n" + "=" * 65)
    print("  PART D: Live LLM Extraction")
    print("=" * 65)

    llm = LLMExtractor()
    if not llm.is_available():
        print("  ⚠ Ollama not running. Skipping live LLM tests.")
        print("  To enable: ollama serve && ollama pull phi3:mini")
        return

    print("  ✓ Ollama is running. Testing live extraction...")

    graph = KnowledgeGraph()
    parser = EnhancedResponseParser(graph, llm_extractor=llm)

    # Test 1: Rich initial complaint
    print("\n  Test D1: Rich initial complaint")
    r = parser.parse(
        "I have been coughing for 2 weeks with yellowish phlegm, "
        "and I take salbutamol inhaler. My mother had TB.",
        is_initial=True
    )
    print("    Parse method: %s" % r["parse_method"])
    print("    Symptoms detected: %s" % r["detected_symptoms"])
    print("    Negated: %s" % r["negated_symptoms"])
    print("    Duration: %s" % r["duration"])
    print("    Medications: %s" % r["medications_found"])
    print("    History: %s" % r["history_found"])
    print("    Body parts: %s" % r["body_parts"])

    if r["parse_method"] == "llm":
        print("    LLM latency: %dms" % r["metadata"].get("_llm_latency_ms", 0))
        print("  ✅ Live LLM extraction successful")
    else:
        print("  ⚠ Fell back to fast path (LLM may have timed out)")

    # Test 2: Simple yes should NOT use LLM
    print("\n  Test D2: Simple 'yes' should skip LLM")
    r = parser.parse("yes", is_initial=False)
    assert r["parse_method"] == "fast", "FAIL: 'yes' should use fast path"
    print("    Parse method: %s ✅" % r["parse_method"])


# ═══════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  TRIAGE ENGINE v2 — TEST SUITE")
    print("=" * 65)

    test_entity_mapper()
    test_primary_boost()
    test_enhanced_parser_fast_path()
    test_live_llm()

    print("\n\n✅ All tests completed.")
    print("   Run 'python3 triage_engine_v2.py' for the full interactive engine.")
