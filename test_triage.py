"""
Automated Test Harness for Triage Engine
=========================================
Simulates 3 patient scenarios end-to-end WITHOUT requiring manual input.
Each scenario provides scripted responses to the engine's dynamically-chosen questions.

Run:  python3 test_triage.py
"""

import sys
sys.path.insert(0, ".")

from graph_engine import KnowledgeGraph, ActivationSpreader
from question_selector import QuestionSelector
from response_parser import ResponseParser


# ─────────────────────────────────────────────
# Configuration (same as triage_engine.py)
# ─────────────────────────────────────────────
MAX_QUESTIONS = 9
CONFIDENCE_THRESHOLD = 0.30
ENTROPY_THRESHOLD = 3.0
MIN_INFO_GAIN = 0.005


def simulate_triage(scenario_name, initial_symptom, responses, patient_age=None, patient_gender=None):
    """
    Run one triage session.

    Args:
        scenario_name: label for this test case
        initial_symptom: what the patient says first (e.g., "fever")
        responses: dict mapping question_id → simulated patient response.
        patient_age: optional int for demographic filtering
        patient_gender: optional "male" or "female"
    """
    graph = KnowledgeGraph()
    spreader = ActivationSpreader(graph)
    selector = QuestionSelector(graph, patient_age=patient_age, patient_gender=patient_gender)
    parser = ResponseParser(graph)

    demo_parts = []
    if patient_age:
        demo_parts.append("age %d" % patient_age)
    if patient_gender:
        demo_parts.append(patient_gender)
    demo_str = " | Patient: %s" % ", ".join(demo_parts) if demo_parts else ""

    print(f"\n{'='*65}")
    print(f"  SCENARIO: {scenario_name}{demo_str}")
    print(f"  Initial symptom: '{initial_symptom}'")
    print(f"{'='*65}")

    # ── Iteration 0: initial symptom ──
    resolved = graph.resolve_symptom(initial_symptom)
    init_parse = parser.parse(initial_symptom)

    all_initial = list(set(init_parse["detected_symptoms"] + ([resolved] if resolved else [])))
    if not all_initial:
        print(f"  ✗ FAIL: Could not recognize '{initial_symptom}'")
        return

    for sym in all_initial:
        spreader.activate(sym, present=True)
    print(f"  [Iter 0] Activated: {all_initial}")
    _show_top(spreader, graph)

    # ── Iterations 1–9 ──
    asked_ids = set()
    for iteration in range(1, MAX_QUESTIONS + 1):
        # Check stopping criteria
        top = spreader.get_top_conditions(1)
        if top and top[0][1] > CONFIDENCE_THRESHOLD:
            H = spreader.get_entropy()
            if H < ENTROPY_THRESHOLD:
                print(f"  [Iter {iteration}] ✓ Confidence threshold reached. Stopping.")
                break

        question, ig = selector.select_next(spreader, asked_ids)
        if question is None or ig < MIN_INFO_GAIN:
            print(f"  [Iter {iteration}] ✓ No more useful questions. Stopping.")
            break

        asked_ids.add(question["id"])

        # Get simulated response
        sim_response = responses.get(question["id"], "")  # default: empty = skip
        print(f"  [Iter {iteration}] Q({question['id']}): {question['text']}")
        print(f"           Response: '{sim_response}' {'(skip)' if not sim_response else ''}")

        parsed = parser.parse(sim_response, current_question=question)
        elicited = question["elicits"]

        if parsed["is_null"]:
            spreader.activate(elicited, present=False)
        else:
            if parsed["is_yes"]:
                spreader.activate(elicited, present=True)
            elif parsed["is_no"]:
                spreader.activate(elicited, present=False)

            for sym in parsed["detected_symptoms"]:
                if sym not in spreader.active_symptoms:
                    spreader.activate(sym, present=True)

            for sym in parsed.get("negated_symptoms", []):
                if sym not in spreader.negative_symptoms:
                    spreader.activate(sym, present=False)

            for finding in parsed["matched_findings"]:
                if not finding.startswith("hint_") and finding not in spreader.active_symptoms:
                    spreader.activate(finding, present=True)

            if parsed["duration"]:
                spreader.metadata[f"{elicited}_duration"] = parsed["duration"]

    # ── Final result ──
    _show_result(spreader, graph)


def _show_top(spreader, graph, n=5):
    top = spreader.get_top_conditions(n)
    H = spreader.get_entropy()
    print(f"  Top conditions (H={H:.2f}):")
    for i, (c, p) in enumerate(top):
        d = graph.conditions[c]["display"]
        s = graph.conditions[c]["severity"]
        print(f"    {i+1}. {p*100:5.1f}%  [{s:8s}]  {d}")


def _show_result(spreader, graph):
    top = spreader.get_top_conditions(3)
    print(f"\n  ── TRIAGE RESULT ──")
    for i, (c, p) in enumerate(top):
        info = graph.conditions[c]
        icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(info["severity"], "⚪")
        tag = "PRIMARY" if i == 0 else f"Diff #{i+1}"
        print(f"  {tag}: {info['display']}  |  {p*100:.1f}%  |  {icon} {info['severity'].upper()}  |  → {info['route_to'].replace('_',' ').title()}")
    print(f"  Confirmed: {sorted(spreader.active_symptoms)}")
    print(f"  Denied:    {sorted(spreader.negative_symptoms)}")
    if spreader.metadata:
        print(f"  Metadata:  {spreader.metadata}")


# ═══════════════════════════════════════════════
# TEST SCENARIOS
# ═══════════════════════════════════════════════

def test_scenario_1_dengue():
    """Patient with fever → dengue-like symptoms."""
    simulate_triage(
        scenario_name="Suspected Dengue Fever",
        initial_symptom="fever",
        responses={
            # Expect questions about fever cluster: duration, chills, body pain, rash, headache, etc.
            "Q001": "5 days",           # fever duration
            "Q003": "yes",              # chills
            "Q004": "yes body pain",    # body pain
            "Q005": "yes red spots",    # rash/spots
            "Q006": "yes behind eyes",  # headache behind eyes
            "Q011": "yes gums",         # bleeding gums
            "Q079": "yes",              # easy bruising
            "Q007": "no",               # travel history
            "Q009": "no",               # cough
        }
    )


def test_scenario_2_gerd():
    """Patient with acidity → GERD path."""
    simulate_triage(
        scenario_name="Suspected GERD",
        initial_symptom="acidity",
        responses={
            "Q024": "yes after eating",   # postprandial burn
            "Q025": "yes bloated",        # bloating
            "Q023": "yes burning",        # burning in chest
            "Q032": "yes ibuprofen",      # NSAID use
            "Q033": "no",                 # alcohol
            "Q026": "no",                 # bowel change
            "Q027": "no",                 # blood in stool
            "Q012": "yes",               # nausea
            "Q028": "no",                 # appendicitis pain
        }
    )


def test_scenario_3_anxiety_thyroid():
    """Patient with anxiety → could be GAD or hyperthyroidism."""
    simulate_triage(
        scenario_name="Anxiety — GAD vs Hyperthyroidism",
        initial_symptom="anxiety",
        responses={
            "Q071": "yes worried",        # excessive worry
            "Q072": "yes",                # insomnia
            "Q073": "no",                 # panic attacks
            "Q050": "yes fast heart",     # palpitations
            "Q075": "yes",                # tremor
            "Q076": "yes",                # heat intolerance
            "Q077": "yes swelling neck",  # neck swelling
            "Q031": "yes",                # weight loss
            "Q081": "yes tired",          # fatigue
        }
    )


def test_scenario_4_negation_regression():
    """
    REGRESSION TEST: Patient says "yes but there is no fever".
    The word 'fever' should be NEGATED, not activated as present.
    """
    from response_parser import ResponseParser
    from graph_engine import KnowledgeGraph
    graph = KnowledgeGraph()
    parser = ResponseParser(graph)

    # The exact input that caused the bug
    q = graph.questions["Q003"]  # "Do you have chills or shivering?"
    parsed = parser.parse("yes but there is no fever", current_question=q)

    print("\n" + "=" * 65)
    print("  REGRESSION TEST: Negation Detection")
    print("=" * 65)
    print(f"  Input: 'yes but there is no fever'")
    print(f"  is_yes:   {parsed['is_yes']}  (expected: True)")
    print(f"  detected: {parsed['detected_symptoms']}  (expected: [] or ['chills'] — NOT 'fever')")
    print(f"  negated:  {parsed['negated_symptoms']}  (expected: ['fever'])")

    assert parsed["is_yes"] == True, "FAIL: should detect 'yes'"
    assert "fever" not in parsed["detected_symptoms"], \
        f"FAIL: 'fever' in detected_symptoms — negation not working! Got: {parsed['detected_symptoms']}"
    assert "fever" in parsed["negated_symptoms"], \
        f"FAIL: 'fever' should be in negated_symptoms! Got: {parsed['negated_symptoms']}"
    print("  ✅ PASSED — fever correctly negated, not activated")

    # Additional negation cases
    cases = [
        ("no headache",       "headache", False),
        ("without nausea",    "nausea",   False),
        ("I don't have rash", "rash",     False),
        ("yes headache",      "headache", True),
        ("headache and fever", "fever",   True),
    ]
    all_ok = True
    for text, symptom, should_detect in cases:
        p = parser.parse(text)
        detected = symptom in p["detected_symptoms"]
        negated = symptom in p["negated_symptoms"]
        ok = (detected == should_detect) and (negated != should_detect)
        status = "✅" if ok else "❌"
        print(f"  {status} '{text}' → detected={detected}, negated={negated} (expect detected={should_detect})")
        if not ok:
            all_ok = False

    if all_ok:
        print("  ✅ ALL negation tests passed")
    else:
        print("  ❌ SOME negation tests failed")


# ═══════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  TRIAGE ENGINE — AUTOMATED TEST SUITE")
    print("  5 scenarios + negation regression tests")
    print("=" * 65)

    test_scenario_4_negation_regression()

    # Male, age 35 — should NOT get Q103 (menstrual/female), Q105 (pregnancy), Q112 (breastfeeding)
    test_scenario_1_dengue.__doc__ = "Dengue — male patient, age 35"
    simulate_triage(
        scenario_name="Suspected Dengue (male, 35)",
        initial_symptom="fever",
        patient_age=35,
        patient_gender="male",
        responses={
            "Q001": "5 days",
            "Q003": "yes",
            "Q004": "yes body pain",
            "Q005": "yes red spots",
            "Q006": "yes behind eyes",
            "Q011": "yes gums",
            "Q079": "yes",
            "Q007": "no",
            "Q009": "no",
        }
    )

    # Female, age 28 — GERD; should get Q105 (pregnancy check) if relevant
    simulate_triage(
        scenario_name="Suspected GERD (female, 28)",
        initial_symptom="acidity",
        patient_age=28,
        patient_gender="female",
        responses={
            "Q024": "yes after eating",
            "Q025": "yes bloated",
            "Q023": "yes burning",
            "Q032": "yes ibuprofen",
            "Q033": "no",
            "Q026": "no",
            "Q027": "no",
            "Q012": "yes",
            "Q028": "no",
            "Q105": "no",    # pregnancy check — should appear for female age 28
        }
    )

    # Male, 45 — anxiety path; should NOT get menstrual/pregnancy questions
    simulate_triage(
        scenario_name="Anxiety — GAD vs Hyperthyroidism (male, 45)",
        initial_symptom="anxiety",
        patient_age=45,
        patient_gender="male",
        responses={
            "Q071": "yes worried",
            "Q072": "yes",
            "Q073": "no",
            "Q050": "yes fast heart",
            "Q075": "yes",
            "Q076": "yes",
            "Q077": "yes swelling neck",
            "Q031": "yes",
            "Q081": "yes tired",
        }
    )

    # ── DEMOGRAPHIC FILTER VERIFICATION ──
    print(f"\n{'='*65}")
    print("  DEMOGRAPHIC FILTER TEST")
    print(f"{'='*65}")

    from question_selector import QuestionSelector as QS
    from graph_engine import KnowledgeGraph as KG, ActivationSpreader as AS

    graph = KG()
    spreader = AS(graph)
    spreader.activate("acne", present=True)

    # Male selector — Q103 (menstrual acne, female-only) should be ineligible
    sel_male = QS(graph, patient_age=30, patient_gender="male")
    q103 = graph.questions.get("Q103")
    if q103:
        eligible = sel_male._is_eligible(q103, spreader)
        print(f"  Q103 (menstrual acne) for male, 30: eligible={eligible} (expected: False) {'✅' if not eligible else '❌'}")

    # Female selector — Q103 should be eligible (acne is active, female, age 30)
    sel_female = QS(graph, patient_age=30, patient_gender="female")
    if q103:
        eligible = sel_female._is_eligible(q103, spreader)
        print(f"  Q103 (menstrual acne) for female, 30: eligible={eligible} (expected: True) {'✅' if eligible else '❌'}")

    # Q106 (vaccination, age 1-18) for age 30 — should be ineligible
    q106 = graph.questions.get("Q106")
    spreader2 = AS(graph)
    spreader2.activate("fever", present=True)
    sel_adult = QS(graph, patient_age=30, patient_gender="male")
    if q106:
        eligible = sel_adult._is_eligible(q106, spreader2)
        print(f"  Q106 (vaccination) for male, 30: eligible={eligible} (expected: False) {'✅' if not eligible else '❌'}")

    # Q106 for age 10 — should be eligible
    sel_child = QS(graph, patient_age=10, patient_gender="male")
    if q106:
        eligible = sel_child._is_eligible(q106, spreader2)
        print(f"  Q106 (vaccination) for male, 10: eligible={eligible} (expected: True) {'✅' if eligible else '❌'}")

    # Q108 (cough type) requires_symptoms=["cough"] — shouldn't fire without cough active
    q108 = graph.questions.get("Q108")
    spreader_no_cough = AS(graph)
    spreader_no_cough.activate("fever", present=True)
    sel_any = QS(graph)
    if q108:
        eligible = sel_any._is_eligible(q108, spreader_no_cough)
        print(f"  Q108 (cough type) without cough active: eligible={eligible} (expected: False) {'✅' if not eligible else '❌'}")

    spreader_cough = AS(graph)
    spreader_cough.activate("cough", present=True)
    if q108:
        eligible = sel_any._is_eligible(q108, spreader_cough)
        print(f"  Q108 (cough type) with cough active: eligible={eligible} (expected: True) {'✅' if eligible else '❌'}")

    print("\n\n✅ All scenarios + demographic tests completed.")
    print("   Run 'python3 triage_engine.py' for interactive mode.")
