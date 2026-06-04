"""
Triage Engine
=============
Orchestrates the full triage conversation loop:
1. Accept initial symptoms
2. Activate knowledge graph
3. Select best follow-up question (max information gain)
4. Parse patient response → update graph
5. Repeat until confident or max questions reached
6. Output: top conditions, severity, routing recommendation

Usage:
    python triage_engine.py
"""

import sys
from graph_engine import KnowledgeGraph, ActivationSpreader
from question_selector import QuestionSelector
from response_parser import ResponseParser
from clinician_feedback import (
    SessionLog, FeedbackCollector, WeightUpdater,
    FeedbackStore, WeightSnapshot,
)


# ─── Configuration ───
MAX_QUESTIONS = 9
CONFIDENCE_THRESHOLD = 0.30   # stop if top condition probability > this
ENTROPY_THRESHOLD = 3.0       # stop if entropy drops below this (low = confident)
MIN_INFO_GAIN = 0.005         # skip question if IG is negligible


def print_header():
    print("\n" + "=" * 65)
    print("  CLINICAL TRIAGE ENGINE  —  Knowledge Graph + Info Gain")
    print("=" * 65)
    print("  How it works:")
    print("  • Enter a symptom to start (e.g., 'fever', 'headache')")
    print("  • The engine asks follow-up questions (max 9)")
    print("  • Respond with:")
    print("      - 'yes' / 'no' / symptom name / duration (e.g. '3 days')")
    print("      - Press ENTER (empty) = skip / no")
    print("  • Engine outputs top conditions + routing")
    print("=" * 65)


def print_distribution(spreader: ActivationSpreader, graph: KnowledgeGraph, n=5):
    """Print the current top conditions with probabilities."""
    top = spreader.get_top_conditions(n)
    H = spreader.get_entropy()
    print(f"\n  ┌─ Current Assessment (entropy={H:.2f} bits) ───")
    for i, (cond, prob) in enumerate(top):
        cond_info = graph.conditions.get(cond, {})
        display = cond_info.get("display", cond)
        severity = cond_info.get("severity", "?")
        bar_len = int(prob * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        sev_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
        print(f"  │ {i+1}. {bar} {prob*100:5.1f}%  {sev_icon} {display}")
    print(f"  └{'─' * 55}")


def print_final_result(spreader: ActivationSpreader, graph: KnowledgeGraph):
    """Print the final triage result."""
    top = spreader.get_top_conditions(3)
    print("\n" + "=" * 65)
    print("  TRIAGE RESULT")
    print("=" * 65)

    for i, (cond, prob) in enumerate(top):
        cond_info = graph.conditions.get(cond, {})
        display = cond_info.get("display", cond)
        severity = cond_info.get("severity", "?")
        route = cond_info.get("route_to", "general_medicine")
        sev_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")

        print(f"\n  {'▸ PRIMARY' if i == 0 else '  Differential'} #{i+1}:")
        print(f"    Condition  : {display}")
        print(f"    Confidence : {prob*100:.1f}%")
        print(f"    Severity   : {sev_icon} {severity.upper()}")
        print(f"    Route to   : {route.replace('_', ' ').title()}")

    # Active symptoms summary
    print(f"\n  Evidence collected:")
    print(f"    ✓ Confirmed : {', '.join(sorted(spreader.active_symptoms)) or 'none'}")
    print(f"    ✗ Denied    : {', '.join(sorted(spreader.negative_symptoms)) or 'none'}")

    if spreader.metadata:
        print(f"    ⏱ Durations : {spreader.metadata}")

    # Urgency flag
    top_cond = top[0][0] if top else None
    if top_cond:
        sev = graph.conditions.get(top_cond, {}).get("severity", "low")
        if sev == "critical":
            print(f"\n  ⚠️  URGENT: This patient may need EMERGENCY attention.")
        elif sev == "high":
            print(f"\n  ⚠️  PRIORITY: Schedule this patient promptly.")

    print("\n" + "=" * 65)


def run_triage():
    """Main triage loop."""
    # Initialize components
    graph = KnowledgeGraph()

    # ───────────────────────────────────────
    # LOAD CALIBRATED WEIGHTS (from past feedback)
    # ───────────────────────────────────────
    import os
    snapshot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrated_weights.json")
    if os.path.exists(snapshot_path):
        loaded = WeightSnapshot.load(graph, snapshot_path)
        if loaded:
            store = FeedbackStore(filepath=os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_log.json"))
            stats = store.get_stats()
            print("  (Calibrated from %d doctor reviews)" % stats["total_sessions_reviewed"])
    
    spreader = ActivationSpreader(graph)
    parser = ResponseParser(graph)

    print_header()

    # ───────────────────────────────────────
    # DEMOGRAPHICS: age and gender (optional)
    # ───────────────────────────────────────
    print("\n  Patient demographics (press ENTER to skip):")
    try:
        age_input = input("  Age: ").strip()
        gender_input = input("  Gender (male/female): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Session ended.")
        return

    patient_age = None
    patient_gender = None
    if age_input.isdigit():
        patient_age = int(age_input)
    if gender_input in ("male", "female", "m", "f"):
        patient_gender = "female" if gender_input.startswith("f") else "male"

    selector = QuestionSelector(graph, patient_age=patient_age, patient_gender=patient_gender)

    if patient_age or patient_gender:
        parts = []
        if patient_age:
            parts.append(f"age {patient_age}")
        if patient_gender:
            parts.append(patient_gender)
        print(f"  ✓ Demographics: {', '.join(parts)}")

    # ───────────────────────────────────────
    # ITERATION 0: Get initial symptom(s)
    # ───────────────────────────────────────
    print("\n  [Iteration 0] Enter your primary symptom:")
    try:
        initial_input = input("  ▸ ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Session ended.")
        return

    if not initial_input:
        print("  No symptom entered. Exiting.")
        return

    # Parse initial input — could be one or more symptoms
    initial_parse = parser.parse(initial_input)

    # Try to resolve as a known symptom
    resolved = graph.resolve_symptom(initial_input)
    if resolved:
        initial_parse["detected_symptoms"].append(resolved)

    # Deduplicate
    initial_symptoms = list(set(initial_parse["detected_symptoms"]))

    if not initial_symptoms:
        # Try harder: split by comma or 'and'
        parts = [p.strip() for p in initial_input.replace(" and ", ",").split(",")]
        for part in parts:
            r = graph.resolve_symptom(part)
            if r:
                initial_symptoms.append(r)

    if not initial_symptoms:
        print(f"  ⚠ Could not recognize '{initial_input}' as a known symptom.")
        print(f"  Known symptoms: {', '.join(sorted(graph.aliases.keys())[:20])}...")
        return

    # Activate initial symptoms
    print(f"\n  ✓ Recognized symptoms: {', '.join(initial_symptoms)}")
    for sym in initial_symptoms:
        spreader.activate(sym, present=True)

    if initial_parse["duration"]:
        spreader.metadata["initial_duration"] = initial_parse["duration"]
        print(f"  ⏱ Duration noted: {initial_parse['duration']}")

    print_distribution(spreader, graph)

    # ───────────────────────────────────────
    # ITERATIONS 1–9: Follow-up questions
    # ───────────────────────────────────────
    asked_ids = set()
    questions_log = []   # for clinician feedback system

    for iteration in range(1, MAX_QUESTIONS + 1):
        # Check stopping criteria
        top = spreader.get_top_conditions(1)
        if top and top[0][1] > CONFIDENCE_THRESHOLD:
            H = spreader.get_entropy()
            if H < ENTROPY_THRESHOLD:
                print(f"\n  ✓ Confidence reached at iteration {iteration}. Stopping.")
                break

        # Select next question
        question, ig = selector.select_next(spreader, asked_ids)

        if question is None or ig < MIN_INFO_GAIN:
            print(f"\n  ✓ No more useful questions. Stopping at iteration {iteration}.")
            break

        asked_ids.add(question["id"])

        # Ask the question
        print(f"\n  [Iteration {iteration}] ({question['id']}, IG={ig:.4f})")
        print(f"  Q: {question['text']}")

        try:
            response = input("  ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Session ended early.")
            break

        # Parse response
        parsed = parser.parse(response, current_question=question)

        # Log for feedback system
        questions_log.append({
            "id": question["id"],
            "text": question["text"],
            "response": response,
        })

        elicited_symptom = question["elicits"]

        if parsed["is_null"]:
            # Null = skip/no → negative evidence for elicited symptom
            spreader.activate(elicited_symptom, present=False)
            print(f"    → Skipped (negative evidence for '{elicited_symptom}')")

        else:
            # Process yes/no
            if parsed["is_yes"]:
                spreader.activate(elicited_symptom, present=True)
                print(f"    → Confirmed: '{elicited_symptom}'")
            elif parsed["is_no"]:
                spreader.activate(elicited_symptom, present=False)
                print(f"    → Denied: '{elicited_symptom}'")

            # Process any new symptoms mentioned in response
            for sym in parsed["detected_symptoms"]:
                if sym not in spreader.active_symptoms:
                    spreader.activate(sym, present=True)
                    print(f"    → New symptom detected: '{sym}'")

            # Process negated symptoms (e.g., "no fever", "without headache")
            for sym in parsed.get("negated_symptoms", []):
                if sym not in spreader.negative_symptoms:
                    spreader.activate(sym, present=False)
                    print(f"    → Negated symptom: '{sym}'")

            # Process matched findings from response_map
            for finding in parsed["matched_findings"]:
                if finding.startswith("hint_"):
                    continue  # condition hints, not direct activation
                if finding not in spreader.active_symptoms:
                    spreader.activate(finding, present=True)
                    print(f"    → Finding activated: '{finding}'")

            # Process duration
            if parsed["duration"]:
                key = f"{elicited_symptom}_duration"
                spreader.metadata[key] = parsed["duration"]
                print(f"    ⏱ Duration: {parsed['duration']}")

                # Duration-based severity adjustment
                # Longer durations for certain symptoms boost chronic conditions
                duration_text = parsed["duration"]
                try:
                    num = int(''.join(filter(str.isdigit, duration_text)))
                    if "week" in duration_text:
                        num *= 7
                    elif "month" in duration_text:
                        num *= 30
                    elif "year" in duration_text:
                        num *= 365

                    # Chronic boost if duration > 14 days
                    if num > 14:
                        chronic_conditions = ["tuberculosis", "COPD", "IBD",
                                              "rheumatoid_arthritis", "thyroid_disorder"]
                        for c in chronic_conditions:
                            if spreader.activations.get(c, 0) > 0:
                                spreader.activations[c] *= 1.2  # 20% boost
                                print(f"    → Chronic duration boost for '{c}'")
                except (ValueError, IndexError):
                    pass

        # Show updated distribution
        print_distribution(spreader, graph)

    # ───────────────────────────────────────
    # FINAL OUTPUT
    # ───────────────────────────────────────
    print_final_result(spreader, graph)

    # ───────────────────────────────────────
    # CLINICIAN FEEDBACK (optional)
    # ───────────────────────────────────────
    print("\n  ─── CLINICIAN REVIEW ───")
    print("  Would a doctor like to review this result? (y/n)")
    try:
        review = input("  ▸ ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        review = "n"

    if review in ("y", "yes"):
        # Build session log
        session_log = SessionLog()
        session_log.capture(spreader, graph, questions_log,
                           patient_info={"age": patient_age, "gender": patient_gender})
        session_log.initial_symptoms = list(initial_symptoms)

        # Collect feedback
        collector = FeedbackCollector()
        feedback = collector.collect(session_log, graph)

        if feedback:
            # Apply weight updates
            updater = WeightUpdater(learning_rate=0.02)
            result = updater.apply_feedback(feedback, session_log, graph)

            # Save to persistent log
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback_log.json")
            store = FeedbackStore(filepath=log_path)
            store.save_feedback(feedback, session_log, result)

            # *** PERSIST the updated weights so next run uses them ***
            WeightSnapshot.save(graph, snapshot_path)

            # Show what changed
            print("\n  ─── WEIGHT UPDATE SUMMARY ───")
            print("  Feedback type: %s" % feedback["type"])
            print("  Edges adjusted: %d" % result["num_changes"])
            for ch in result.get("changes", [])[:10]:
                print("    %s: %s → %s: %.3f → %.3f (%s)" % (
                    ch["action"], ch["symptom"], ch["condition"],
                    ch["old_weight"], ch["new_weight"],
                    "▲" if ch["delta"] > 0 else "▼"
                ))

            # Show cumulative stats
            stats = store.get_stats()
            print("\n  ─── CUMULATIVE STATS ───")
            print("  Sessions reviewed: %d" % stats["total_sessions_reviewed"])
            print("  Confirms: %d | Corrections: %d | Reranks: %d" % (
                stats["confirms"], stats["corrections"], stats["reranks"]))
            print("  Total edge changes: %d" % stats["total_edge_changes"])
        else:
            print("  Skipped.")


if __name__ == "__main__":
    run_triage()