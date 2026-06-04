"""
Test: Clinician Feedback Loop
=============================
Simulates multiple triage sessions followed by doctor feedback,
showing how edge weights drift over time.

Scenario: The engine keeps predicting "Viral Fever" for patients who
actually have "Dengue". After 5 doctor corrections, the weights should
shift to favor Dengue more strongly.

Run:  python3 test_feedback.py
"""

import os
import sys
sys.path.insert(0, ".")

from graph_engine import KnowledgeGraph, ActivationSpreader
from question_selector import QuestionSelector
from clinician_feedback import (
    SessionLog, WeightUpdater, FeedbackStore, WeightSnapshot,
)


def simulate_session(graph, symptoms):
    """Run a minimal triage session (no questions, just activate symptoms)."""
    spreader = ActivationSpreader(graph)
    for sym in symptoms:
        spreader.activate(sym, present=True)
    return spreader


def show_edge(graph, symptom, condition):
    """Show the current weight of a specific edge."""
    for cond, w in graph.edges.get(symptom, []):
        if cond == condition:
            return w
    return 0.0


def main():
    # Clean up old feedback log
    if os.path.exists("feedback_log.json"):
        os.remove("feedback_log.json")

    graph = KnowledgeGraph()
    updater = WeightUpdater(learning_rate=0.02)
    store = FeedbackStore(filepath="feedback_log.json")

    print("=" * 65)
    print("  FEEDBACK LOOP TEST")
    print("  Simulating 10 doctor corrections over triage sessions")
    print("=" * 65)

    # ── Show initial weights ──
    print("\n  INITIAL EDGE WEIGHTS (fever → conditions):")
    fever_edges = graph.edges.get("fever", [])
    for cond, w in sorted(fever_edges, key=lambda x: -x[1])[:6]:
        display = graph.conditions.get(cond, {}).get("display", cond)
        print("    fever → %-30s  weight=%.3f" % (display, w))

    body_pain_edges = graph.edges.get("body_pain", [])
    print("\n  INITIAL EDGE WEIGHTS (body_pain → conditions):")
    for cond, w in sorted(body_pain_edges, key=lambda x: -x[1])[:5]:
        display = graph.conditions.get(cond, {}).get("display", cond)
        print("    body_pain → %-27s  weight=%.3f" % (display, w))

    # Save initial snapshot
    WeightSnapshot.save(graph, "weights_before.json")

    # ── Run 10 sessions with doctor feedback ──
    # Scenario: patient has fever + body_pain + headache
    # Engine keeps ranking viral_fever first, but doctor says it's dengue
    test_symptoms = ["fever", "body_pain", "headache", "chills"]

    print("\n" + "-" * 65)
    print("  RUNNING 10 FEEDBACK SESSIONS...")
    print("  Patient symptoms each time: %s" % test_symptoms)
    print("-" * 65)

    for i in range(10):
        spreader = simulate_session(graph, test_symptoms)
        top = spreader.get_top_conditions(3)

        session_log = SessionLog()
        session_log.session_id = "TEST_%03d" % (i + 1)
        session_log.confirmed_symptoms = list(test_symptoms)
        session_log.denied_symptoms = []
        session_log.top_conditions = top
        session_log.initial_symptoms = test_symptoms[:1]

        engine_top = top[0][0]
        engine_top_display = graph.conditions.get(engine_top, {}).get("display", engine_top)

        # Doctor says: it's dengue every time
        if engine_top == "dengue":
            # Engine got it right — confirm
            feedback = {
                "type": "confirm",
                "correct_condition": "dengue",
                "confidence": 1.0,
                "session_id": session_log.session_id,
            }
            fb_label = "CONFIRM (engine correct)"
        else:
            # Engine got it wrong — correct to dengue
            feedback = {
                "type": "correct",
                "correct_condition": "dengue",
                "engine_top_condition": engine_top,
                "confidence": 1.0,
                "session_id": session_log.session_id,
            }
            fb_label = "CORRECT (engine said %s)" % engine_top_display

        result = updater.apply_feedback(feedback, session_log, graph)
        store.save_feedback(feedback, session_log, result)

        # Show progress
        dengue_w = show_edge(graph, "fever", "dengue")
        viral_w = show_edge(graph, "fever", "viral_fever")
        malaria_w = show_edge(graph, "fever", "malaria")
        top_display = graph.conditions.get(top[0][0], {}).get("display", top[0][0])

        print("  Session %2d │ Engine top: %-20s │ %s │ edges: dengue=%.3f viral=%.3f malaria=%.3f │ changes: %d" % (
            i + 1, top_display, fb_label[:20], dengue_w, viral_w, malaria_w, result["num_changes"]))

    # ── Show final weights ──
    print("\n" + "-" * 65)
    print("  AFTER 10 FEEDBACK SESSIONS:")
    print("-" * 65)

    print("\n  FINAL EDGE WEIGHTS (fever → conditions):")
    fever_edges = graph.edges.get("fever", [])
    for cond, w in sorted(fever_edges, key=lambda x: -x[1])[:6]:
        display = graph.conditions.get(cond, {}).get("display", cond)
        print("    fever → %-30s  weight=%.3f" % (display, w))

    print("\n  FINAL EDGE WEIGHTS (body_pain → conditions):")
    body_pain_edges = graph.edges.get("body_pain", [])
    for cond, w in sorted(body_pain_edges, key=lambda x: -x[1])[:5]:
        display = graph.conditions.get(cond, {}).get("display", cond)
        print("    body_pain → %-27s  weight=%.3f" % (display, w))

    # ── Run one more session to see if the engine now predicts correctly ──
    print("\n" + "-" * 65)
    print("  VERIFICATION: Same symptoms after calibration")
    print("-" * 65)
    spreader = simulate_session(graph, test_symptoms)
    top = spreader.get_top_conditions(5)
    for i, (cond, prob) in enumerate(top):
        display = graph.conditions.get(cond, {}).get("display", cond)
        rank = "★ PRIMARY" if i == 0 else "  #%d" % (i + 1)
        print("  %s: %s (%.1f%%)" % (rank, display, prob * 100))

    # ── Show diff ──
    print("\n" + "-" * 65)
    print("  WEIGHT DIFF (initial → final):")
    WeightSnapshot.diff(graph, "weights_before.json")

    # ── Show feedback stats ──
    stats = store.get_stats()
    print("\n  FEEDBACK STORE STATS:")
    print("    Sessions reviewed: %d" % stats["total_sessions_reviewed"])
    print("    Confirms: %d" % stats["confirms"])
    print("    Corrections: %d" % stats["corrections"])
    print("    Total edge changes: %d" % stats["total_edge_changes"])

    # ── Edge history for one specific edge ──
    history = store.get_edge_history("fever", "dengue")
    if history:
        print("\n  EDGE HISTORY: fever → dengue")
        for h in history[:5]:
            print("    %s: %.3f → %.3f (%s%.3f) [%s]" % (
                h["timestamp"], h["old"], h["new"],
                "+" if h["delta"] > 0 else "", h["delta"], h["action"]))

    # Clean up
    for f in ["weights_before.json", "feedback_log.json"]:
        if os.path.exists(f):
            os.remove(f)

    print("\n✅ Feedback loop test complete.")


if __name__ == "__main__":
    main()
