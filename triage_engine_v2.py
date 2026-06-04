"""
Triage Engine v2 — LLM-Enhanced
================================
Full pipeline:
  1. Patient demographics
  2. Chief complaint → LLM extraction → KG activation at 2.0x PRIMARY BOOST
  3. Follow-up questions → LLM or fast parse → KG update at 1.0x
  4. Body-part-aware, medication-aware question selection
  5. Auto-skip questions whose findings are already known from LLM
  6. Final triage result + clinician feedback

New vs v1:
  - PRIMARY_BOOST: initial symptom activated at 2.0x strength
  - VOLUNTEER_BOOST: symptoms patient mentions unprompted get 1.3x
  - LLM extraction for complex responses (4+ words)
  - Body part filtering in question selection
  - Auto-activation: if LLM detects "inhaler", skip asking "do you use an inhaler?"
  - Medication → condition edges (inhaler_use → bronchial_asthma boost)

Usage:
    python3 triage_engine_v2.py
"""

import os
import sys
from graph_engine import KnowledgeGraph, ActivationSpreader
from question_selector import QuestionSelector
from enhanced_parser import EnhancedResponseParser
from llm_extractor import LLMExtractor
from entity_mapper import EntityMapper
from question_dedup import ExtraMedicalInfo, QuestionDedupGate
from smart_resolver import SmartSymptomResolver
from clinician_feedback import (
    SessionLog, FeedbackCollector, WeightUpdater,
    FeedbackStore, WeightSnapshot,
)


# ─── Configuration ───
MAX_QUESTIONS = 9
CONFIDENCE_THRESHOLD = 0.30
ENTROPY_THRESHOLD = 3.0
MIN_INFO_GAIN = 0.005
PRIMARY_BOOST = 2.0     # initial symptom gets 2x activation
VOLUNTEER_BOOST = 1.3   # symptoms mentioned unprompted by patient get 1.3x


def print_header(llm_available):
    print("\n" + "=" * 65)
    print("  CLINICAL TRIAGE ENGINE v2  —  KG + LLM Extraction")
    print("=" * 65)
    if llm_available:
        print("  ✓ LLM extraction: ONLINE (complex responses will use LLM)")
    else:
        print("  ⚠ LLM extraction: OFFLINE (using keyword parser only)")
        print("    To enable: run 'ollama serve' and 'ollama pull phi3:mini'")
    print("  How it works:")
    print("  • Enter your chief complaint (describe freely)")
    print("  • Answer follow-up questions (max %d)" % MAX_QUESTIONS)
    print("  • Respond naturally — the system extracts medical info")
    print("=" * 65)


def print_distribution(spreader, graph, n=5):
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


def print_final_result(spreader, graph):
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

        tag = "▸ PRIMARY" if i == 0 else "  Differential"
        print(f"\n  {tag} #{i+1}:")
        print(f"    Condition  : {display}")
        print(f"    Confidence : {prob*100:.1f}%")
        print(f"    Severity   : {sev_icon} {severity.upper()}")
        print(f"    Route to   : {route.replace('_', ' ').title()}")

    print(f"\n  Evidence collected:")
    print(f"    ✓ Confirmed : {', '.join(sorted(spreader.active_symptoms)) or 'none'}")
    print(f"    ✗ Denied    : {', '.join(sorted(spreader.negative_symptoms)) or 'none'}")
    if spreader.metadata:
        print(f"    ⏱ Meta      : {spreader.metadata}")

    top_cond = top[0][0] if top else None
    if top_cond:
        sev = graph.conditions.get(top_cond, {}).get("severity", "low")
        if sev == "critical":
            print(f"\n  ⚠️  URGENT: This patient may need EMERGENCY attention.")
        elif sev == "high":
            print(f"\n  ⚠️  PRIORITY: Schedule this patient promptly.")

    print("\n" + "=" * 65)


def apply_parsed_response(parsed, spreader, graph, elicited_symptom, is_initial=False):
    """
    Apply a parsed response to the activation spreader.
    Handles all the activation logic in one place.
    Returns a list of log strings for display.
    """
    logs = []

    # Determine activation strength
    if is_initial:
        strength = PRIMARY_BOOST
        strength_label = "PRIMARY 2.0x"
    else:
        strength = 1.0
        strength_label = "1.0x"

    if parsed["is_null"]:
        spreader.activate(elicited_symptom, present=False)
        logs.append("→ Skipped (negative evidence for '%s')" % elicited_symptom)
        return logs

    # Yes/No for elicited symptom
    if parsed["is_yes"]:
        spreader.activate(elicited_symptom, present=True, strength=strength)
        logs.append("→ Confirmed: '%s' [%s]" % (elicited_symptom, strength_label))
    elif parsed["is_no"]:
        spreader.activate(elicited_symptom, present=False)
        logs.append("→ Denied: '%s'" % elicited_symptom)

    # Detected symptoms (from LLM or keyword match)
    for sym in parsed.get("detected_symptoms", []):
        if sym == elicited_symptom:
            continue  # already handled above
        if sym not in spreader.active_symptoms:
            # Volunteered symptoms get VOLUNTEER_BOOST if not initial
            vol_strength = VOLUNTEER_BOOST if not is_initial else PRIMARY_BOOST
            spreader.activate(sym, present=True, strength=vol_strength)
            logs.append("→ Detected: '%s' [%.1fx]" % (sym, vol_strength))

    # Negated symptoms
    for sym in parsed.get("negated_symptoms", []):
        if sym not in spreader.negative_symptoms:
            spreader.activate(sym, present=False)
            logs.append("→ Negated: '%s'" % sym)

    # Matched findings from response_map
    for finding in parsed.get("matched_findings", []):
        if finding.startswith("hint_"):
            continue
        if finding not in spreader.active_symptoms:
            spreader.activate(finding, present=True)
            logs.append("→ Finding: '%s'" % finding)

    # Duration
    dur = parsed.get("duration")
    if dur and dur.get("value") is not None:
        key = "%s_duration" % elicited_symptom
        dur_str = "%s %s" % (dur["value"], dur["unit"] or "days")
        spreader.metadata[key] = dur_str
        logs.append("⏱ Duration: %s" % dur_str)

        # Chronic boost if > 14 days
        days = dur["value"]
        unit = dur.get("unit", "days")
        if unit in ("weeks", "week"):
            days *= 7
        elif unit in ("months", "month"):
            days *= 30
        elif unit in ("years", "year"):
            days *= 365

        if days > 14:
            chronic_conds = ["tuberculosis", "COPD", "IBD",
                           "rheumatoid_arthritis", "thyroid_disorder",
                           "osteoarthritis"]
            for c in chronic_conds:
                if spreader.activations.get(c, 0) > 0:
                    spreader.activations[c] *= 1.2
                    logs.append("→ Chronic boost for '%s'" % c)

    # Body parts (store as metadata for question selector)
    for bp in parsed.get("body_parts", []):
        spreader.metadata.setdefault("body_parts", [])
        if bp not in spreader.metadata["body_parts"]:
            spreader.metadata["body_parts"].append(bp)

    for cat in parsed.get("body_part_categories", []):
        spreader.metadata.setdefault("body_part_categories", [])
        if cat not in spreader.metadata["body_part_categories"]:
            spreader.metadata["body_part_categories"].append(cat)

    # Medications (from LLM extraction)
    for med in parsed.get("medications_found", []):
        if med not in spreader.active_symptoms:
            spreader.activate(med, present=True)
            logs.append("→ Medication: '%s'" % med)

    # History findings (from LLM extraction)
    for hist in parsed.get("history_found", []):
        if hist not in spreader.active_symptoms:
            spreader.activate(hist, present=True)
            logs.append("→ History: '%s'" % hist)

    # Parse method indicator
    method = parsed.get("parse_method", "fast")
    latency = parsed.get("metadata", {}).get("_llm_latency_ms", 0)
    if method == "llm":
        logs.append("  [parsed via LLM, %dms]" % latency)

    return logs


def run_triage():
    """Main triage loop with LLM integration."""
    # Initialize graph
    graph = KnowledgeGraph()

    # Load calibrated weights from past feedback
    script_dir = os.path.dirname(os.path.abspath(__file__))
    snapshot_path = os.path.join(script_dir, "calibrated_weights.json")
    if os.path.exists(snapshot_path):
        WeightSnapshot.load(graph, snapshot_path)
        store = FeedbackStore(filepath=os.path.join(script_dir, "feedback_log.json"))
        stats = store.get_stats()
        print("  (Calibrated from %d doctor reviews)" % stats["total_sessions_reviewed"])

    # Initialize LLM
    llm = LLMExtractor()
    llm_available = llm.is_available()

    # Initialize parser (uses LLM when available, falls back to fast)
    parser = EnhancedResponseParser(graph, llm_extractor=llm)
    spreader = ActivationSpreader(graph)

    print_header(llm_available)

    # ───────────────────────────────────────
    # DEMOGRAPHICS
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
            parts.append("age %d" % patient_age)
        if patient_gender:
            parts.append(patient_gender)
        print("  ✓ Demographics: %s" % ", ".join(parts))

    # ───────────────────────────────────────
    # ITERATION 0: Chief complaint (PRIMARY BOOST)
    # ───────────────────────────────────────
    print("\n  [Iteration 0] Describe your main health concern:")
    print("  (You can say things like 'I have been coughing for 2 weeks with phlegm')")
    try:
        initial_input = input("  ▸ ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Session ended.")
        return

    if not initial_input:
        print("  No symptom entered. Exiting.")
        return

    # Parse with LLM (is_initial=True)
    parsed = parser.parse(initial_input, is_initial=True)

    # Smart resolution: handles body-part combos, typos, fuzzy matching
    smart = SmartSymptomResolver(graph)
    smart_resolved = smart.resolve(initial_input)  # [(symptom, priority), ...]

    # Merge: LLM-detected + smart-resolved + simple alias
    all_detected = set(parsed.get("detected_symptoms", []))
    for sym, _ in smart_resolved:
        all_detected.add(sym)
    resolved = graph.resolve_symptom(initial_input)
    if resolved:
        all_detected.add(resolved)

    initial_symptoms = list(all_detected)

    # If still nothing, try splitting
    if not initial_symptoms:
        parts = [p.strip() for p in initial_input.replace(" and ", ",").split(",")]
        for part in parts:
            part_resolved = smart.resolve(part)
            for sym, _ in part_resolved:
                initial_symptoms.append(sym)
            r = graph.resolve_symptom(part)
            if r and r not in initial_symptoms:
                initial_symptoms.append(r)

    initial_symptoms = list(set(initial_symptoms))

    if not initial_symptoms:
        print("  ⚠ Could not recognize any symptoms from: '%s'" % initial_input)
        print("  Known symptoms: %s..." % ", ".join(sorted(graph.aliases.keys())[:20]))
        return

    # Multi-symptom prioritization: allocate questions proportionally
    prioritized = smart.prioritize_symptoms(initial_symptoms)

    print("\n  ✓ Recognized symptoms (PRIMARY BOOST %.1fx):" % PRIMARY_BOOST)
    for sym, pri, alloc in prioritized:
        display = graph.symptoms.get(sym, {}).get("display", sym)
        print("    • %s (priority=%d, ~%d questions)" % (display, pri, alloc))

    # Store allocation in metadata for question selector
    spreader.metadata["symptom_allocations"] = {sym: alloc for sym, _, alloc in prioritized}

    # Activate with PRIMARY BOOST
    for sym in initial_symptoms:
        spreader.activate(sym, present=True, strength=PRIMARY_BOOST)

    # Handle duration from initial parse
    dur = parsed.get("duration")
    if dur and dur.get("value") is not None:
        dur_str = "%s %s" % (dur["value"], dur["unit"] or "days")
        spreader.metadata["initial_duration"] = dur_str
        print("  ⏱ Duration: %s" % dur_str)

    # Handle body parts from initial parse
    for bp in parsed.get("body_parts", []):
        spreader.metadata.setdefault("body_parts", [])
        if bp not in spreader.metadata["body_parts"]:
            spreader.metadata["body_parts"].append(bp)
            print("  🔍 Body part: %s" % bp)

    # Handle medications from initial parse
    for med in parsed.get("medications_found", []):
        spreader.activate(med, present=True, strength=PRIMARY_BOOST)
        print("  💊 Medication: %s" % med)

    # Handle history from initial parse
    for hist in parsed.get("history_found", []):
        spreader.activate(hist, present=True, strength=PRIMARY_BOOST)
        print("  📋 History: %s" % hist)

    # Handle negated symptoms
    for sym in parsed.get("negated_symptoms", []):
        spreader.activate(sym, present=False)
        print("  ✗ Negated: %s" % sym)

    if parsed.get("parse_method") == "llm":
        latency = parsed.get("metadata", {}).get("_llm_latency_ms", 0)
        print("  [parsed via LLM, %dms]" % latency)

    print_distribution(spreader, graph)

    # ───────────────────────────────────────
    # ITERATIONS 1–9: Follow-up questions
    # Uses QuestionDedupGate for 3-layer dedup
    # ───────────────────────────────────────
    extra_info = ExtraMedicalInfo()
    # Absorb what we already know from iteration 0
    extra_info.absorb_parsed_response(parsed)
    dedup_gate = QuestionDedupGate(extra_info=extra_info)
    questions_log = []

    for iteration in range(1, MAX_QUESTIONS + 1):
        # Check stopping criteria
        top = spreader.get_top_conditions(1)
        if top and top[0][1] > CONFIDENCE_THRESHOLD:
            H = spreader.get_entropy()
            if H < ENTROPY_THRESHOLD:
                print(f"\n  ✓ Confidence reached at iteration {iteration}. Stopping.")
                break

        # Select next question (pass gate's tracked IDs and elicits)
        question, ig = selector.select_next(
            spreader,
            dedup_gate.asked_question_ids,
            asked_elicits=dedup_gate.asked_elicits,
        )

        if question is None or ig < MIN_INFO_GAIN:
            print(f"\n  ✓ No more useful questions. Stopping at iteration {iteration}.")
            break

        # ── 3-LAYER DEDUP CHECK ──
        should_ask, reason, auto_fill = dedup_gate.should_ask(question, spreader)

        if not should_ask:
            if auto_fill:
                # Auto-fill: activate the finding without asking
                spreader.activate(auto_fill["symptom"], present=auto_fill["present"])
                print(f"\n  [Iteration {iteration}] ⚡ Auto-filled '{auto_fill['symptom']}' ({auto_fill['reason']})")
            else:
                # Silently skip (category/text/elicits duplicate)
                pass
            continue

        # Register in gate BEFORE asking (prevents any race condition)
        dedup_gate.register_question(question)
        elicited = question["elicits"]

        # Ask the question
        print(f"\n  [Iteration {iteration}] ({question['id']}, IG={ig:.4f})")
        print(f"  Q: {question['text']}")

        try:
            response = input("  ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Session ended early.")
            break

        # Parse response (LLM or fast path — automatic)
        # Pass known info to LLM so it doesn't re-extract what we already know
        parsed = parser.parse(response, current_question=question)

        # Show parse details (debugging + LLM timing)
        method = parsed.get("parse_method", "fast")
        latency = parsed.get("metadata", {}).get("_llm_latency_ms", 0)
        method_str = "[%s%s]" % (method, ", %dms" % latency if latency else "")
        print("    %s yes=%s no=%s syms=%s neg=%s" % (
            method_str,
            parsed["is_yes"], parsed["is_no"],
            parsed.get("detected_symptoms", []),
            parsed.get("negated_symptoms", []),
        ))

        # Absorb everything the parser found into extra_medical_info
        extra_info.absorb_parsed_response(parsed)

        # Log for feedback
        questions_log.append({
            "id": question["id"],
            "text": question["text"],
            "response": response,
            "parse_method": parsed.get("parse_method", "fast"),
        })

        # Apply parsed response to spreader
        logs = apply_parsed_response(parsed, spreader, graph, elicited)
        for log in logs:
            print("    %s" % log)

        # Show distribution
        print_distribution(spreader, graph)

        # Show cumulative symptom state after each iteration
        print("    ✓ Active : %s" % ", ".join(sorted(spreader.active_symptoms)))
        print("    ✗ Denied : %s" % ", ".join(sorted(spreader.negative_symptoms)))

    # ───────────────────────────────────────
    # FINAL OUTPUT
    # ───────────────────────────────────────
    print_final_result(spreader, graph)

    # ───────────────────────────────────────
    # CLINICIAN FEEDBACK
    # ───────────────────────────────────────
    print("\n  ─── CLINICIAN REVIEW ───")
    print("  Would a doctor like to review this result? (y/n)")
    try:
        review = input("  ▸ ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        review = "n"

    if review in ("y", "yes"):
        session_log = SessionLog()
        session_log.capture(spreader, graph, questions_log,
                           patient_info={"age": patient_age, "gender": patient_gender})
        session_log.initial_symptoms = list(initial_symptoms)

        collector = FeedbackCollector()
        feedback = collector.collect(session_log, graph)

        if feedback:
            updater = WeightUpdater(learning_rate=0.02)
            result = updater.apply_feedback(feedback, session_log, graph)

            log_path = os.path.join(script_dir, "feedback_log.json")
            store = FeedbackStore(filepath=log_path)
            store.save_feedback(feedback, session_log, result)

            # Persist updated weights
            WeightSnapshot.save(graph, snapshot_path)

            print("\n  ─── WEIGHT UPDATE SUMMARY ───")
            print("  Feedback type: %s" % feedback["type"])
            print("  Edges adjusted: %d" % result["num_changes"])
            for ch in result.get("changes", [])[:10]:
                print("    %s: %s → %s: %.3f → %.3f (%s)" % (
                    ch["action"], ch["symptom"], ch["condition"],
                    ch["old_weight"], ch["new_weight"],
                    "▲" if ch["delta"] > 0 else "▼"
                ))

            stats = store.get_stats()
            print("\n  ─── CUMULATIVE STATS ───")
            print("  Sessions reviewed: %d" % stats["total_sessions_reviewed"])
            print("  Confirms: %d | Corrections: %d | Reranks: %d" % (
                stats["confirms"], stats["corrections"], stats["reranks"]))
        else:
            print("  Skipped.")


if __name__ == "__main__":
    run_triage()
