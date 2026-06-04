"""
Clinician Feedback System (RLHF-style Weight Calibration)
==========================================================

After each triage session, a clinician reviews the engine's output and provides
feedback. The system uses this to adjust knowledge graph edge weights over time.

NO PROGRAMMING REQUIRED by the doctor. They just:
  1. See the triage result (top conditions + evidence)
  2. Pick ONE action:
      - CONFIRM  → "Yes, #1 was correct"
      - CORRECT  → "No, the real diagnosis was ___" (pick from list)
      - RERANK   → "Your top 3 were right but in wrong order"
      - SKIP     → "I'm not sure / can't evaluate this case"

The system logs every feedback event and adjusts edge weights using
exponential moving averages — slow enough that one doctor's opinion doesn't
dominate, fast enough that consistent corrections shift weights meaningfully.

Architecture:
  ┌─────────────┐     ┌──────────────┐     ┌────────────────┐
  │ Triage       │ ──▸ │ Session Log  │ ──▸ │ Feedback UI    │
  │ Engine       │     │ (symptoms,   │     │ (doctor picks  │
  │ (runs)       │     │  conditions, │     │  confirm/      │
  │              │     │  scores)     │     │  correct)      │
  └─────────────┘     └──────────────┘     └───────┬────────┘
                                                    │
                                                    ▼
                                           ┌────────────────┐
                                           │ Weight Updater │
                                           │ (adjusts edges │
                                           │  in graph)     │
                                           └───────┬────────┘
                                                    │
                                                    ▼
                                           ┌────────────────┐
                                           │ Feedback Store │
                                           │ (JSON log of   │
                                           │  all feedback)  │
                                           └────────────────┘
"""

import json
import os
import time
from typing import Optional
from collections import defaultdict


# ─────────────────────────────────────────────
# 1. SESSION LOG — captures triage session state
# ─────────────────────────────────────────────

class SessionLog:
    """
    Captures everything about a triage session so the doctor can review it
    and the weight updater can trace which edges to adjust.
    """

    def __init__(self):
        self.session_id = ""
        self.timestamp = ""
        self.patient_demographics = {}       # {"age": 35, "gender": "male"}
        self.initial_symptoms = []           # ["fever", "cough"]
        self.confirmed_symptoms = []         # everything activated as present
        self.denied_symptoms = []            # everything activated as absent
        self.questions_asked = []            # [{"id": "Q001", "text": "...", "response": "yes"}]
        self.top_conditions = []             # [("dengue", 0.38), ("malaria", 0.15), ...]
        self.metadata = {}                   # durations, etc.

    def capture(self, spreader, graph, questions_log, patient_info=None):
        """Capture current state from a completed triage session."""
        self.session_id = "S%d" % int(time.time() * 1000)
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.patient_demographics = patient_info or {}
        self.confirmed_symptoms = sorted(list(spreader.active_symptoms))
        self.denied_symptoms = sorted(list(spreader.negative_symptoms))
        self.questions_asked = list(questions_log)
        self.top_conditions = spreader.get_top_conditions(5)
        self.metadata = dict(spreader.metadata)

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "patient": self.patient_demographics,
            "initial_symptoms": self.initial_symptoms,
            "confirmed_symptoms": self.confirmed_symptoms,
            "denied_symptoms": self.denied_symptoms,
            "questions_asked": self.questions_asked,
            "top_conditions": [(c, round(p, 4)) for c, p in self.top_conditions],
            "metadata": self.metadata,
        }

    def display_for_doctor(self, graph):
        """
        Print a clean summary for the clinician to review.
        No technical jargon — just the facts they need.
        """
        print("\n" + "=" * 60)
        print("  TRIAGE SUMMARY FOR REVIEW")
        print("=" * 60)

        # Patient info
        demo = self.patient_demographics
        if demo:
            parts = []
            if demo.get("age"):
                parts.append("Age: %s" % demo["age"])
            if demo.get("gender"):
                parts.append("Gender: %s" % demo["gender"])
            print("  Patient: %s" % ", ".join(parts))

        # Symptoms
        print("\n  Symptoms reported:")
        for sym in self.confirmed_symptoms:
            display = sym
            if sym in graph.symptoms:
                display = graph.symptoms[sym].get("display", sym)
            print("    ✓ %s" % display)

        if self.denied_symptoms:
            print("  Symptoms denied:")
            for sym in self.denied_symptoms[:5]:  # show max 5
                print("    ✗ %s" % sym)

        # Engine's assessment
        print("\n  ENGINE'S ASSESSMENT:")
        for i, (cond, prob) in enumerate(self.top_conditions):
            info = graph.conditions.get(cond, {})
            display = info.get("display", cond)
            severity = info.get("severity", "?")
            route = info.get("route_to", "?").replace("_", " ").title()
            rank = "PRIMARY" if i == 0 else "#%d" % (i + 1)
            print("    %s: %s (%.1f%%) → %s [%s]" % (rank, display, prob * 100, route, severity))

        print("=" * 60)


# ─────────────────────────────────────────────
# 2. FEEDBACK COLLECTOR — doctor's input
# ─────────────────────────────────────────────

class FeedbackCollector:
    """
    Simple CLI interface for the doctor. No programming needed.
    Just numbered choices.
    """

    def collect(self, session_log, graph):
        # type: (SessionLog, object) -> Optional[dict]
        """
        Show the triage result and ask the doctor for feedback.
        Returns a feedback dict or None if skipped.
        """
        session_log.display_for_doctor(graph)

        print("\n  YOUR FEEDBACK:")
        print("    1. CONFIRM  — The primary diagnosis is correct")
        print("    2. CORRECT  — The real diagnosis is different")
        print("    3. RERANK   — Right conditions, wrong order")
        print("    4. SKIP     — Cannot evaluate this case")

        try:
            choice = input("\n  Enter choice (1-4): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if choice == "1":
            return self._confirm(session_log)
        elif choice == "2":
            return self._correct(session_log, graph)
        elif choice == "3":
            return self._rerank(session_log, graph)
        elif choice == "4":
            return None
        else:
            print("  Invalid choice. Skipping.")
            return None

    def _confirm(self, session_log):
        top_cond = session_log.top_conditions[0][0] if session_log.top_conditions else None
        confidence = input("  How confident are you? (high/medium/low): ").strip().lower()
        conf_val = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(confidence, 0.6)
        return {
            "type": "confirm",
            "correct_condition": top_cond,
            "confidence": conf_val,
            "session_id": session_log.session_id,
        }

    def _correct(self, session_log, graph):
        print("\n  What is the correct diagnosis?")

        # Show numbered list of ALL conditions (grouped by category)
        all_conds = sorted(graph.conditions.items(), key=lambda x: x[1].get("route_to", ""))
        current_route = ""
        cond_list = []
        for cond_key, info in all_conds:
            route = info.get("route_to", "other").replace("_", " ").title()
            if route != current_route:
                current_route = route
                print("    ── %s ──" % route)
            cond_list.append(cond_key)
            idx = len(cond_list)
            display = info.get("display", cond_key)
            # Mark if it was in the engine's top 5
            marker = ""
            for rank, (tc, _) in enumerate(session_log.top_conditions):
                if tc == cond_key:
                    marker = " ← engine said #%d" % (rank + 1)
                    break
            print("    %3d. %s%s" % (idx, display, marker))

        try:
            idx_input = input("\n  Enter number of correct diagnosis: ").strip()
            idx = int(idx_input) - 1
            if 0 <= idx < len(cond_list):
                correct = cond_list[idx]
                confidence = input("  How confident? (high/medium/low): ").strip().lower()
                conf_val = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(confidence, 0.6)
                return {
                    "type": "correct",
                    "correct_condition": correct,
                    "engine_top_condition": session_log.top_conditions[0][0] if session_log.top_conditions else None,
                    "confidence": conf_val,
                    "session_id": session_log.session_id,
                }
        except (ValueError, IndexError):
            pass

        print("  Invalid input. Skipping.")
        return None

    def _rerank(self, session_log, graph):
        print("\n  Current ranking:")
        top_keys = []
        for i, (cond, prob) in enumerate(session_log.top_conditions[:3]):
            display = graph.conditions.get(cond, {}).get("display", cond)
            top_keys.append(cond)
            print("    %d. %s (%.1f%%)" % (i + 1, display, prob * 100))

        try:
            correct_rank = input("\n  Which should be #1? Enter number (1/2/3): ").strip()
            idx = int(correct_rank) - 1
            if 0 <= idx < len(top_keys):
                return {
                    "type": "rerank",
                    "correct_condition": top_keys[idx],
                    "original_rank": idx + 1,
                    "session_id": session_log.session_id,
                    "confidence": 0.6,
                }
        except (ValueError, IndexError):
            pass

        print("  Invalid input. Skipping.")
        return None


# ─────────────────────────────────────────────
# 3. WEIGHT UPDATER — adjusts graph edges
# ─────────────────────────────────────────────

class WeightUpdater:
    """
    Adjusts edge weights in the knowledge graph based on doctor feedback.

    The update rule is simple:
      - For each CONFIRMED symptom in the session:
          - Edge to CORRECT condition:   weight += learning_rate * confidence
          - Edge to WRONG top condition:  weight -= learning_rate * confidence * 0.5
      - Weights are clamped to [0.05, 0.99] to prevent collapse

    Learning rate is intentionally small (0.02 default) so:
      - 1 doctor's feedback nudges weights ~2%
      - 10 consistent corrections shift weights ~20%
      - It takes broad consensus to fundamentally change the graph

    This is conceptually similar to RLHF reward model updates, but much simpler:
      - RLHF: update neural network weights via policy gradient
      - Ours: update graph edge weights via direct adjustment
    """

    def __init__(self, learning_rate=0.02, min_weight=0.05, max_weight=0.99):
        self.lr = learning_rate
        self.min_w = min_weight
        self.max_w = max_weight

    def apply_feedback(self, feedback, session_log, graph):
        # type: (dict, SessionLog, object) -> dict
        """
        Apply one feedback event to the graph's edges.
        Returns a summary of what changed.
        """
        if feedback is None:
            return {"changes": 0}

        fb_type = feedback["type"]
        correct_cond = feedback["correct_condition"]
        confidence = feedback.get("confidence", 0.6)
        active_symptoms = session_log.confirmed_symptoms

        changes = []

        if fb_type == "confirm":
            changes = self._reinforce(graph, active_symptoms, correct_cond, confidence)

        elif fb_type == "correct":
            wrong_cond = feedback.get("engine_top_condition")
            changes = self._correct(graph, active_symptoms, correct_cond, wrong_cond, confidence)

        elif fb_type == "rerank":
            original_rank = feedback.get("original_rank", 2)
            # Mild reinforcement of the correct condition
            changes = self._reinforce(graph, active_symptoms, correct_cond, confidence * 0.5)

        return {
            "type": fb_type,
            "correct_condition": correct_cond,
            "num_changes": len(changes),
            "changes": changes,
        }

    def _reinforce(self, graph, active_symptoms, correct_cond, confidence):
        """Increase weights from active symptoms to the correct condition."""
        changes = []
        for sym in active_symptoms:
            edges = graph.edges.get(sym, [])
            for i, (cond, weight) in enumerate(edges):
                if cond == correct_cond:
                    delta = self.lr * confidence
                    new_weight = min(self.max_w, weight + delta)
                    if new_weight != weight:
                        edges[i] = (cond, round(new_weight, 4))
                        changes.append({
                            "symptom": sym,
                            "condition": cond,
                            "old_weight": round(weight, 4),
                            "new_weight": round(new_weight, 4),
                            "delta": round(delta, 4),
                            "action": "reinforce",
                        })
        return changes

    def _correct(self, graph, active_symptoms, correct_cond, wrong_cond, confidence):
        """
        Decrease weights to wrong condition, increase weights to correct condition.
        If no edge exists from a symptom to the correct condition, CREATE one with
        a small initial weight — this is how the graph learns new associations.
        """
        changes = []

        for sym in active_symptoms:
            edges = graph.edges.get(sym, [])

            # --- Weaken edges to wrong condition ---
            if wrong_cond:
                for i, (cond, weight) in enumerate(edges):
                    if cond == wrong_cond:
                        delta = self.lr * confidence * 0.5  # weaker than reinforcement
                        new_weight = max(self.min_w, weight - delta)
                        if new_weight != weight:
                            edges[i] = (cond, round(new_weight, 4))
                            changes.append({
                                "symptom": sym,
                                "condition": wrong_cond,
                                "old_weight": round(weight, 4),
                                "new_weight": round(new_weight, 4),
                                "delta": round(-delta, 4),
                                "action": "weaken",
                            })

            # --- Strengthen edges to correct condition ---
            found = False
            for i, (cond, weight) in enumerate(edges):
                if cond == correct_cond:
                    found = True
                    delta = self.lr * confidence
                    new_weight = min(self.max_w, weight + delta)
                    if new_weight != weight:
                        edges[i] = (cond, round(new_weight, 4))
                        changes.append({
                            "symptom": sym,
                            "condition": correct_cond,
                            "old_weight": round(weight, 4),
                            "new_weight": round(new_weight, 4),
                            "delta": round(delta, 4),
                            "action": "strengthen",
                        })

            # --- Create new edge if none exists ---
            if not found:
                initial_weight = 0.15  # small initial weight for new associations
                edges.append((correct_cond, initial_weight))
                changes.append({
                    "symptom": sym,
                    "condition": correct_cond,
                    "old_weight": 0.0,
                    "new_weight": initial_weight,
                    "delta": initial_weight,
                    "action": "create_new_edge",
                })

        return changes


# ─────────────────────────────────────────────
# 4. FEEDBACK STORE — persistent log
# ─────────────────────────────────────────────

class FeedbackStore:
    """
    Stores all feedback events and weight change history as JSON.
    This creates an audit trail: you can always see WHY a weight changed,
    who changed it, and when.
    """

    def __init__(self, filepath="feedback_log.json"):
        self.filepath = filepath
        self.records = []
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    self.records = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.records = []

    def save_feedback(self, feedback, session_log, update_result):
        """Save one feedback event with full context."""
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_log.session_id,
            "session": session_log.to_dict(),
            "feedback": feedback,
            "weight_changes": update_result,
        }
        self.records.append(record)
        self._persist()

    def _persist(self):
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.records, f, indent=2)
        except IOError as e:
            print("  Warning: could not save feedback log: %s" % e)

    def get_stats(self):
        """Summary statistics for the feedback history."""
        total = len(self.records)
        confirms = sum(1 for r in self.records if r.get("feedback", {}).get("type") == "confirm")
        corrects = sum(1 for r in self.records if r.get("feedback", {}).get("type") == "correct")
        reranks = sum(1 for r in self.records if r.get("feedback", {}).get("type") == "rerank")
        total_edge_changes = sum(
            r.get("weight_changes", {}).get("num_changes", 0) for r in self.records
        )
        return {
            "total_sessions_reviewed": total,
            "confirms": confirms,
            "corrections": corrects,
            "reranks": reranks,
            "total_edge_changes": total_edge_changes,
        }

    def get_edge_history(self, symptom, condition):
        """Get all weight changes for a specific edge over time."""
        history = []
        for r in self.records:
            for change in r.get("weight_changes", {}).get("changes", []):
                if change.get("symptom") == symptom and change.get("condition") == condition:
                    history.append({
                        "timestamp": r["timestamp"],
                        "action": change["action"],
                        "old": change["old_weight"],
                        "new": change["new_weight"],
                        "delta": change["delta"],
                    })
        return history


# ─────────────────────────────────────────────
# 5. WEIGHT SNAPSHOT — save/restore calibrated weights
# ─────────────────────────────────────────────

class WeightSnapshot:
    """
    Save and restore the current edge weights to/from a JSON file.
    This lets you:
      - Save after a batch of doctor reviews ("calibrated_weights_v1.json")
      - Restore to a known-good state if something goes wrong
      - Compare weights before and after N feedback sessions
    """

    @staticmethod
    def save(graph, filepath="weight_snapshot.json"):
        """Export all edges with current weights."""
        edges = {}
        for sym, edge_list in graph.edges.items():
            edges[sym] = [(cond, round(w, 4)) for cond, w in edge_list]
        with open(filepath, "w") as f:
            json.dump(edges, f, indent=2)
        print("  Saved %d symptom groups to %s" % (len(edges), filepath))

    @staticmethod
    def load(graph, filepath="weight_snapshot.json"):
        """Import edges from a snapshot, overwriting current weights."""
        if not os.path.exists(filepath):
            print("  Snapshot file not found: %s" % filepath)
            return False
        with open(filepath, "r") as f:
            edges = json.load(f)
        for sym, edge_list in edges.items():
            graph.edges[sym] = [(cond, w) for cond, w in edge_list]
        print("  Loaded %d symptom groups from %s" % (len(edges), filepath))
        return True

    @staticmethod
    def diff(graph, filepath="weight_snapshot.json"):
        """Show which weights changed vs a snapshot."""
        if not os.path.exists(filepath):
            print("  No snapshot to compare against.")
            return
        with open(filepath, "r") as f:
            old_edges = json.load(f)

        changes = []
        for sym, edge_list in graph.edges.items():
            old_list = {c: w for c, w in old_edges.get(sym, [])}
            for cond, new_w in edge_list:
                old_w = old_list.get(cond)
                if old_w is not None and abs(new_w - old_w) > 0.001:
                    changes.append((sym, cond, old_w, new_w))
                elif old_w is None:
                    changes.append((sym, cond, 0.0, new_w))

        if not changes:
            print("  No weight changes detected.")
        else:
            print("  %d edges changed:" % len(changes))
            for sym, cond, old, new in sorted(changes, key=lambda x: abs(x[3]-x[2]), reverse=True)[:20]:
                direction = "▲" if new > old else "▼"
                print("    %s %s → %s: %.3f → %.3f (%s%.3f)" % (
                    direction, sym, cond, old, new, "+" if new > old else "", new - old))
