"""
Question Selector  (v4 — v2 schema aware)
==========================================
Now handles:
- Separate condition_tags vs symptom_tags
- Demographic filtering (age, gender)
- requires_symptoms / excludes_symptoms gating
- Relevance = condition-probability overlap + symptom-activation overlap
"""

import math
from typing import Optional, Dict, Set, Tuple
from graph_engine import KnowledgeGraph, ActivationSpreader


def entropy(dist):
    # type: (Dict[str, float]) -> float
    return -sum(p * math.log2(p + 1e-12) for p in dist.values())


class QuestionSelector:

    def __init__(self, graph, patient_age=None, patient_gender=None):
        # type: (KnowledgeGraph, Optional[int], Optional[str]) -> None
        self.graph = graph
        self.patient_age = patient_age        # e.g. 28
        self.patient_gender = patient_gender  # "male" or "female" or None

        # Pre-build: for each elicited symptom, which conditions connect?
        self._elicited_conds = {}  # type: Dict[str, Set[str]]
        for qid, q in graph.questions.items():
            sym = q["elicits"]
            if sym not in self._elicited_conds:
                self._elicited_conds[sym] = set()
                for cond, _ in graph.get_connected_conditions(sym):
                    self._elicited_conds[sym].add(cond)

    def _is_eligible(self, q, spreader):
        # type: (dict, ActivationSpreader) -> bool
        """
        Check if this question should even be considered, based on:
        1. Demographic filters (age, gender)
        2. requires_symptoms — all must be active
        3. excludes_symptoms — none must be active
        4. Already asked / already have evidence for elicited symptom
        """
        # --- Gender filter ---
        q_gender = q.get("gender")
        if q_gender and self.patient_gender:
            if q_gender != self.patient_gender:
                return False

        # --- Age filter ---
        if self.patient_age is not None:
            age_min = q.get("age_min")
            age_max = q.get("age_max")
            if age_min is not None and self.patient_age < age_min:
                return False
            if age_max is not None and self.patient_age > age_max:
                return False

        # --- requires_symptoms: ALL must be currently active ---
        for req in q.get("requires_symptoms", []):
            if req not in spreader.active_symptoms:
                return False

        # --- excludes_symptoms: NONE must be active ---
        for exc in q.get("excludes_symptoms", []):
            if exc in spreader.active_symptoms or exc in spreader.negative_symptoms:
                return False

        # --- Already have evidence for elicited symptom ---
        elicited = q["elicits"]
        if elicited in spreader.active_symptoms or elicited in spreader.negative_symptoms:
            return False

        return True

    def select_next(
        self,
        spreader,          # type: ActivationSpreader
        asked_question_ids,  # type: Set[str]
        top_n_candidates=50, # type: int
        asked_elicits=None,  # type: Optional[Set[str]]
    ):
        # type: (...) -> Tuple[Optional[dict], float]
        """
        Select next question. Now also tracks asked_elicits to prevent
        duplicate questions that probe the same symptom/finding.
        """
        if asked_elicits is None:
            asked_elicits = set()

        current_dist = spreader.get_distribution()
        current_H = entropy(current_dist)

        # Focus set: top 10 conditions
        top_conds = spreader.get_top_conditions(10)
        top_cond_probs = {c: p for c, p in top_conds}
        top_cond_set = set(top_cond_probs)

        # Get candidate questions tagged for top conditions
        # Support BOTH v1 ("tags") and v2 ("condition_tags") format
        candidates = self._get_candidates(top_cond_set, spreader)

        best_question = None
        best_score = -1.0
        best_ig = 0.0

        # Pre-populate asked texts from already-asked questions
        asked_texts = set()
        for qid in asked_question_ids:
            if qid in self.graph.questions:
                asked_texts.add(self.graph.questions[qid]["text"].strip().lower())

        evaluated = 0
        asked_texts = set()  # deduplicate by question text too
        for q in candidates:
            if q["id"] in asked_question_ids:
                continue

            # Skip if this symptom/finding was already probed by another question
            if q["elicits"] in asked_elicits:
                continue

            # Skip if identical question text was already asked (catches Q013/Q108 type dupes)
            q_text_norm = q["text"].strip().lower()
            if q_text_norm in asked_texts:
                continue

            if not self._is_eligible(q, spreader):
                continue

            if evaluated >= top_n_candidates:
                break
            evaluated += 1

            elicited = q["elicits"]

            # --- Condition relevance ---
            # Use "condition_tags" if present (v2), else "tags" (v1)
            c_tags = set(q.get("condition_tags", q.get("tags", [])))
            elicited_conds = self._elicited_conds.get(elicited, set())

            triple_overlap = top_cond_set & c_tags & elicited_conds
            double_overlap = top_cond_set & c_tags

            cond_relevance = 0.0
            for c in triple_overlap:
                cond_relevance += top_cond_probs[c] * 3.0
            for c in (double_overlap - triple_overlap):
                cond_relevance += top_cond_probs[c] * 1.0

            # --- Symptom relevance (v2 only) ---
            # If the question's symptom_tags overlap with currently active symptoms,
            # it means the question is contextually relevant right now
            s_tags = set(q.get("symptom_tags", []))
            sym_relevance = 0.0
            for st in s_tags:
                if st in spreader.active_symptoms:
                    sym_relevance += 0.5  # bonus per matching active symptom

            total_relevance = cond_relevance + sym_relevance

            if total_relevance < 0.001 and not triple_overlap and not double_overlap and not s_tags:
                continue

            # --- Simulate YES / NO ---
            sim_yes = spreader.clone()
            sim_yes.activate(elicited, present=True)
            rmap = q.get("response_map", {})
            if "yes" in rmap and isinstance(rmap["yes"], list):
                for extra in rmap["yes"]:
                    if extra not in sim_yes.active_symptoms:
                        sim_yes.activate(extra, present=True)
            dist_yes = sim_yes.get_distribution()

            sim_no = spreader.clone()
            sim_no.activate(elicited, present=False)
            dist_no = sim_no.get_distribution()

            # P(yes) estimate
            p_yes = 0.0
            for cond, weight in self.graph.get_connected_conditions(elicited):
                p_yes += current_dist.get(cond, 0) * weight
            p_yes = min(max(p_yes, 0.1), 0.9)

            # Information Gain
            ig = current_H - (p_yes * entropy(dist_yes) + (1 - p_yes) * entropy(dist_no))

            # Severity tiebreaker
            sev_bonus = 0.0
            for tag in c_tags:
                if tag in self.graph.conditions:
                    sev = self.graph.conditions[tag].get("severity", "low")
                    if sev == "critical":
                        sev_bonus = max(sev_bonus, 0.002)
                    elif sev == "high":
                        sev_bonus = max(sev_bonus, 0.001)

            # Final score
            score = total_relevance * 50.0 + ig + sev_bonus

            if score > best_score:
                best_score = score
                best_ig = ig
                best_question = q

        if best_question:
            return best_question, best_score
        return None, 0.0

    def _get_candidates(self, top_cond_set, spreader):
        # type: (Set[str], ActivationSpreader) -> list
        """
        Get candidate questions that are relevant to the current state.
        Supports both v1 (tags) and v2 (condition_tags + symptom_tags).
        """
        scored = []
        for qid, q in self.graph.questions.items():
            # v2 condition_tags, falling back to v1 tags
            c_tags = set(q.get("condition_tags", q.get("tags", [])))
            s_tags = set(q.get("symptom_tags", []))

            # Score by condition tag overlap
            c_overlap = len(top_cond_set.intersection(c_tags))

            # Score by symptom tag overlap with active symptoms
            s_overlap = len(s_tags.intersection(spreader.active_symptoms))

            total = c_overlap * 2 + s_overlap
            if total > 0:
                scored.append((total, q))

        scored.sort(key=lambda x: -x[0])
        return [q for _, q in scored]
