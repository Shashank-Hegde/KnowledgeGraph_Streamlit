"""
Knowledge Graph Engine
======================
Builds the graph from knowledge_base data.
Handles activation spreading and condition scoring.
"""

import math
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from knowledge_base import (
    SYMPTOMS, CONDITIONS,
    SYMPTOM_CONDITION_EDGES, EXTRA_FINDING_EDGES,
    QUESTIONS, SYMPTOM_ALIASES,
)


class KnowledgeGraph:
    """
    Weighted directed graph: symptom/finding nodes → condition nodes.
    Stores edges, conditions, questions, and aliases.
    """

    def __init__(self):
        # edges[symptom_key] = [(condition_key, weight), ...]
        self.edges = defaultdict(list)
        self.conditions = {}      # condition_key → metadata
        self.symptoms = {}        # symptom_key → metadata
        self.questions = {}       # question_id → question dict
        self.aliases = {}         # user input text → canonical symptom key
        self._build()

    def _build(self):
        """Load all data from knowledge_base module."""
        self.symptoms = dict(SYMPTOMS)
        self.conditions = dict(CONDITIONS)

        # Primary symptom → condition edges
        for sym, cond, weight in SYMPTOM_CONDITION_EDGES:
            self.edges[sym].append((cond, weight))

        # Extra finding → condition edges (from follow-up question responses)
        for finding, cond, weight in EXTRA_FINDING_EDGES:
            self.edges[finding].append((cond, weight))

        # Questions indexed by ID
        for q in QUESTIONS:
            self.questions[q["id"]] = q

        # Aliases
        self.aliases = dict(SYMPTOM_ALIASES)

    def resolve_symptom(self, user_input: str) -> Optional[str]:
        """Map user input text to a canonical symptom key, or None."""
        text = user_input.strip().lower()
        if text in self.aliases:
            return self.aliases[text]
        if text in self.symptoms:
            return text
        # Fuzzy: check if input is a substring of any alias
        for alias, canonical in self.aliases.items():
            if text in alias or alias in text:
                return canonical
        return None

    def get_connected_conditions(self, symptom_key: str) -> list[tuple[str, float]]:
        """Return [(condition_key, weight), ...] for a given symptom/finding."""
        return self.edges.get(symptom_key, [])

    def get_relevant_questions(self, condition_keys: list[str]) -> list[dict]:
        """Return questions tagged for any of the given conditions, sorted by relevance."""
        scored = []
        cond_set = set(condition_keys)
        for qid, q in self.questions.items():
            overlap = len(cond_set.intersection(set(q["tags"])))
            if overlap > 0:
                scored.append((overlap, q))
        scored.sort(key=lambda x: -x[0])
        return [q for _, q in scored]


class ActivationSpreader:
    """
    Accumulates evidence from symptoms/findings and produces
    a probability distribution over conditions.
    """

    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.activations = defaultdict(float)  # condition_key → cumulative score
        self.active_symptoms = set()           # track what's been activated
        self.negative_symptoms = set()         # track what's been denied
        self.metadata = {}                     # e.g. {"fever_duration": "3 days"}

    def activate(self, symptom_key: str, present: bool = True, strength: float = 1.0):
        """
        Spread activation from a symptom/finding to connected conditions.
        present=False applies negative evidence (weakens connected conditions).
        """
        if present:
            self.active_symptoms.add(symptom_key)
            multiplier = strength
        else:
            self.negative_symptoms.add(symptom_key)
            multiplier = -0.3 * strength  # denial is weaker than confirmation

        for cond, weight in self.graph.get_connected_conditions(symptom_key):
            self.activations[cond] += weight * multiplier

    def activate_multiple(self, symptom_keys: list[str]):
        """Activate a list of symptom keys (all present=True)."""
        for key in symptom_keys:
            if key not in self.active_symptoms:
                self.activate(key, present=True)

    def get_distribution(self) -> dict[str, float]:
        """
        Convert raw activations to a probability distribution using softmax.
        Returns {condition_key: probability}.
        """
        all_conditions = list(self.graph.conditions.keys())
        if not self.activations:
            n = len(all_conditions)
            return {c: 1.0 / n for c in all_conditions}

        # Softmax normalization
        raw = {c: self.activations.get(c, 0.0) for c in all_conditions}
        max_val = max(raw.values()) if raw else 0
        exp_vals = {c: math.exp(v - max_val) for c, v in raw.items()}
        total = sum(exp_vals.values())
        return {c: v / total for c, v in exp_vals.items()}

    def get_top_conditions(self, n: int = 5) -> list[tuple[str, float]]:
        """Return top N conditions by probability."""
        dist = self.get_distribution()
        sorted_conds = sorted(dist.items(), key=lambda x: -x[1])
        return sorted_conds[:n]

    def get_entropy(self) -> float:
        """Shannon entropy of the current distribution (in bits)."""
        dist = self.get_distribution()
        return -sum(p * math.log2(p + 1e-12) for p in dist.values())

    def clone(self) -> "ActivationSpreader":
        """Create a copy for simulation (what-if analysis)."""
        new = ActivationSpreader(self.graph)
        new.activations = defaultdict(float, self.activations)
        new.active_symptoms = set(self.active_symptoms)
        new.negative_symptoms = set(self.negative_symptoms)
        new.metadata = dict(self.metadata)
        return new
