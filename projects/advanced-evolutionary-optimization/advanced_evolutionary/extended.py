"""Interactive, coevolutionary, classifier-system, and genetic-programming examples."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np


def interactive_evolution(
    scorer: Callable[[np.ndarray], float],
    *,
    dimensions: int = 6,
    population_size: int = 12,
    generations: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Evolve vectors using an external scorer, including human-in-the-loop callbacks."""
    if dimensions <= 0 or population_size < 4 or generations <= 0:
        raise ValueError("invalid interactive-evolution configuration")
    rng = np.random.default_rng(seed)
    population = rng.uniform(-1.0, 1.0, size=(population_size, dimensions))
    best = population[0].copy()
    best_score = float("-inf")
    for _ in range(generations):
        scores = np.array([float(scorer(candidate.copy())) for candidate in population])
        order = np.argsort(scores)[::-1]
        if scores[order[0]] > best_score:
            best_score = float(scores[order[0]])
            best = population[order[0]].copy()
        elites = population[order[: max(2, population_size // 3)]]
        parents = elites[rng.integers(0, len(elites), size=population_size - len(elites))]
        children = np.clip(
            parents + rng.normal(0.0, 0.12, size=parents.shape),
            -1.0,
            1.0,
        )
        population = np.vstack([elites, children])
    return best, best_score


def competitive_coevolution(
    payoff: Callable[[float, float], float],
    *,
    population_size: int = 30,
    generations: int = 40,
    opponents_per_candidate: int = 5,
    seed: int = 0,
) -> tuple[float, float]:
    """Coevolve two scalar populations in a sampled zero-sum game."""
    if population_size < 4 or generations <= 0 or opponents_per_candidate <= 0:
        raise ValueError("invalid coevolution configuration")
    rng = np.random.default_rng(seed)
    a_pop = rng.uniform(-1.0, 1.0, population_size)
    b_pop = rng.uniform(-1.0, 1.0, population_size)

    for _ in range(generations):
        a_fit = np.zeros(population_size)
        b_fit = np.zeros(population_size)
        k = min(opponents_per_candidate, population_size)
        for i, a in enumerate(a_pop):
            js = rng.choice(population_size, size=k, replace=False)
            a_fit[i] = np.mean([payoff(float(a), float(b_pop[j])) for j in js])
        for j, b in enumerate(b_pop):
            is_ = rng.choice(population_size, size=k, replace=False)
            b_fit[j] = -np.mean([payoff(float(a_pop[i]), float(b)) for i in is_])

        def breed(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
            elite_count = max(2, population_size // 4)
            elites = population[np.argsort(fitness)[-elite_count:]]
            children = elites[rng.integers(0, elite_count, size=population_size - elite_count)]
            children = np.clip(
                children + rng.normal(0.0, 0.08, len(children)),
                -1.0,
                1.0,
            )
            return np.concatenate([elites, children])

        a_pop = breed(a_pop, a_fit)
        b_pop = breed(b_pop, b_fit)

    return float(np.mean(a_pop)), float(np.mean(b_pop))


@dataclass
class Rule:
    condition: tuple[int | None, ...]
    action: int
    prediction: float = 0.5
    experience: int = 0

    def matches(self, state: Sequence[int]) -> bool:
        return all(c is None or c == int(x) for c, x in zip(self.condition, state, strict=True))


class LearningClassifierSystem:
    """Compact Michigan-style ternary-rule learning classifier system."""

    def __init__(self, n_features: int, *, max_rules: int = 200, seed: int = 0) -> None:
        if n_features <= 0 or max_rules < 2:
            raise ValueError("invalid LCS configuration")
        self.n_features = n_features
        self.max_rules = max_rules
        self.rng = np.random.default_rng(seed)
        self.rules: list[Rule] = []

    def _cover(self, state: Sequence[int], action: int) -> Rule:
        condition = tuple(None if self.rng.random() < 0.35 else int(x) for x in state)
        rule = Rule(condition=condition, action=int(action))
        self.rules.append(rule)
        if len(self.rules) > self.max_rules:
            self.rules.sort(key=lambda item: (item.experience, item.prediction))
            self.rules.pop(0)
        return rule

    def _matching(self, state: Sequence[int]) -> list[Rule]:
        return [rule for rule in self.rules if rule.matches(state)]

    def predict(self, state: Sequence[int]) -> int:
        matching = self._matching(state)
        if not matching:
            self._cover(state, 0)
            self._cover(state, 1)
            matching = self._matching(state)
        scores = []
        for action in (0, 1):
            action_rules = [rule for rule in matching if rule.action == action]
            scores.append(
                np.mean([rule.prediction for rule in action_rules]) if action_rules else 0.5
            )
        return int(np.argmax(scores))

    def fit(
        self,
        samples: Iterable[tuple[Sequence[int], int]],
        *,
        epochs: int = 20,
        learning_rate: float = 0.2,
    ) -> LearningClassifierSystem:
        data = [(tuple(map(int, state)), int(label)) for state, label in samples]
        if not data:
            raise ValueError("samples must not be empty")
        for state, label in data:
            if len(state) != self.n_features:
                raise ValueError("sample has wrong number of features")
            if label not in (0, 1):
                raise ValueError("LCS labels must be binary")
        for _ in range(epochs):
            self.rng.shuffle(data)
            for state, label in data:
                matching = self._matching(state)
                for action in (0, 1):
                    if not any(rule.action == action for rule in matching):
                        matching.append(self._cover(state, action))
                for rule in matching:
                    reward = 1.0 if rule.action == label else 0.0
                    rule.prediction += learning_rate * (reward - rule.prediction)
                    rule.experience += 1
        return self


OPS: tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], ...] = (
    np.add,
    np.subtract,
    np.multiply,
    lambda a, b: a / np.where(np.abs(b) < 1e-8, 1.0, b),
)


@dataclass
class CGPGenome:
    """Minimal single-output Cartesian Genetic Programming genome."""

    n_inputs: int
    genes: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.genes) // 3

    @classmethod
    def random(
        cls,
        n_inputs: int,
        n_nodes: int,
        rng: np.random.Generator,
    ) -> CGPGenome:
        genes = []
        for node in range(n_nodes):
            max_source = n_inputs + node
            genes.extend(
                [
                    int(rng.integers(0, len(OPS))),
                    int(rng.integers(0, max_source)),
                    int(rng.integers(0, max_source)),
                ]
            )
        return cls(n_inputs=n_inputs, genes=np.asarray(genes, dtype=np.int64))

    def evaluate(self, inputs: Sequence[np.ndarray]) -> np.ndarray:
        if len(inputs) != self.n_inputs:
            raise ValueError("wrong number of CGP inputs")
        values = [np.asarray(value, dtype=float) for value in inputs]
        for node in range(self.n_nodes):
            op, a_idx, b_idx = self.genes[node * 3 : node * 3 + 3]
            values.append(OPS[int(op)](values[int(a_idx)], values[int(b_idx)]))
        return np.nan_to_num(values[-1], nan=1e6, posinf=1e6, neginf=-1e6)

    def mutated(
        self,
        rng: np.random.Generator,
        mutation_rate: float = 0.1,
    ) -> CGPGenome:
        genes = self.genes.copy()
        for node in range(self.n_nodes):
            base = node * 3
            max_source = self.n_inputs + node
            if rng.random() < mutation_rate:
                genes[base] = rng.integers(0, len(OPS))
            if rng.random() < mutation_rate:
                genes[base + 1] = rng.integers(0, max_source)
            if rng.random() < mutation_rate:
                genes[base + 2] = rng.integers(0, max_source)
        return CGPGenome(self.n_inputs, genes)


def evolve_cgp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_nodes: int = 12,
    generations: int = 100,
    offspring: int = 8,
    seed: int = 0,
) -> tuple[CGPGenome, float]:
    """Fit one-dimensional symbolic regression using a (1+lambda) CGP strategy."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be same-shaped one-dimensional arrays")
    rng = np.random.default_rng(seed)
    parent = CGPGenome.random(1, n_nodes, rng)

    def loss(genome: CGPGenome) -> float:
        prediction = genome.evaluate([x])
        return float(np.mean((prediction - y) ** 2))

    parent_loss = loss(parent)
    for _ in range(generations):
        candidates = [parent.mutated(rng) for _ in range(offspring)]
        scored = [(loss(candidate), candidate) for candidate in candidates]
        candidate_loss, candidate = min(scored, key=lambda item: item[0])
        if candidate_loss <= parent_loss:
            parent, parent_loss = candidate, candidate_loss
    return parent, parent_loss


def _expand_grammar(
    codons: np.ndarray,
    grammar: dict[str, list[str]],
    max_steps: int = 64,
) -> str:
    text = "<expr>"
    cursor = 0
    for _ in range(max_steps):
        symbol = next((token for token in grammar if token in text), None)
        if symbol is None:
            break
        options = grammar[symbol]
        choice = options[int(codons[cursor % len(codons)]) % len(options)]
        cursor += 1
        text = text.replace(symbol, choice, 1)
    if "<" in text:
        raise ValueError("grammar expansion did not terminate")
    return text


def evolve_grammar(
    x: np.ndarray,
    y: np.ndarray,
    *,
    population_size: int = 60,
    generations: int = 60,
    codon_count: int = 24,
    seed: int = 0,
) -> tuple[str, float]:
    """Grammatical evolution for symbolic regression with a closed arithmetic grammar."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be same-shaped one-dimensional arrays")
    grammar = {
        "<expr>": ["(<expr>+<expr>)", "(<expr>-<expr>)", "(<expr>*<expr>)", "<var>"],
        "<var>": ["x", "1.0", "2.0", "-1.0"],
    }
    rng = np.random.default_rng(seed)
    population = rng.integers(
        0,
        256,
        size=(population_size, codon_count),
        dtype=np.int64,
    )

    def score(codons: np.ndarray) -> tuple[float, str]:
        try:
            expression = _expand_grammar(codons, grammar)
            prediction = eval(expression, {"__builtins__": {}}, {"x": x})
            prediction = np.asarray(prediction, dtype=float)
            if prediction.ndim == 0:
                prediction = np.full_like(x, float(prediction))
            prediction = np.nan_to_num(
                prediction,
                nan=1e6,
                posinf=1e6,
                neginf=-1e6,
            )
            return float(np.mean((prediction - y) ** 2)), expression
        except (ValueError, FloatingPointError, OverflowError, SyntaxError, TypeError):
            return float("inf"), "x"

    best_expression = "x"
    best_loss = float("inf")
    elite_count = max(2, population_size // 5)
    for _ in range(generations):
        scored = [score(individual) for individual in population]
        losses = np.asarray([item[0] for item in scored])
        order = np.argsort(losses)
        if losses[order[0]] < best_loss:
            best_loss = float(losses[order[0]])
            best_expression = scored[order[0]][1]
        elites = population[order[:elite_count]].copy()
        parent_idx = rng.integers(
            0,
            elite_count,
            size=(population_size - elite_count, 2),
        )
        children = []
        for a_idx, b_idx in parent_idx:
            point = int(rng.integers(1, codon_count))
            child = np.concatenate(
                [elites[a_idx, :point], elites[b_idx, point:]]
            ).copy()
            mask = rng.random(codon_count) < 0.05
            child[mask] = rng.integers(0, 256, size=int(np.sum(mask)))
            children.append(child)
        population = np.vstack([elites, np.asarray(children)])
    return best_expression, best_loss
