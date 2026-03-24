"""
Program Evolver — Thermodynamic evolution of DSL programs for ARC-AGI.

Uses the same stagnation detection + Lorenz chaos injection pattern as the RL agent,
but evolves programs (lists of grid operations) instead of neural network topology.
"""
import sys
import os
import numpy as np
from typing import List, Tuple, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.chaos import LorenzGenerator
from arc.dsl import DSL_OPS, DSL_REGISTRY, apply_program


def evaluate_test(best_steps: list, test_examples: list) -> tuple:
    """
    Apply a program to test examples and compute accuracy.
    Returns (predictions, test_accuracy).
    """
    predictions = []
    for test_ex in test_examples:
        input_grid = np.array(test_ex["input"])
        predicted = apply_program(input_grid, best_steps)
        predictions.append(predicted)

    n_correct = 0
    has_answers = test_examples and "output" in test_examples[0]
    for i, test_ex in enumerate(test_examples):
        if "output" in test_ex:
            expected = np.array(test_ex["output"])
            if predictions[i].shape == expected.shape and np.array_equal(predictions[i], expected):
                n_correct += 1

    test_accuracy = n_correct / len(test_examples) if has_answers else 0.0
    return predictions, test_accuracy


class Program:
    """A candidate program = sequence of (op_name, params) steps."""

    def __init__(self, steps: Optional[List[Tuple[str, list]]] = None):
        if steps is None:
            steps = []
        self.steps = steps
        self.fitness = 0.0

    def copy(self):
        p = Program([(name, list(params)) for name, params in self.steps])
        p.fitness = self.fitness
        return p

    def __repr__(self):
        if not self.steps:
            return "Program([])"
        ops = " -> ".join(f"{name}({','.join(map(str,p))})" if p else name
                          for name, p in self.steps)
        return f"Program[{ops}] (fitness={self.fitness:.3f})"

    def __len__(self):
        return len(self.steps)


class ProgramEvolver:
    """
    Evolves a population of programs using thermodynamic chaos injection.

    The Lorenz attractor drives mutations:
    - Low |Z|: fine-tune parameters, swap adjacent ops
    - Medium |Z|: insert/delete operations
    - High |Z|: radical restructure (replace subsequence)
    """

    def __init__(self, population_size: int = 50, max_program_len: int = 6):
        self.pop_size = population_size
        self.max_len = max_program_len
        self.chaos = LorenzGenerator()
        self.generation = 0
        self.stagnation_count = 0
        self.best_fitness_history = []
        self.population: List[Program] = []

    def _random_op(self) -> Tuple[str, list]:
        """Generate a random DSL operation with random valid params."""
        name, func, n_params, param_ranges = DSL_OPS[np.random.randint(len(DSL_OPS))]
        params = [np.random.randint(lo, hi + 1) for lo, hi in param_ranges]
        return (name, params)

    def _random_program(self) -> Program:
        """Generate a random program of 1-max_len steps."""
        length = np.random.randint(1, self.max_len + 1)
        steps = [self._random_op() for _ in range(length)]
        return Program(steps)

    def initialize(self) -> List[Program]:
        """Create initial population."""
        self.population = [self._random_program() for _ in range(self.pop_size)]
        # Always include the identity (empty) program
        self.population[0] = Program([])
        # And some simple single-op programs
        for i, (name, func, n_params, param_ranges) in enumerate(DSL_OPS):
            if i + 1 < self.pop_size:
                params = [lo for lo, hi in param_ranges]
                self.population[i + 1] = Program([(name, params)])
        return self.population

    def evaluate(self, population: List[Program],
                 train_examples: List[dict]) -> List[Program]:
        """
        Evaluate each program against ARC training examples.
        Fitness = fraction of pixels correct, averaged across examples.
        """
        for prog in population:
            total_score = 0.0
            for example in train_examples:
                input_grid = np.array(example["input"])
                expected = np.array(example["output"])

                try:
                    predicted = apply_program(input_grid, prog.steps)
                except Exception:
                    continue

                # Score: pixel-level accuracy (shape must match too)
                if predicted.shape == expected.shape:
                    match = np.sum(predicted == expected)
                    total = expected.size
                    total_score += match / total
                else:
                    # Partial credit for getting dimensions right
                    if predicted.shape[0] == expected.shape[0]:
                        total_score += 0.05
                    if predicted.shape[1] == expected.shape[1]:
                        total_score += 0.05

            prog.fitness = total_score / max(len(train_examples), 1)

        return population

    def select(self, population: List[Program]) -> List[Program]:
        """Tournament selection — keep the best half, clone to fill."""
        sorted_pop = sorted(population, key=lambda p: p.fitness, reverse=True)

        # Keep top 50%
        elite_size = max(2, self.pop_size // 2)
        elite = sorted_pop[:elite_size]

        # Fill the rest with clones of elite (will be mutated)
        new_pop = [p.copy() for p in elite]
        while len(new_pop) < self.pop_size:
            parent = elite[np.random.randint(len(elite))]
            new_pop.append(parent.copy())

        return new_pop

    def mutate(self, population: List[Program]) -> List[Program]:
        """
        Thermodynamic mutation driven by Lorenz chaos.

        Chaos magnitude |Z| determines mutation severity:
        - |Z| < 0.5: tweak a parameter
        - 0.5 <= |Z| < 1.5: insert or delete an operation
        - |Z| >= 1.5: radical restructure
        """
        # Never mutate the best program
        for i in range(1, len(population)):
            prog = population[i]
            z = self.chaos.get_perturbation()
            magnitude = abs(z)

            if magnitude < 0.5:
                # Fine-tune: mutate a parameter
                self._mutate_param(prog)
            elif magnitude < 1.5:
                # Structural: insert or delete
                if z > 0 and len(prog) < self.max_len:
                    self._insert_op(prog)
                elif len(prog) > 1:
                    self._delete_op(prog)
                else:
                    self._mutate_param(prog)
            else:
                # Radical: replace a subsequence
                self._radical_restructure(prog)

        return population

    def _mutate_param(self, prog: Program):
        """Mutate a single parameter in a random operation."""
        if not prog.steps:
            prog.steps.append(self._random_op())
            return
        idx = np.random.randint(len(prog.steps))
        name, params = prog.steps[idx]
        _, n_params, param_ranges = DSL_REGISTRY[name]
        if n_params > 0:
            p_idx = np.random.randint(n_params)
            lo, hi = param_ranges[p_idx]
            params[p_idx] = np.random.randint(lo, hi + 1)
        else:
            # No params to mutate — swap with a random op
            prog.steps[idx] = self._random_op()

    def _insert_op(self, prog: Program):
        """Insert a random operation at a random position."""
        pos = np.random.randint(len(prog.steps) + 1)
        prog.steps.insert(pos, self._random_op())

    def _delete_op(self, prog: Program):
        """Delete a random operation."""
        if prog.steps:
            idx = np.random.randint(len(prog.steps))
            prog.steps.pop(idx)

    def _radical_restructure(self, prog: Program):
        """Replace a section of the program with new random ops."""
        if len(prog.steps) <= 1:
            prog.steps = [self._random_op() for _ in range(np.random.randint(1, 4))]
            return
        # Replace a random contiguous subsequence
        start = np.random.randint(len(prog.steps))
        end = min(start + np.random.randint(1, 3), len(prog.steps))
        new_ops = [self._random_op() for _ in range(np.random.randint(1, 3))]
        prog.steps[start:end] = new_ops
        # Trim if too long
        prog.steps = prog.steps[:self.max_len]

    def evolve(self, train_examples: List[dict], generations: int = 100,
               verbose: bool = True) -> Program:
        """
        Run the full thermodynamic evolution loop.
        Returns the best program found.
        """
        self.initialize()
        self.evaluate(self.population, train_examples)

        best_ever = max(self.population, key=lambda p: p.fitness).copy()
        prev_best = best_ever.fitness

        for gen in range(generations):
            self.generation = gen

            # Select + mutate
            self.population = self.select(self.population)
            self.population = self.mutate(self.population)

            # Evaluate
            self.evaluate(self.population, train_examples)

            # Track best
            gen_best = max(self.population, key=lambda p: p.fitness)
            if gen_best.fitness > best_ever.fitness:
                best_ever = gen_best.copy()

            self.best_fitness_history.append(best_ever.fitness)

            # Stagnation detection
            if abs(gen_best.fitness - prev_best) < 1e-6:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
            prev_best = gen_best.fitness

            # Thermodynamic intervention on stagnation
            if self.stagnation_count > 10:
                if verbose:
                    print(f"   [Gen {gen}] Stagnation! Chaos restructure...", flush=True)
                # Force radical mutations on bottom 70%
                sorted_pop = sorted(self.population, key=lambda p: p.fitness, reverse=True)
                for i in range(self.pop_size // 3, len(sorted_pop)):
                    self._radical_restructure(sorted_pop[i])
                self.population = sorted_pop
                self.stagnation_count = 0

            if verbose and gen % 10 == 0:
                avg_f = np.mean([p.fitness for p in self.population])
                print(f"Gen {gen:4d} | Best: {best_ever.fitness:.4f} | "
                      f"Avg: {avg_f:.4f} | Pop best: {gen_best}", flush=True)

            # Early termination — perfect score
            if best_ever.fitness >= 1.0 - 1e-6:
                if verbose:
                    print(f"\n>>> PERFECT SOLUTION at gen {gen}!", flush=True)
                break

        return best_ever
