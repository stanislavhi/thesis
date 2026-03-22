"""
Multi-Strategy ARC Swarm — Parallel specialist evolvers communicating
best candidates through a noisy channel.

Architecture:
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Color Expert │  │Spatial Expert│  │ Geom Expert  │
    │ (swap,fill,  │  │ (gravity,    │  │ (rotate,flip │
    │  replace)    │  │  flood,crop) │  │  mirror,tile)│
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │
           └────────┬────────┴────────┬────────┘
                    │  Noisy Channel  │
                    ▼                 ▼
              ┌───────────────────────────┐
              │     Aggregator Pool       │
              │  (crossover + selection)  │
              └───────────────────────────┘

Each specialist evolves programs biased toward its domain. Every N
generations, they share their top candidates through the noisy channel
(with random perturbation). The aggregator pool mixes fragments from
different specialists.
"""
import sys
import os
import numpy as np
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core.chaos import LorenzGenerator
from arc.dsl import DSL_OPS, apply_program
from arc.evolver import ProgramEvolver, Program
from arc.hybrid_solver import GridAnalyzer, GuidedProgramEvolver


# Define specialist profiles — which ops each expert focuses on
SPECIALIST_PROFILES = {
    "color": {
        "replace_color": 8.0, "swap_colors": 8.0, "fill_bg": 6.0,
        "keep_color": 5.0, "invert_colors": 4.0,
        # Suppress non-color ops
        "rotate": 0.2, "flip_h": 0.2, "flip_v": 0.2, "transpose": 0.2,
        "gravity": 0.2, "tile": 0.2, "scale": 0.2,
    },
    "spatial": {
        "flood_fill": 8.0, "gravity": 8.0, "crop": 6.0, "largest_obj": 6.0,
        "hollow": 5.0, "draw_border": 5.0, "fill_rect": 5.0,
        # Suppress non-spatial ops
        "replace_color": 0.3, "swap_colors": 0.3, "tile": 0.3,
    },
    "geometric": {
        "rotate": 8.0, "flip_h": 8.0, "flip_v": 8.0, "transpose": 6.0,
        "mirror_h": 6.0, "mirror_v": 6.0, "tile": 5.0, "scale": 4.0,
        "repeat": 4.0, "pad": 3.0,
        # Suppress non-geometric ops
        "replace_color": 0.3, "swap_colors": 0.3, "flood_fill": 0.3,
    },
}


class NoisyChannel:
    """
    Simulates a bandwidth-limited noisy communication channel.
    Programs sent through it get perturbed — random param noise,
    occasional op drops, or op insertions.
    """
    def __init__(self, noise_level: float = 0.3):
        self.noise_level = noise_level
        self.chaos = LorenzGenerator()

    def transmit(self, program: Program) -> Program:
        """Send a program through the noisy channel."""
        result = program.copy()
        if not result.steps:
            return result

        z = abs(self.chaos.get_perturbation())

        # Light noise: tweak a parameter
        if z < 1.0 and result.steps:
            idx = np.random.randint(len(result.steps))
            name, params = result.steps[idx]
            if params:
                p_idx = np.random.randint(len(params))
                params[p_idx] = max(0, min(9, params[p_idx] + np.random.randint(-2, 3)))

        # Medium noise: drop a random op
        if z >= 1.0 and z < 2.0 and len(result.steps) > 1:
            idx = np.random.randint(len(result.steps))
            result.steps.pop(idx)

        return result


def crossover(parent_a: Program, parent_b: Program) -> Program:
    """
    Crossover two programs — take a prefix from A and suffix from B.
    """
    if not parent_a.steps or not parent_b.steps:
        return parent_a.copy() if parent_a.fitness >= parent_b.fitness else parent_b.copy()

    cut_a = np.random.randint(0, len(parent_a.steps) + 1)
    cut_b = np.random.randint(0, len(parent_b.steps) + 1)

    new_steps = parent_a.steps[:cut_a] + parent_b.steps[cut_b:]
    # Trim to max 6 steps
    new_steps = new_steps[:6]

    child = Program([(name, list(params)) for name, params in new_steps])
    return child


class SwarmSolver:
    """
    Multi-strategy swarm solver for ARC-AGI tasks.

    Multiple specialist evolvers run in parallel, sharing their best
    candidates periodically through a noisy channel. An aggregator pool
    mixes fragments from different specialists via crossover.
    """

    def __init__(self, population_per_specialist: int = 40,
                 share_interval: int = 15, n_shared: int = 3,
                 noise_level: float = 0.3):
        self.pop_per_spec = population_per_specialist
        self.share_interval = share_interval
        self.n_shared = n_shared
        self.channel = NoisyChannel(noise_level)
        self.best_fitness_history = []

    def solve(self, task: dict, generations: int = 300,
              verbose: bool = True) -> dict:
        """Run the swarm solver."""
        train_examples = task["train"]

        # Phase 1: Analyze task to boost relevant specialists
        analyzer = GridAnalyzer()
        global_weights = analyzer.analyze(train_examples)

        if verbose:
            from arc.data import task_summary
            summary = task_summary(task)
            print(f"Task: {summary['n_train']} train, {summary['n_test']} test")
            print(f"Shapes: {summary['input_shapes']} → {summary['output_shapes']}")

        # Phase 2: Initialize specialists
        specialists: Dict[str, GuidedProgramEvolver] = {}
        for name, profile in SPECIALIST_PROFILES.items():
            evolver = GuidedProgramEvolver(
                population_size=self.pop_per_spec,
                max_program_len=6,
            )
            # Merge profile with global analysis
            merged = {}
            all_ops = [op_name for op_name, _, _, _ in DSL_OPS]
            for op in all_ops:
                prof_w = profile.get(op, 1.0)
                glob_w = global_weights.get(op, 0.05)
                merged[op] = prof_w * (1.0 + glob_w * 5.0)  # Profile × analysis boost
            evolver.set_op_weights(merged)
            evolver.initialize()
            evolver.evaluate(evolver.population, train_examples)
            specialists[name] = evolver

        if verbose:
            print(f"Specialists: {', '.join(specialists.keys())}")
            print()

        # Phase 3: Parallel evolution with periodic sharing
        best_ever = Program([])
        best_ever.fitness = 0.0

        for gen in range(generations):
            # Evolve each specialist for one generation
            for name, evolver in specialists.items():
                evolver.population = evolver.select(evolver.population)
                evolver.population = evolver.mutate(evolver.population)
                evolver.evaluate(evolver.population, train_examples)

                spec_best = max(evolver.population, key=lambda p: p.fitness)
                if spec_best.fitness > best_ever.fitness:
                    best_ever = spec_best.copy()

            self.best_fitness_history.append(best_ever.fitness)

            # Periodic sharing through noisy channel
            if gen > 0 and gen % self.share_interval == 0:
                shared_programs = []
                for name, evolver in specialists.items():
                    top = sorted(evolver.population, key=lambda p: p.fitness, reverse=True)
                    for p in top[:self.n_shared]:
                        transmitted = self.channel.transmit(p)
                        shared_programs.append(transmitted)

                # Inject shared programs + crossovers into each specialist
                for name, evolver in specialists.items():
                    # Replace worst members with shared programs
                    evolver.population.sort(key=lambda p: p.fitness)
                    for i, shared in enumerate(shared_programs):
                        if i < len(evolver.population) // 3:
                            evolver.population[i] = shared.copy()

                    # Also create crossovers between shared candidates
                    for i in range(min(3, len(shared_programs) - 1)):
                        child = crossover(shared_programs[i], shared_programs[i + 1])
                        if len(evolver.population) > i + len(shared_programs):
                            evolver.population[len(shared_programs) + i] = child

                    evolver.evaluate(evolver.population, train_examples)

                if verbose:
                    spec_bests = {n: max(e.population, key=lambda p: p.fitness).fitness
                                  for n, e in specialists.items()}
                    print(f"Gen {gen:4d} | Share! | Specialists: "
                          f"{' | '.join(f'{n}={f:.3f}' for n, f in spec_bests.items())} "
                          f"| Best={best_ever.fitness:.4f}", flush=True)

            elif verbose and gen % 30 == 0:
                spec_bests = {n: max(e.population, key=lambda p: p.fitness).fitness
                              for n, e in specialists.items()}
                print(f"Gen {gen:4d} | Evolve | Specialists: "
                      f"{' | '.join(f'{n}={f:.3f}' for n, f in spec_bests.items())} "
                      f"| Best={best_ever.fitness:.4f}", flush=True)

            # Early termination
            if best_ever.fitness >= 1.0 - 1e-6:
                if verbose:
                    print(f"\n>>> PERFECT SOLUTION at gen {gen}!", flush=True)
                break

        # Apply to test
        predictions = []
        for test_ex in task["test"]:
            input_grid = np.array(test_ex["input"])
            predicted = apply_program(input_grid, best_ever.steps)
            predictions.append(predicted)

        n_correct = 0
        for i, test_ex in enumerate(task["test"]):
            if "output" in test_ex:
                expected = np.array(test_ex["output"])
                if predictions[i].shape == expected.shape and np.array_equal(predictions[i], expected):
                    n_correct += 1

        test_acc = n_correct / len(task["test"]) if task["test"] and "output" in task["test"][0] else 0.0

        return {
            "best_program": best_ever,
            "predictions": predictions,
            "train_fitness": best_ever.fitness,
            "test_accuracy": test_acc,
            "generations_used": gen if best_ever.fitness >= 1.0 - 1e-6 else generations,
            "fitness_history": self.best_fitness_history,
        }
