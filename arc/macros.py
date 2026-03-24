"""
Self-Inventing Primitives — Macro system for the ARC DSL.

When the evolver discovers a useful multi-step sequence, it can be
"compiled" into a single named macro and added to the DSL registry.
This lets the system literally invent new operations.
"""
import sys
import os
import numpy as np
from typing import List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from arc.dsl import DSL_OPS, DSL_REGISTRY, apply_program, apply_op


class Macro:
    """A compound operation made from a sequence of primitives."""

    def __init__(self, name: str, steps: List[Tuple[str, list]], fitness: float = 0.0):
        self.name = name
        self.steps = steps
        self.fitness = fitness
        self.times_used = 0

    def apply(self, grid: np.ndarray) -> np.ndarray:
        """Apply the macro's sequence of operations."""
        return apply_program(grid, self.steps)

    def __repr__(self):
        ops = " → ".join(
            f"{n}({','.join(map(str,p))})" if p else n
            for n, p in self.steps
        )
        return f"Macro<{self.name}: {ops}> (used={self.times_used})"


class MacroLibrary:
    """
    A library of learned macros. Programs can reference macros as
    single operations, compressing successful patterns.
    """

    def __init__(self, max_macros: int = 10):
        self.max_macros = max_macros
        self.macros: List[Macro] = []
        self._counter = 0
        self._registered_names: List[str] = []

    def extract_macro(self, program_steps: List[Tuple[str, list]],
                      start: int, end: int, fitness: float) -> Optional[str]:
        """
        Extract a subsequence from a program and register it as a macro.
        Returns the macro name.
        """
        sub_steps = [(name, list(params)) for name, params in program_steps[start:end]]
        if len(sub_steps) < 2:
            return None

        # Check for duplicates
        for existing in self.macros:
            if self._steps_equal(existing.steps, sub_steps):
                existing.times_used += 1
                return existing.name

        self._counter += 1
        name = f"macro_{self._counter}"
        macro = Macro(name, sub_steps, fitness)

        if len(self.macros) >= self.max_macros:
            # Evict least-used macro
            self.macros.sort(key=lambda m: m.times_used)
            evicted = self.macros.pop(0)
            self._unregister_macro(evicted.name)

        self.macros.append(macro)

        # Register in DSL
        self._register_macro(macro)

        return name

    def _register_macro(self, macro: Macro):
        """Add macro to the DSL registry so programs can reference it."""
        def macro_fn(grid):
            return macro.apply(grid)

        DSL_OPS.append((macro.name, macro_fn, 0, []))
        DSL_REGISTRY[macro.name] = (macro_fn, 0, [])
        self._registered_names.append(macro.name)

    def _unregister_macro(self, name: str):
        """Remove a macro from the DSL registry."""
        DSL_REGISTRY.pop(name, None)
        for i, (op_name, _, _, _) in enumerate(DSL_OPS):
            if op_name == name:
                DSL_OPS.pop(i)
                break
        if name in self._registered_names:
            self._registered_names.remove(name)

    def unregister_all(self):
        """Remove all macros from the DSL registry. Call between solver runs."""
        for name in list(self._registered_names):
            self._unregister_macro(name)
        self.macros.clear()
        self._registered_names.clear()

    def _steps_equal(self, a, b):
        if len(a) != len(b):
            return False
        for (n1, p1), (n2, p2) in zip(a, b):
            if n1 != n2 or p1 != p2:
                return False
        return True

    def learn_from_population(self, population, min_fitness: float = 0.5):
        """
        Scan a population for common subsequences in high-fitness programs.
        Extract them as macros.
        """
        good_programs = [p for p in population if p.fitness >= min_fitness and len(p.steps) >= 3]

        if not good_programs:
            return []

        # Find common 2-3 step subsequences
        subseq_counts = {}
        for prog in good_programs:
            for length in [2, 3]:
                for i in range(len(prog.steps) - length + 1):
                    sub = tuple((n, tuple(p)) for n, p in prog.steps[i:i+length])
                    if sub not in subseq_counts:
                        subseq_counts[sub] = {"count": 0, "fitness": 0.0}
                    subseq_counts[sub]["count"] += 1
                    subseq_counts[sub]["fitness"] = max(subseq_counts[sub]["fitness"],
                                                        prog.fitness)

        # Extract the most common subsequences as macros
        new_macros = []
        for sub, info in sorted(subseq_counts.items(),
                                key=lambda x: x[1]["count"] * x[1]["fitness"],
                                reverse=True)[:3]:
            if info["count"] >= 2 and info["fitness"] >= min_fitness:
                steps = [(n, list(p)) for n, p in sub]
                name = self.extract_macro(steps, 0, len(steps), info["fitness"])
                if name:
                    new_macros.append(name)

        return new_macros

    def summary(self) -> str:
        """Human-readable summary of the macro library."""
        if not self.macros:
            return "Macro Library: empty"
        lines = ["Macro Library:"]
        for m in sorted(self.macros, key=lambda m: m.times_used, reverse=True):
            lines.append(f"  {m}")
        return "\n".join(lines)
