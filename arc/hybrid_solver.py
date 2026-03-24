"""
Hybrid ARC Solver — Neural-guided thermodynamic program synthesis.

Uses a small feature extractor to analyze input/output grid pairs and
predict which DSL operations are likely useful. This biases the search
instead of pure random mutation, dramatically improving solve rates.
"""
import sys
import os
import numpy as np
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from arc.dsl import DSL_OPS
from arc.evolver import ProgramEvolver, Program, evaluate_test


class GridAnalyzer:
    """
    Analyzes input/output grid pairs to extract features that hint at
    which DSL operations are likely to solve the task.

    No neural network needed — just smart heuristics derived from
    grid properties. This is faster and more interpretable than a CNN.
    """

    def __init__(self):
        self.op_names = [name for name, _, _, _ in DSL_OPS]

    def analyze(self, train_examples: List[dict]) -> dict:
        """
        Extract features from training examples and compute op probabilities.
        Returns dict of op_name -> probability (0.0 to 1.0).
        """
        features = self._extract_features(train_examples)
        return self._features_to_op_weights(features)

    def _extract_features(self, examples: List[dict]) -> dict:
        """Extract structural features from I/O pairs."""
        features = {
            "same_shape": True,
            "shape_ratio": [],
            "color_changes": [],
            "n_colors_change": [],
            "has_symmetry_h": False,
            "has_symmetry_v": False,
            "output_is_tiled": False,
            "output_larger": False,
            "output_smaller": False,
            "bg_changes": False,
            "n_objects_change": False,
            "colors_swapped": False,
            "has_separator_v": False,
            "has_separator_h": False,
            "separator_color": 0,
            "output_extends_input": False,
        }

        for ex in examples:
            inp = np.array(ex["input"])
            out = np.array(ex["output"])

            # Shape analysis
            if inp.shape != out.shape:
                features["same_shape"] = False
            if inp.shape[0] > 0 and inp.shape[1] > 0:
                features["shape_ratio"].append(
                    (out.shape[0] / inp.shape[0], out.shape[1] / inp.shape[1])
                )
            features["output_larger"] |= (out.size > inp.size)
            features["output_smaller"] |= (out.size < inp.size)

            # Color analysis
            in_colors = set(np.unique(inp))
            out_colors = set(np.unique(out))
            features["color_changes"].append(in_colors != out_colors)
            features["n_colors_change"].append(len(out_colors) != len(in_colors))

            # Background changes
            if 0 in in_colors and 0 in out_colors:
                in_bg = np.sum(inp == 0)
                out_bg = np.sum(out == 0)
                if abs(in_bg - out_bg) > inp.size * 0.1:
                    features["bg_changes"] = True

            # Color swap detection
            for c1 in in_colors:
                for c2 in in_colors:
                    if c1 < c2:
                        mask1_in = inp == c1
                        mask2_in = inp == c2
                        if out.shape == inp.shape:
                            if np.any(mask1_in) and np.any(mask2_in):
                                if np.all(out[mask1_in] == c2) and np.all(out[mask2_in] == c1):
                                    features["colors_swapped"] = True

            # Symmetry detection on output
            if out.shape[0] > 1 and out.shape[1] > 1:
                features["has_symmetry_h"] |= np.array_equal(out, np.fliplr(out))
                features["has_symmetry_v"] |= np.array_equal(out, np.flipud(out))

            # Tiling detection
            if out.shape[0] > 1 and out.shape[1] > 1 and inp.shape[0] > 0 and inp.shape[1] > 0:
                if out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0:
                    rr = out.shape[0] // inp.shape[0]
                    cr = out.shape[1] // inp.shape[1]
                    if rr >= 2 or cr >= 2:
                        tiled = np.tile(inp, (rr, cr))
                        if np.array_equal(tiled, out):
                            features["output_is_tiled"] = True

            # Separator detection — uniform color row/column in input
            h, w = inp.shape
            for c in range(w):
                col = inp[:, c]
                if len(np.unique(col)) == 1 and col[0] != 0:
                    features["has_separator_v"] = True
                    features["separator_color"] = int(col[0])
            for r in range(h):
                row = inp[r, :]
                if len(np.unique(row)) == 1 and row[0] != 0:
                    features["has_separator_h"] = True
                    features["separator_color"] = int(row[0])

            # Pattern extension — output is ~1.5x height of input
            if out.shape[1] == inp.shape[1] and inp.shape[0] > 0:
                ratio = out.shape[0] / inp.shape[0]
                if 1.3 < ratio < 2.1:
                    features["output_extends_input"] = True

        return features

    def _features_to_op_weights(self, features: dict) -> dict:
        """Convert features to operation probability weights."""
        weights = {name: 1.0 for name in self.op_names}

        # Boost relevant ops based on features
        if features["colors_swapped"]:
            weights["swap_colors"] = 8.0
            weights["replace_color"] = 5.0

        if any(features["color_changes"]):
            weights["replace_color"] = max(weights["replace_color"], 4.0)
            weights["swap_colors"] = max(weights["swap_colors"], 3.0)
            weights["fill_bg"] = 3.0
            weights["invert_colors"] = 2.0

        if features["output_is_tiled"]:
            weights["tile"] = 10.0
            weights["repeat"] = 5.0

        if features["output_larger"]:
            weights["tile"] = max(weights["tile"], 4.0)
            weights["scale"] = 4.0
            weights["pad"] = 3.0
            weights["mirror_h"] = 3.0
            weights["mirror_v"] = 3.0
            weights["repeat"] = 3.0
            weights["extend_v"] = 5.0

        if features["output_smaller"]:
            weights["crop"] = 6.0
            weights["keep_color"] = 4.0
            weights["largest_obj"] = 5.0
            weights["top_half"] = 3.0
            weights["bottom_half"] = 3.0
            weights["left_half"] = 3.0
            weights["right_half"] = 3.0

        if features["has_symmetry_h"]:
            weights["mirror_h"] = 5.0
            weights["flip_h"] = 3.0

        if features["has_symmetry_v"]:
            weights["mirror_v"] = 5.0
            weights["flip_v"] = 3.0

        if features["same_shape"]:
            for op in ["tile", "scale", "pad", "crop", "mirror_h", "mirror_v", "repeat",
                        "extend_v", "top_half", "bottom_half", "left_half", "right_half"]:
                if op in weights and weights[op] <= 2.0:
                    weights[op] = 0.3

        if features["bg_changes"]:
            weights["fill_bg"] = max(weights["fill_bg"], 4.0)
            weights["flood_fill"] = 3.0
            weights["draw_border"] = 2.0

        # Separator → split + overlay ops
        if features["has_separator_v"]:
            weights["split_v"] = 8.0
            weights["overlay_and"] = 10.0
            weights["left_half"] = 4.0
            weights["right_half"] = 4.0

        if features["has_separator_h"]:
            weights["split_h"] = 8.0
            weights["top_half"] = 4.0
            weights["bottom_half"] = 4.0

        # Pattern extension
        if features["output_extends_input"]:
            weights["extend_v"] = 10.0
            weights["repeat"] = 5.0

        # Normalize to probabilities
        total = sum(weights.values())
        return {name: w / total for name, w in weights.items()}


class GuidedProgramEvolver(ProgramEvolver):
    """
    Extends ProgramEvolver with neural/heuristic guidance.
    Random op selection is biased by the GridAnalyzer's predictions.
    """

    def __init__(self, op_weights: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.op_weights = op_weights or {}
        self._op_probs = None

    def set_op_weights(self, weights: dict):
        """Set the probability distribution over DSL operations."""
        self.op_weights = weights
        names = [name for name, _, _, _ in DSL_OPS]
        probs = np.array([weights.get(name, 1.0) for name in names])
        probs = probs / probs.sum()
        self._op_probs = probs

    def _random_op(self):
        """Override: choose ops according to learned weights instead of uniform."""
        if self._op_probs is not None:
            idx = np.random.choice(len(DSL_OPS), p=self._op_probs)
        else:
            idx = np.random.randint(len(DSL_OPS))
        name, func, n_params, param_ranges = DSL_OPS[idx]
        params = [np.random.randint(lo, hi + 1) for lo, hi in param_ranges]
        return (name, params)


def solve_task_hybrid(task: dict, generations: int = 300, population_size: int = 100,
                      verbose: bool = True) -> dict:
    """
    Solve an ARC task using the hybrid approach:
    1. Analyze I/O pairs to predict useful ops
    2. Bias the evolutionary search toward those ops
    3. Run thermodynamic evolution with guided mutations
    """
    from arc.data import task_summary, grid_to_string

    train_examples = task["train"]
    test_examples = task["test"]

    if verbose:
        summary = task_summary(task)
        print(f"Task: {summary['n_train']} train, {summary['n_test']} test")
        print(f"Shapes: {summary['input_shapes']} → {summary['output_shapes']}")

    # Phase 1: Analyze
    analyzer = GridAnalyzer()
    op_weights = analyzer.analyze(train_examples)

    if verbose:
        top_ops = sorted(op_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top predicted ops: {', '.join(f'{n}({w:.2f})' for n, w in top_ops)}")
        print()

    # Phase 2: Guided evolution
    evolver = GuidedProgramEvolver(population_size=population_size, max_program_len=6)
    evolver.set_op_weights(op_weights)
    best = evolver.evolve(train_examples, generations=generations, verbose=verbose)

    # Phase 3: Apply to test and score
    predictions, test_accuracy = evaluate_test(best.steps, test_examples)

    return {
        "best_program": best,
        "predictions": predictions,
        "train_fitness": best.fitness,
        "test_accuracy": test_accuracy,
        "generations_used": evolver.generation,
        "fitness_history": evolver.best_fitness_history,
        "op_weights": op_weights,
    }
