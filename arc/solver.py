"""
ARC-AGI Solver — Top-level interface for solving ARC tasks using
thermodynamic program synthesis.
"""
import sys
import os
import numpy as np
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from arc.dsl import apply_program
from arc.evolver import ProgramEvolver, Program
from arc.data import load_task, download_task, grid_to_string, task_summary


def solve_task(task: dict, generations: int = 200, population_size: int = 80,
               verbose: bool = True) -> dict:
    """
    Solve an ARC task using thermodynamic program synthesis.

    Returns:
        dict with keys: best_program, predictions, fitness, generations_used
    """
    train_examples = task["train"]
    test_examples = task["test"]

    if verbose:
        summary = task_summary(task)
        print(f"Task: {summary['n_train']} train, {summary['n_test']} test")
        print(f"Input shapes: {summary['input_shapes']}")
        print(f"Output shapes: {summary['output_shapes']}")
        print(f"Colors in: {sorted(summary['input_colors'])} | "
              f"Colors out: {sorted(summary['output_colors'])}")
        print()

    # Run evolution
    evolver = ProgramEvolver(population_size=population_size, max_program_len=6)
    best = evolver.evolve(train_examples, generations=generations, verbose=verbose)

    # Apply best program to test inputs
    predictions = []
    for test_ex in test_examples:
        input_grid = np.array(test_ex["input"])
        predicted = apply_program(input_grid, best.steps)
        predictions.append(predicted)

    # Check test accuracy (if answers available)
    test_accuracy = 0.0
    n_correct = 0
    for i, test_ex in enumerate(test_examples):
        if "output" in test_ex:
            expected = np.array(test_ex["output"])
            if predictions[i].shape == expected.shape and np.array_equal(predictions[i], expected):
                n_correct += 1
    if test_examples and "output" in test_examples[0]:
        test_accuracy = n_correct / len(test_examples)

    return {
        "best_program": best,
        "predictions": predictions,
        "train_fitness": best.fitness,
        "test_accuracy": test_accuracy,
        "generations_used": evolver.generation,
        "fitness_history": evolver.best_fitness_history,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ARC-AGI Solver — Thermodynamic Program Synthesis")
    parser.add_argument("--task", type=str, default=None,
                        help="Path to task JSON file, or task ID to download")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--sample", action="store_true",
                        help="Download and solve sample tasks")
    args = parser.parse_args()

    print("=" * 60)
    print("  ARC-AGI SOLVER — Thermodynamic Program Synthesis")
    print("  σ²·ε ≥ C_phys: Chaos-driven evolution of grid programs")
    print("=" * 60 + "\n")

    if args.sample:
        from arc.data import download_sample_tasks, list_local_tasks
        print("Downloading sample tasks...")
        downloaded = download_sample_tasks(10)
        print(f"Downloaded {len(downloaded)} tasks\n")

        results = []
        for task_id in downloaded[:5]:  # Solve first 5
            print(f"\n{'─' * 50}")
            print(f"TASK: {task_id}")
            print(f"{'─' * 50}")
            task = download_task(task_id)
            result = solve_task(task, generations=args.generations,
                                population_size=args.population)
            results.append((task_id, result))

            print(f"\nBest program: {result['best_program']}")
            print(f"Train fitness: {result['train_fitness']:.4f}")
            print(f"Test accuracy: {result['test_accuracy']:.4f}")

            # Show predictions
            for i, pred in enumerate(result["predictions"]):
                print(f"\nTest {i} prediction:")
                print(grid_to_string(pred))

        # Summary
        print(f"\n{'=' * 50}")
        print("RESULTS SUMMARY")
        print(f"{'=' * 50}")
        for task_id, result in results:
            status = "✅" if result["test_accuracy"] > 0.99 else "❌"
            print(f"  {status} {task_id}: train={result['train_fitness']:.3f} "
                  f"test={result['test_accuracy']:.3f} | {result['best_program']}")

    elif args.task:
        # Load single task
        if os.path.exists(args.task):
            task = load_task(args.task)
        else:
            task = download_task(args.task)

        result = solve_task(task, generations=args.generations,
                            population_size=args.population)

        print(f"\n{'─' * 50}")
        print(f"RESULT")
        print(f"{'─' * 50}")
        print(f"Best program: {result['best_program']}")
        print(f"Train fitness: {result['train_fitness']:.4f}")
        print(f"Test accuracy: {result['test_accuracy']:.4f}")

        for i, pred in enumerate(result["predictions"]):
            print(f"\nTest {i} prediction:")
            print(grid_to_string(pred))
            if "output" in task["test"][i]:
                print(f"\nExpected:")
                print(grid_to_string(task["test"][i]["output"]))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
