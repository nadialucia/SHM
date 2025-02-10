from pathlib import Path
import os
import sys
import yaml
import numpy as np
import json
from typing import Dict, Any

# Add project root to Python path for Abaqus compatibility
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.dirname(os.path.abspath('src/main.py'))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.beam.beam import Beam
from src.ga.optimizer import GAOptimizer
from src.ga.operators.fitness import fitness_single_objective, fitness_multi_objective
from src.plotting import plot_optimization_progress

def load_yaml(path: Path) -> Any:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_damage_scenario(beam_scenario_path: Path) -> Dict[str, Any]:
    scenario = load_yaml(beam_scenario_path)
    beam_config = load_yaml(Path(scenario["beam_config"]))
    beam = Beam(config=beam_config)
    for case in scenario["damage_cases"]:
        beam.apply_damage(case["element"], case["factor"])
    n_modes = scenario.get("n_modes", 4)
    target_frequencies, target_modes = beam.get_modal_properties(n_eigen=n_modes)
    target_props = {"frequencies": target_frequencies, "modes": target_modes}
    return target_props

def create_fitness_func(beam_config: dict, target_props: dict, fitness_config: dict):
    save_results = fitness_config.get("save_results", True)
    save_filepath = fitness_config.get("save_filepath", "fitness_results.jsonl")
    
    def fitness_func(ga_instance, solution: np.ndarray, solution_idx: int) -> float:
        return fitness_single_objective(solution,
                                        target_props,
                                        beam_config,
                                        ga_instance=ga_instance,
                                        save_results=save_results,
                                        save_filepath=save_filepath)
    return fitness_func


def main() -> None:
    beam_scenario_path = Path("config/beam_scenario.yaml")
    beam_properties_path = Path("config/beam_properties.yaml")
    ga_config_path = Path("config/GA_config.yaml")
    
    ga_config = load_yaml(ga_config_path)
    beam_config = load_yaml(beam_properties_path)
    
    fitness_config = ga_config.get("fitness", {})
    target_props = run_damage_scenario(beam_scenario_path)
    fitness_func = create_fitness_func(beam_config, target_props, fitness_config)
    
    optimizer = GAOptimizer(config=ga_config)
    #optimizer = GAOptimizer(saved_run_path=Path("results/ga_optimization/20250209_153706"))
    optimizer.fitness_func = fitness_func
    print("Starting optimization...")
    optimizer.optimize()
    
    solution, best_fitness, _ = optimizer.ga_instance.best_solution()
    print("\nBest solution E_vector:")
    for i, e in enumerate(solution):
        print(f"Segment {i+1}: {np.round(e/beam_config['material']['E'],2)}")
    print(f"Fitness: {best_fitness:.6f}")
    
    save_dir = optimizer.save()
    print(f"Results saved in: {save_dir}")

    print(optimizer.ga_instance.summary())
    optimizer.ga_instance.plot_fitness()


    

if __name__ == "__main__":
    main()