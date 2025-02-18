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
from src.ga.operators.fitness import get_fitness_function  #fitness_single_objective, fitness_multi_objective
from src.plotting import plot_optimization_progress

def load_yaml(path: Path) -> Any:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# def run_damage_scenario(beam_scenario_path: Path) -> Dict[str, Any]:
#     scenario = load_yaml(beam_scenario_path)
#     beam_config = load_yaml(Path(scenario["beam_config"]))
#     beam = Beam(config=beam_config)
#     for case in scenario["damage_cases"]:
#         beam.apply_damage(case["element"], case["factor"])
#     n_modes = scenario.get("n_modes", 4)
#     target_frequencies, target_modes = beam.get_modal_properties(n_eigen=n_modes)
#     target_props = {"frequencies": target_frequencies, "modes": target_modes}
#     return target_props

# def create_fitness_func(beam_config: dict, target_props: dict, fitness_config: dict):
#     save_results = fitness_config.get("save_results", True)
#     save_filepath = fitness_config.get("save_filepath", "fitness_results.jsonl")
    
#     def fitness_func(ga_instance, solution: np.ndarray, solution_idx: int) -> float:
#         return fitness_single_objective(solution,
#                                         target_props,
#                                         beam_config,
#                                         ga_instance=ga_instance,
#                                         save_results=save_results,
#                                         save_filepath=save_filepath)
#     return fitness_func

def create_fitness_func(beam_config: dict, scenario_config: dict, fitness_config: dict):
    """Create fitness function with target beam"""
    # Create target beam with damage scenario
    target_beam = Beam(config=beam_config)
    target_beam.apply_damage(scenario_config)

    # Get fitness function from factory
    return get_fitness_function(
        name=fitness_config.get("type", "standard"),
        target_beam=target_beam,
        beam_config=beam_config,
        config=fitness_config.get("parameters", {})
    )




def main() -> None:
    beam_scenario_path = Path("config/beam_scenario.yaml")
    beam_properties_path = Path("config/beam_properties.yaml")
    ga_config_path = Path("config/GA_config.yaml")
    
    ga_config = load_yaml(ga_config_path)
    beam_config = load_yaml(beam_properties_path)
    scenario_configs = load_yaml(beam_scenario_path)
    max_seed = 7
    for random_seed in range(4,max_seed):
        ga_config["genetic_algorithm"]["random_seed"] = random_seed
        for noise_stddev_factor in [0.0, 0.05, 0.1, 0.2]:
            ga_config["optimization"]["fitness"]["parameters"]["noise_stddev_factor"] = noise_stddev_factor
            for scenario in scenario_configs["damage_cases"]:
                fitness_config = ga_config.get("optimization", {}).get("fitness", {})
                fitness_func = create_fitness_func(beam_config, scenario, fitness_config)
                
                optimizer = GAOptimizer(config=ga_config)
                # optimizer = GAOptimizer(saved_run_path=Path("results/ga_optimization/20250209_153706"))
                optimizer.fitness_func = fitness_func
                print("Starting optimization...")
                optimizer.optimize(scenario_config=scenario) # providing scenario config to save damage scenario with results
                
                solution, best_fitness, _ = optimizer.ga_instance.best_solution()
                print("\nBest solution E_vector:")
                for i, e in enumerate(solution):
                    print(f"Segment {i+1}: {np.round(e/beam_config['material']['E'],4)}") #rounds to 2 decimals
                print(f"Fitness: {best_fitness}")
                
                save_dir = optimizer.save()
                print(f"Results saved in: {save_dir}")

                print(optimizer.ga_instance.summary())
# optimizer.ga_instance.plot_fitness()


    

if __name__ == "__main__":
    main()