from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Callable

class FitnessFunction(ABC):
    """Base class for all fitness functions"""
    def __init__(self, beam, config: Dict[str, Any]):
        self.beam = beam
        self.config = config
        
    @abstractmethod
    def __call__(self, ga_instance, solution, solution_idx) -> float:
        pass

class StandardFitness(FitnessFunction):
    """Original modal-based fitness"""
    def __call__(self, ga_instance, solution, solution_idx) -> float:
        freq_A, modes_A = self.beam.get_modal_properties(solution)
        n_modes = len(self.beam.frequencies)
        
        mode_objectives = []
        for i in range(n_modes):
            freq_diff = abs(freq_A[i] - self.beam.frequencies[i])
            mac = self._calculate_MAC(modes_A[i], self.beam.mode_shapes[i])
            mode_obj = freq_diff + (1 - mac)
            mode_objectives.append(mode_obj)
            
        total_objective = sum(mode_objectives) / n_modes
        return 1.0 / (total_objective + 1e-10)

class SmoothedFitness(FitnessFunction):
    """Fitness with smoothness penalty"""
    def __call__(self, ga_instance, solution, solution_idx) -> float:
        standard_fitness = StandardFitness(self.beam, self.config)(
            ga_instance, solution, solution_idx
        )
        
        stdev_penalty = (np.std(solution) / np.mean(solution)) * \
                       self.config.get("smoothness_weight", 10.0)
        
        return standard_fitness / (1 + stdev_penalty)

class AdaptiveFitness(FitnessFunction):
    """Generation-dependent adaptive fitness"""
    def __call__(self, ga_instance, solution, solution_idx) -> float:
        standard_fitness = StandardFitness(self.beam, self.config)(
            ga_instance, solution, solution_idx
        )
        
        generation = ga_instance.generations_completed
        max_gen = ga_instance.num_generations
        adaptation = min(generation / max_gen, 1.0)
        
        stdev_penalty = adaptation * \
                       (np.std(solution) / np.mean(solution)) * \
                       self.config.get("smoothness_weight", 10.0)
                       
        return standard_fitness / (1 + stdev_penalty)

def get_fitness_function(name: str, beam, config: Dict[str, Any]) -> Callable:
    """Factory function to return appropriate fitness function"""
    fitness_functions = {
        "standard": StandardFitness,
        "smoothed": SmoothedFitness,
        "adaptive": AdaptiveFitness
    }
    
    if name not in fitness_functions:
        raise ValueError(f"Unknown fitness function: {name}. " \
                       f"Available options: {list(fitness_functions.keys())}")
    
    return lambda *args: fitness_functions[name](beam, config)(*args)



###################### SIMPLER APPROACH ###############################

#### filepath: /C:/Users/attri/OneDrive/Documents/Academic Work Nadia/MEng Y5/CEE Thesis/SEM2/SHM/src/ga/operators/fitness.py
import numpy as np
import json
from src.beam.beam import Beam
import os

def calculate_MAC(phi_A: np.ndarray, phi_E: np.ndarray) -> float:
    """
    Calculate the Modal Assurance Criterion (MAC) between two mode shapes.
    """
    numerator = np.abs(np.dot(phi_A, phi_E)) ** 2
    denominator = np.dot(phi_A, phi_A) * np.dot(phi_E, phi_E)
    return numerator / denominator

def calculate_frequency_diff(f_A: float, f_E: float) -> float:
    """
    Calculate a normalized frequency difference.
    """
    return abs(f_A - f_E) / f_E

def save_fitness_evaluation(evaluation: dict, save_filepath: str) -> None:
    """
    Append an evaluation dictionary as a JSON record (line-delimited).
    """
    with open(save_filepath, "a") as f:
        json.dump(evaluation, f)
        f.write("\n")

def fitness_single_objective(candidate_solution: np.ndarray,
                             target_props: dict,
                             beam_config: dict,
                             ga_instance=None,
                             weight_freq: float = 1,
                             weight_mode: float = 1,
                             save_results: bool = False,
                             save_filepath: str = "fitness_results.jsonl") -> float:
    candidate_beam = Beam(config=beam_config, E_vector=candidate_solution)
    n_modes = len(target_props["frequencies"])
    cand_freq, cand_modes = candidate_beam.get_modal_properties(n_eigen=n_modes)
    
    freq_diffs = [calculate_frequency_diff(cand_freq[i], target_props["frequencies"][i])
                  for i in range(n_modes)]
    total_freq_diff = np.sum(freq_diffs)
    
    mac_diffs = [1 - calculate_MAC(cand_modes[i], target_props["modes"][i])
                 for i in range(n_modes)]
    total_mac_diff = np.sum(mac_diffs)
    
    total_error = weight_freq * total_freq_diff + weight_mode * total_mac_diff
    epsilon = 1e-10
    fitness = 1.0 / (total_error + epsilon)
    
    if save_results and ga_instance is not None:
        filepath = save_filepath
        # If the provided filepath is not absolute, join it with ga_instance run_dir
        # using only the basename so that it is saved alongside other GA results.
        if not os.path.isabs(filepath) and hasattr(ga_instance, "run_dir"):
            filepath = os.path.join(ga_instance.run_dir, os.path.basename(save_filepath))
        evaluation = {
            "generation": getattr(ga_instance, "generations_completed", None),
            "solution": candidate_solution.tolist(),
            "frequencies": np.array(cand_freq).tolist(),
            "mode_shapes": [mode.tolist() for mode in cand_modes],
            "fitness": fitness
        }
        save_fitness_evaluation(evaluation, filepath)
    
    return fitness

# def fitness_single_objective(candidate_solution: np.ndarray,
#                            target_props: dict,
#                            beam_config: dict,
#                            ga_instance=None,
#                            weight_freq: float = 0.5,
#                            weight_mode: float = 0.5,
#                            save_results: bool = False,
#                            save_filepath: str = "fitness_results.jsonl") -> float:
    
#     # Interpret the solution
#     damage_location = int(round(candidate_solution[0]))  # Convert to integer
#     undamaged_value = candidate_solution[1]
#     damaged_value = candidate_solution[2]
    
#     # Create E_vector with undamaged values
#     E_vector = np.full(beam_config['n_elements'], undamaged_value)
    
#     # Apply damage at specified location
#     E_vector[damage_location-1] = damaged_value  # -1 because locations are 1-based
    
#     # Create beam with this damage configuration
#     candidate_beam = Beam(config=beam_config, E_vector=E_vector)
#     n_modes = len(target_props["frequencies"])
#     cand_freq, cand_modes = candidate_beam.get_modal_properties(n_eigen=n_modes)
    
#     freq_diffs = [calculate_frequency_diff(cand_freq[i], target_props["frequencies"][i])
#                   for i in range(n_modes)]
#     total_freq_diff = np.sum(freq_diffs)
    
#     mac_diffs = [1 - calculate_MAC(cand_modes[i], target_props["modes"][i])
#                  for i in range(n_modes)]
#     total_mac_diff = np.sum(mac_diffs)
    
#     total_error = weight_freq * total_freq_diff + weight_mode * total_mac_diff
#     epsilon = 1e-10
#     fitness = 1.0 / (total_error + epsilon)
    
#     if save_results and ga_instance is not None:
#         filepath = save_filepath
#         if not os.path.isabs(filepath) and hasattr(ga_instance, "run_dir"):
#             filepath = os.path.join(ga_instance.run_dir, os.path.basename(save_filepath))
#         evaluation = {
#             "generation": getattr(ga_instance, "generations_completed", None),
#             "damage_location": damage_location,
#             "undamaged_value": undamaged_value,
#             "damaged_value": damaged_value,
#             "E_vector": E_vector.tolist(),
#             "frequencies": np.array(cand_freq).tolist(),
#             "mode_shapes": [mode.tolist() for mode in cand_modes],
#             "fitness": fitness
#         }
#         save_fitness_evaluation(evaluation, filepath)
    
#     return fitness

def fitness_multi_objective(candidate_solution: np.ndarray, target_props: dict, beam_config: dict) -> (np.ndarray, dict):
    """
    Calculate several metric values for a candidate solution.
    
    This function returns an objective vector (e.g., frequency errors and MAC errors)
    along with a dictionary of detailed metric breakdowns. This can be used directly in
    a multiobjective GA or for diagnostic purposes.
    """
    candidate_beam = Beam(config=beam_config, E_vector=candidate_solution)
    n_modes = len(target_props["frequencies"])
    cand_freq, cand_modes = candidate_beam.get_modal_properties(n_eigen=n_modes)
    
    freq_errors = np.array([calculate_frequency_diff(cand_freq[i], target_props["frequencies"][i]) for i in range(n_modes)])
    mac_errors = np.array([1 - calculate_MAC(cand_modes[i], target_props["modes"][i]) for i in range(n_modes)])
    
    # For instance, you can combine errors (or return them separately)
    overall_error = np.array([np.mean(freq_errors), np.mean(mac_errors)])
    
    details = {
        "freq_errors": freq_errors,
        "mac_errors": mac_errors
    }
    return overall_error, details

