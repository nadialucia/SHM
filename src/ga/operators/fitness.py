import numpy as np
from typing import Dict, Any, Tuple, List, Callable
import json
import os
from src.beam.beam import Beam

def calculate_MAC(phi_A: np.ndarray, phi_E: np.ndarray) -> float:
    """Calculate Modal Assurance Criterion between two mode shapes"""
    numerator = np.abs(np.dot(phi_A, phi_E)) ** 2
    denominator = np.dot(phi_A, phi_A) * np.dot(phi_E, phi_E)
    return numerator / denominator

def calculate_MAC_matrix(phi_A: np.ndarray, phi_E: np.ndarray) -> float:
    """Calculate MAC matrix using matrix operations
    
    Parameters:
    -----------
    phi_A : np.ndarray
        Matrix of candidate mode shapes (n_points × n_modes)
    phi_E : np.ndarray
        Matrix of target mode shapes (n_points × n_modes)
    """
    
    n = len(phi_E)
    corr_matrix = np.corrcoef(phi_A, phi_E)[0:n,n:]
    mac_matrix = corr_matrix**2
    
    # Calculate metrics
    identity = np.eye(n)
    
    # Calculate off-diagonal asymmetry
    # Get upper and lower triangular matrices
    upper_tri = np.triu(corr_matrix, k=1)
    lower_tri = np.tril(corr_matrix, k=-1)
    
    # Calculate diagonal error
    diagonal_error = np.mean(1 - np.diag(mac_matrix))

    # Calculate absolute differences between corresponding elements
    off_diag_diff = np.mean(np.abs(upper_tri - lower_tri.T))
    
    # Print for debugging
    # print("Correlation Matrix:\n", corr_matrix)
    # print("MAC matrix:\n", mac_matrix)
    # print("Diagonal error terms:\n", 1 - np.diag(mac_matrix))
    # print("Off-diagonal differences:\n", np.abs(upper_tri - lower_tri.T))

    return diagonal_error + off_diag_diff

def calculate_mode_diff(phi_A: np.ndarray, phi_E: np.ndarray, normalized: bool = True) -> float:
    """Calculate direct difference between mode shapes
    
    Parameters:
    -----------
    phi_A : np.ndarray
        First mode shape vector
    phi_E : np.ndarray
        Second mode shape vector
    normalized : bool
        If True, normalize by the mean amplitude of target mode shape
        
    Returns:
    --------
    float
        Mean absolute difference between mode shapes
    """
    diff = np.abs(phi_A - phi_E)
    if normalized:
        return np.mean(diff) / (np.mean(np.abs(phi_E)) + 1e-10)
    return np.mean(diff)

def calculate_frequency_diff(f_A: float, f_E: float, normalized: bool = True) -> float:
    """Calculate frequency difference, optionally normalized"""
    if normalized:
        return abs(f_A - f_E) / f_E
    return abs(f_A - f_E)

# class StandardFitness:
#     """
#     Standard single-objective fitness using weighted sum of frequency and MAC differences.
    
#     Config parameters:
#         weight_freq: Weight for frequency differences (default: 0.5)
#         weight_mode: Weight for mode shape differences (default: 0.5)
#         num_modes: Number of modes to consider (default: 4)
#         save_results: Whether to save detailed results (default: False)
#         save_filepath: Where to save results (default: "fitness_results.jsonl")
#     """
#     def __init__(self, target_beam: Beam, beam_config: Dict[str, Any], config: Dict[str, Any]):
#         self.target_beam = target_beam
#         self.beam_config = beam_config
#         self.weight_freq = config.get("weight_freq", 0.5)
#         self.weight_mode = config.get("weight_mode", 0.5)
#         self.num_modes = config.get("num_modes", 4)
#         self.noise_stdev_factor = config.get("noise_stdev_factor", 0.0)

#         self.save_results = config.get("save_results", False)
#         self.save_filepath = config.get("save_filepath", "fitness_results.jsonl")
        
#         # Cache target properties
#         self.target_freqs, self.target_modes = self.target_beam.get_modal_properties(
#             n_eigen=self.num_modes
#         )

#         if self.noise_stdev_factor != 0.0:
#                 # Calculate mean absolute amplitude for scaling
#             mean_amplitude = np.mean(np.abs(self.target_modes))
            
#             # Generate noise for each mode shape
#             for i in range(len(self.target_modes)):
#                 # Generate random noise with specified standard deviation
#                 noise = np.random.normal(
#                     loc=0, 
#                     scale=self.noise_stdev_factor * mean_amplitude, 
#                     size=len(self.target_modes[i])
#                 )
#                 # Add noise to mode shape
#                 self.target_modes[i] += noise

    
#     def __call__(self, ga_instance, solution: np.ndarray, solution_idx: int) -> float:
#         """Calculate fitness for a single solution"""
#         # Create beam with candidate solution
#         candidate_beam = Beam(config=self.beam_config, E_vector=solution)
#         cand_freqs, cand_modes = candidate_beam.get_modal_properties(n_eigen=self.num_modes)
        
#         # Calculate frequency differences
#         freq_errors = [
#             calculate_frequency_diff(cand_freqs[i], self.target_freqs[i])
#             for i in range(self.num_modes)
#         ]
#         freq_objective = np.mean(freq_errors)
        
#         # # Calculate MAC differences
#         # mac_errors = [
#         #     1 - calculate_MAC(cand_modes[i], self.target_modes[i])
#         #     for i in range(self.num_modes)
#         # ]
#         # mac_objective = np.mean(mac_errors)
#         # mode_objective=mac_objective
        
#         # calculating mac matrix sum
#         # mode_objective = calculate_MAC_matrix(cand_modes, self.target_modes)

#         # Calculate direct mode shape differences
#         mode_errors = [
#             calculate_mode_diff(cand_modes[i], self.target_modes[i]) 
#                             #   normalized=self.normalize_modes)
#             for i in range(self.num_modes)
#         ]
#         mode_objective = np.mean(mode_errors)
        
#         fitness = [np.log(1.0 / (freq_objective + 1e-10)), np.log(1.0 / (mode_objective + 1e-10))]
#         return fitness


#         # Combine objectives
#         total_error = (
#             self.weight_freq * freq_objective + 
#             self.weight_mode * mode_objective
#         )
#         fitness = 1.0 / (total_error + 1e-10)
        
#         # Save detailed results if requested
#         if self.save_results and hasattr(ga_instance, "run_dir"):
#             self._save_evaluation(
#                 ga_instance, solution, cand_freqs, cand_modes, fitness
#             )
        
#         return fitness
    
#     def _save_evaluation(self, ga_instance, solution, freqs, modes, fitness):
#         """Save detailed evaluation results"""
#         filepath = os.path.join(
#             ga_instance.run_dir, 
#             os.path.basename(self.save_filepath)
#         )
#         evaluation = {
#             "generation": ga_instance.generations_completed,
#             "solution": solution.tolist(),
#             "frequencies": np.array(freqs).tolist(),
#             "mode_shapes": [mode.tolist() for mode in modes],
#             "fitness": float(fitness)
#         }
#         with open(filepath, "a") as f:
#             json.dump(evaluation, f)
#             f.write("\n")

# class MultiObjectiveFitness:
#     """
#     Multi-objective fitness calculator for NSGA-II compatibility.
#     Returns separate objectives for frequency and mode shape matching.
    
#     Config parameters:
#         num_modes: Number of modes to consider (default: 4)
#         save_results: Whether to save detailed results (default: False)
#         save_filepath: Where to save results (default: "mo_fitness_results.jsonl")
#     """
#     def __init__(self, target_beam: Beam, config: Dict[str, Any]):
#         self.target_beam = target_beam
#         self.num_modes = config.get("num_modes", 4)
#         self.save_results = config.get("save_results", False)
#         self.save_filepath = config.get("save_filepath", "mo_fitness_results.jsonl")
        
#         # Cache target properties
#         self.target_freqs, self.target_modes = self.target_beam.get_modal_properties(
#             n_eigen=self.num_modes
#         )
    
#     def __call__(self, ga_instance, solution: np.ndarray, solution_idx: int) -> List[float]:
#         """Calculate multiple objectives for NSGA-II"""
#         candidate_beam = Beam(config=self.target_beam.config, E_vector=solution)
#         cand_freqs, cand_modes = candidate_beam.get_modal_properties(n_eigen=self.num_modes)
        
#         # Calculate frequency objective
#         freq_errors = [
#             calculate_frequency_diff(cand_freqs[i], self.target_freqs[i])
#             for i in range(self.num_modes)
#         ]
#         freq_objective = np.mean(freq_errors)
        
#         # Calculate MAC objective
#         mac_errors = [
#             1 - calculate_MAC(cand_modes[i], self.target_modes[i])
#             for i in range(self.num_modes)
#         ]
#         mac_objective = np.mean(mac_errors)
        
#         if self.save_results and hasattr(ga_instance, "run_dir"):
#             self._save_evaluation(
#                 ga_instance, solution, cand_freqs, cand_modes, 
#                 [freq_objective, mac_objective]
#             )
        
#         return [freq_objective, mac_objective]

#     def _save_evaluation(self, ga_instance, solution, freqs, modes, objectives):
#         """Save detailed evaluation results"""
#         filepath = os.path.join(
#             ga_instance.run_dir, 
#             os.path.basename(self.save_filepath)
#         )
#         evaluation = {
#             "generation": ga_instance.generations_completed,
#             "solution": solution.tolist(),
#             "frequencies": np.array(freqs).tolist(),
#             "mode_shapes": [mode.tolist() for mode in modes],
#             "objectives": objectives
#         }
#         with open(filepath, "a") as f:
#             json.dump(evaluation, f)
#             f.write("\n")


class ObjectiveFunction:
    """Base class for individual objective functions"""
    def __init__(self, name: str, use_log: bool = False, weight: float = 1.0):
        self.name = name
        self.use_log = use_log
        self.weight = weight
    
    def calculate(self, candidate_data: dict, target_data: dict) -> float:
        raise NotImplementedError
    
    def process_result(self, value: float) -> float:
        """Apply log transform if requested"""
        if self.use_log:
            return np.log(1.0 / (value + 1e-10))
        return value

class FrequencyObjective(ObjectiveFunction):
    def __init__(self, use_log: bool = False, weight: float = 1.0):
        super().__init__("frequency", use_log, weight)
    
    def calculate(self, candidate_data: dict, target_data: dict) -> float:
        freq_errors = [
            calculate_frequency_diff(
                candidate_data["frequencies"][i], 
                target_data["frequencies"][i]
            ) for i in range(len(target_data["frequencies"]))
        ]
        return np.mean(freq_errors)

class MACObjective(ObjectiveFunction):
    def __init__(self, use_log: bool = False, weight: float = 1.0):
        super().__init__("mac", use_log, weight)
    
    def calculate(self, candidate_data: dict, target_data: dict) -> float:
        mac_errors = [
            1 - calculate_MAC(
                candidate_data["modes"][i], 
                target_data["modes"][i]
            ) for i in range(len(target_data["modes"]))
        ]
        return np.mean(mac_errors)

class DirectModeObjective(ObjectiveFunction):
    def __init__(self, use_log: bool = False, weight: float = 1.0):
        super().__init__("mode_direct", use_log, weight)
    
    def calculate(self, candidate_data: dict, target_data: dict) -> float:
        mode_errors = [
            calculate_mode_diff(
                candidate_data["modes"][i], 
                target_data["modes"][i]
            ) for i in range(len(target_data["modes"]))
        ]
        return np.mean(mode_errors)

class IndividualModeObjective(ObjectiveFunction):
    """Objective function for individual mode shape matching"""
    def __init__(self, mode_index: int, use_mac: bool = True, use_log: bool = False, weight: float = 1.0):
        """
        Parameters:
        -----------
        mode_index : int
            Index of the mode shape to evaluate
        use_mac : bool
            If True, use MAC criterion, otherwise use direct mode shape difference
        use_log : bool
            Whether to apply log transform to output
        weight : float
            Weight for this objective
        """
        super().__init__(f"mode_{mode_index}", use_log, weight)
        self.mode_index = mode_index
        self.use_mac = use_mac
    
    def calculate(self, candidate_data: dict, target_data: dict) -> float:
        if self.mode_index >= len(target_data["modes"]):
            raise ValueError(f"Mode index {self.mode_index} exceeds available modes")
            
        if self.use_mac:
            error = 1 - calculate_MAC(
                candidate_data["modes"][self.mode_index],
                target_data["modes"][self.mode_index]
            )
        else:
            error = calculate_mode_diff(
                candidate_data["modes"][self.mode_index],
                target_data["modes"][self.mode_index]
            )
        return error
    

class BaseFitness:
    """Base class for fitness calculation"""
    def __init__(self, target_beam: Beam, beam_config: Dict[str, Any], config: Dict[str, Any]):
        self.target_beam = target_beam
        self.beam_config = beam_config
        self.num_modes = config.get("num_modes", 4)
        self.save_results = config.get("save_results", False)
        self.save_filepath = config.get("save_filepath", "fitness_results.jsonl")
        self.noise_stdev_factor = config.get("noise_stddev_factor", 0.0)
        # print(self.noise_stdev_factor)
        
        
        # Cache target properties
        self.target_freqs, self.target_modes = self.target_beam.get_modal_properties(
            n_eigen=self.num_modes
        )

        # Apply noise if specified
        if self.noise_stdev_factor != 0.0:
            self._apply_noise_to_target_modes()
            
        self.target_data = {
            "frequencies": self.target_freqs,
            "modes": self.target_modes
        }
        
        # Setup objectives based on config
        self.objectives = self._setup_objectives(config)

    def _apply_noise_to_target_modes(self):
        """Apply random noise to target mode shapes"""
        # Calculate mean absolute amplitude for scaling
        mean_amplitude = np.mean(np.abs(self.target_modes))
        
        # Generate and apply noise for each mode shape
        for i in range(len(self.target_modes)):
            noise = np.random.normal(
                loc=0, 
                scale=self.noise_stdev_factor * mean_amplitude, 
                size=len(self.target_modes[i])
            )
            self.target_modes[i] += noise
    

    def _setup_objectives(self, config: Dict[str, Any]) -> List[ObjectiveFunction]:
        """Setup objective functions based on configuration"""
        objective_map = {
            "frequency": FrequencyObjective,
            "mac": MACObjective,
            "mode_direct": DirectModeObjective,
            "individual_mode": IndividualModeObjective
        }
        
        objectives = []
        for obj_config in config.get("objectives", []):
            obj_type = obj_config["type"]
            if obj_type in objective_map:
                if obj_type == "individual_mode":
                    objectives.append(
                        objective_map[obj_type](
                            mode_index=obj_config["mode_index"],
                            use_mac=obj_config.get("use_mac", False),
                            use_log=obj_config.get("use_log", False),
                            weight=obj_config.get("weight", 1.0)
                        )
                    )
                else:
                    objectives.append(
                        objective_map[obj_type](
                            use_log=obj_config.get("use_log", False),
                            weight=obj_config.get("weight", 1.0)
                        )
                    )
        return objectives
    
    def _get_candidate_data(self, solution: np.ndarray) -> dict:
        """Get modal properties for candidate solution"""
        candidate_beam = Beam(config=self.beam_config, E_vector=solution)
        freqs, modes = candidate_beam.get_modal_properties(n_eigen=self.num_modes)
        return {"frequencies": freqs, "modes": modes}

    def _save_evaluation(self, ga_instance, solution, cand_data, fitness):
        """Save detailed evaluation results"""
        if not self.save_results or not hasattr(ga_instance, "run_dir"):
            return
            
        filepath = os.path.join(ga_instance.run_dir, os.path.basename(self.save_filepath))
        evaluation = {
            "generation": ga_instance.generations_completed,
            "target_beam": self.target_beam.E_vector.tolist(),
            "solution": solution.tolist(),
            "frequencies": np.array(cand_data["frequencies"]).tolist(),
            "mode_shapes": [mode.tolist() for mode in cand_data["modes"]],
            "fitness": fitness
        }
        with open(filepath, "a") as f:
            json.dump(evaluation, f)
            f.write("\n")

class SingleObjectiveFitness(BaseFitness):
    def __call__(self, ga_instance, solution: np.ndarray, solution_idx: int) -> float:
        cand_data = self._get_candidate_data(solution)
        
        # Calculate weighted sum of objectives
        total_fitness = 0
        for objective in self.objectives:
            value = objective.calculate(cand_data, self.target_data)
            # print("value:", value)
            processed_value = objective.process_result(value)
            # print("processed_value:", processed_value)
            total_fitness += objective.weight * processed_value
            # if processed_value > 6:
            
            #     print(objective.weight, processed_value)
            
        # print("obj", total_fitness)
        self._save_evaluation(ga_instance, solution, cand_data, total_fitness)
        return total_fitness

class MultiObjectiveFitness(BaseFitness):
    def __call__(self, ga_instance, solution: np.ndarray, solution_idx: int) -> List[float]:
        cand_data = self._get_candidate_data(solution)
        
        # Calculate separate objectives
        fitness_values = []
        for objective in self.objectives:
            value = objective.calculate(cand_data, self.target_data)
            processed_value = objective.process_result(value)
            fitness_values.append(processed_value)
            
        self._save_evaluation(ga_instance, solution, cand_data, fitness_values)
        return fitness_values

def get_fitness_function(name: str, target_beam: Beam, beam_config: Dict[str, Any], config: Dict[str, Any]) -> Callable:
    """Factory function to return appropriate fitness function"""
    fitness_classes = {
        "single_objective": SingleObjectiveFitness,
        "multi_objective": MultiObjectiveFitness
    }
    
    if name not in fitness_classes:
        raise ValueError(f"Unknown fitness type: {name}. "
                       f"Available options: {list(fitness_classes.keys())}")
    
    fitness_instance = fitness_classes[name](target_beam, beam_config, config)
    return fitness_instance.__call__