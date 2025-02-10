from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Union, Callable

class SinglePointCrossover:
    """Custom single point crossover operator"""
    def __init__(self, config: Dict[str, Any]):
        self.probability = config.get("probability", 0.8)
    
    def crossover(self, parents: np.ndarray, offspring_size: tuple, ga_instance: Any) -> np.ndarray:
        """
        Implement single point crossover
        
        Args:
            parents: Array of selected parents
            offspring_size: Tuple of (n_offspring, n_genes)
            ga_instance: Instance of pygad.GA
            
        Returns:
            offspring: Array of new solutions
        """
        offspring = np.empty(offspring_size)
        
        # Iterate over parents to create offspring
        for k in range(offspring_size[0]):
            # Get parent indices
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k+1) % parents.shape[0]
            
            # Get parents
            parent1 = parents[parent1_idx].copy()
            parent2 = parents[parent2_idx].copy()
            
            if np.random.random() < self.probability:
                # Generate crossover point
                crossover_point = np.random.randint(0, offspring_size[1])
                
                # Create offspring
                offspring[k, 0:crossover_point] = parent1[0:crossover_point]
                offspring[k, crossover_point:] = parent2[crossover_point:]
            else:
                # No crossover - copy parent1
                offspring[k] = parent1

        return offspring

class BlendCrossover:
    """Blend crossover operator that samples uniformly between parent values"""
    def __init__(self, config: Dict[str, Any]):
        self.probability = config.get("probability", 0.8)
    
    def crossover(self, parents: np.ndarray, offspring_size: tuple, ga_instance: Any) -> np.ndarray:
        """
        Implement blend crossover by sampling uniformly between parent values
        
        Args:
            parents: Array of selected parents
            offspring_size: Tuple of (n_offspring, n_genes)
            ga_instance: Instance of pygad.GA
            
        Returns:
            offspring: Array of new solutions
        """
        offspring = np.empty(offspring_size)
        
        for k in range(offspring_size[0]):
            # Get parent indices
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k+1) % parents.shape[0]
            
            # Get parents
            parent1 = parents[parent1_idx].copy()
            parent2 = parents[parent2_idx].copy()
            
            if np.random.random() < self.probability:
                # Generate random weights for blending
                alpha = np.random.random(offspring_size[1])
                
                # Blend parents using element-wise operations
                offspring[k] = alpha * parent1 + (1 - alpha) * parent2
            else:
                # No crossover - copy parent1
                offspring[k] = parent1

        return offspring

# Update the get_crossover_operator function to include the new operator
def get_crossover_operator(name: str, config: Dict[str, Any]) -> Union[str, Callable]:
    """Factory function for crossover operators"""
    builtin_operators = {
        "single_point": "single_point",
        "two_points": "two_points",
        "uniform": "uniform",
        "scattered": "scattered"
    }
    
    if name in builtin_operators:
        return builtin_operators[name]
    elif name == "single_point_custom":
        crossover_instance = SinglePointCrossover(config)
        return crossover_instance.crossover
    elif name == "blend":
        crossover_instance = BlendCrossover(config)
        return crossover_instance.crossover
    else:
        raise ValueError(f"Unknown crossover: {name}")