from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, Union

class SelectionOperator(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tournament_size = config.get("tournament_size", 3)
        
    @abstractmethod
    def __call__(self, fitness, num_parents, ga_instance) -> Tuple[np.ndarray, np.ndarray]:
        pass

class TournamentSelection(SelectionOperator):
    """PyGAD's tournament selection"""
    def __call__(self, fitness, num_parents, ga_instance) -> str:
        return "tournament"

class CustomRankBasedSelection(SelectionOperator):
    """Custom rank-based selection with diversity preservation"""
    def __call__(self, fitness, num_parents, ga_instance) -> Tuple[np.ndarray, np.ndarray]:
        # Rank-based selection with diversity measure
        ranked_indices = np.argsort(fitness)[::-1]
        
        # Calculate diversity scores
        population = ga_instance.population
        diversity_scores = np.array([
            np.mean([np.linalg.norm(sol - other) 
                    for other in population])
            for sol in population
        ])
        
        # Combine fitness and diversity
        combined_scores = 0.7 * fitness + 0.3 * diversity_scores
        selected_indices = np.argsort(combined_scores)[-num_parents:]
        
        return population[selected_indices], selected_indices

def get_selection_operator(name: str, config: Dict[str, Any]) -> Union[str, SelectionOperator]:
    """Factory function for selection operators"""
    builtin_operators = {
        "sss": "sss",
        "rws": "rws",
        "sus": "sus",
        "rank": "rank",
        "random": "random",
        "tournament": "tournament"
    }
    
    custom_operators = {
        "rank_diversity": CustomRankBasedSelection
    }
    
    if name in builtin_operators:
        return builtin_operators[name]
    elif name in custom_operators:
        return custom_operators[name](config)
    else:
        raise ValueError(f"Unknown selection: {name}")