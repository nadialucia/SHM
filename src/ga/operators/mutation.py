from typing import Dict, Any, Union, Callable
import numpy as np
from pygad.utils.mutation import Mutation  # Ensure this is imported if not already


def print_offspring_stats(offspring: np.ndarray, label: str = ""):
    """
    Print statistics about the offspring array.
    
    Args:
        offspring: numpy array of shape (n_solutions, n_genes)
        label: optional string to identify the print statement
    """
    # min_val = np.min(offspring)
    # max_val = np.max(offspring)
    # mean_val = np.mean(offspring)
    
    # if label:
    #     print(f"\n{label}:")
    # print(f"Min value: {min_val:.2e}")
    # print(f"Max value: {max_val:.2e}")
    # print(f"Mean value: {mean_val:.2e}")

    pass


class RandomMutation:
    """Random mutation operator"""
    def __init__(self, config: Dict[str, Any]):
        self.probability = config.get("probability", 0.1)
        self.mutation_percent_genes = config.get("mutation_percent_genes", 10)
    
    def mutate(self, offspring: np.ndarray, ga_instance: Any) -> np.ndarray:
        """
        Implement random mutation
        
        Args:
            offspring: Array of solutions to mutate
            ga_instance: Instance of pygad.GA
            
        Returns:
            offspring: Mutated solutions
        """
        for idx in range(offspring.shape[0]):
            mutation_genes = np.random.choice(offspring.shape[1], 
                                           size=int(self.mutation_percent_genes * offspring.shape[1] / 100),
                                           replace=False)
            
            for gene_idx in mutation_genes:
                if np.random.random() < self.probability:
                    # Random value between mutation bounds
                    offspring[idx, gene_idx] = np.random.uniform(
                        ga_instance.random_mutation_min_val,
                        ga_instance.random_mutation_max_val
                    )
        
        return offspring


class SpilloverMutation:
    """Spillover mutation that passes a portion of gene value to next/previous gene"""
    def __init__(self, config: Dict[str, Any]):
        self.probability = config.get("probability", 0.1)
        self.spillover_rate = config.get("spillover_rate", 0.2)
        self.forward_prob = config.get("forward_prob", 0.5)
        self.variation_prob = config.get("variation_prob", 0.3)  # Probability of adding variation
        self.variation_range = config.get("variation_range", 0.1)  # Range for random variation
        
        assert 0 <= self.spillover_rate <= 1, "Spillover rate must be between 0 and 1"
        assert 0 <= self.forward_prob <= 1, "Forward probability must be between 0 and 1"
        assert 0 <= self.variation_prob <= 1, "Variation probability must be between 0 and 1"
        assert self.variation_range > 0, "Variation range must be positive"
    
    def mutate(self, offspring: np.ndarray, ga_instance: Any) -> np.ndarray:
        """
        Implement spillover mutation with random variation
        """
        for idx in range(offspring.shape[0]):
            if np.random.random() < self.probability:
                # Phase 1: Select gene pairs for spillover
                spillover_pairs = []
                valid_genes = np.arange(offspring.shape[1])
                
                # For each gene, decide if it will spill over
                for gene_idx in valid_genes:
                    if np.random.random() < self.probability:
                        spill_forward = np.random.random() < self.forward_prob
                        if spill_forward and gene_idx < offspring.shape[1] - 1:
                            spillover_pairs.append((gene_idx, gene_idx + 1))
                        elif not spill_forward and gene_idx > 0:
                            spillover_pairs.append((gene_idx, gene_idx - 1))
                
                # Phase 2: Calculate spillover amounts with variation
                spillover_changes = {}
                for source, target in spillover_pairs:
                    base_spillover = offspring[idx, source] * self.spillover_rate
                    
                    # Add random variation if selected
                    if np.random.random() < self.variation_prob:
                        variation = np.random.uniform(-self.variation_range, self.variation_range) * base_spillover
                        spillover_amount = base_spillover + variation
                    else:
                        spillover_amount = base_spillover
                    
                    # Store changes
                    spillover_changes[source] = spillover_changes.get(source, 0) - base_spillover
                    spillover_changes[target] = spillover_changes.get(target, 0) + spillover_amount
                
                # Phase 3: Apply all spillovers at once
                for gene_idx, change in spillover_changes.items():
                    offspring[idx, gene_idx] += change
        
        # Apply bounds if defined in ga_instance
        if hasattr(ga_instance, 'gene_space') and isinstance(ga_instance.gene_space, dict):
            low = ga_instance.gene_space.get('low', float('-inf'))
            high = ga_instance.gene_space.get('high', float('inf'))
            np.clip(offspring, low, high, out=offspring)

        print_offspring_stats(offspring, "After Spillover mutation")
        
        return offspring
    

class DistancingMutation:
    """
    Distancing Mutation:
    For each individual (if selected by probability), scales the deviation from the gene-wise mean.
    For a gene value x with column mean μ, the new value is:
        new_x = μ + factor * (x - μ)
    where 'factor' is sampled uniformly between 'distancing_min_factor' and 'distancing_max_factor'.
    """
    def __init__(self, config: dict):
        self.probability = config.get("probability", 0.1)
        self.distancing_min_factor = config.get("distancing_min_factor", 0.0)
        self.distancing_max_factor = config.get("distancing_max_factor", 2.0)
    
    def mutate(self, offspring: np.ndarray, ga_instance: any) -> np.ndarray:
        # Make a copy to avoid modifying the original array
        mutated = offspring.copy()
        
        # Compute per-gene means across the current offspring
        means = np.mean(mutated, axis=0)
        
        for idx in range(mutated.shape[0]):
            if np.random.random() < self.probability:
                # Apply to all genes of the selected individual
                for j in range(mutated.shape[1]):
                    factor = np.random.uniform(self.distancing_min_factor, self.distancing_max_factor)
                    deviation = mutated[idx, j] - means[j]
                    mutated[idx, j] = means[j] + factor * deviation
        
        # Apply bounds if defined in ga_instance
        if hasattr(ga_instance, 'gene_space') and isinstance(ga_instance.gene_space, dict):
            low = ga_instance.gene_space.get('low', float('-inf'))
            high = ga_instance.gene_space.get('high', float('inf'))
            np.clip(mutated, low, high, out=mutated)

        print_offspring_stats(mutated, "After Distancing mutation")
        
        return mutated


class CreepMutation:
    """
    Creep Mutation:
    For each gene, with a given probability, a random value is generated between
    'mutation_min_val' and 'mutation_max_val'. Then, based on the 'addition_probability',
    it is either added or subtracted from the gene value.
    """
    def __init__(self, config: dict):
        self.probability = config.get("probability", 0.1)
        self.addition_probability = config.get("addition_probability", 0.5)
        self.mutation_min_val = config.get("mutation_min_val", 0.0)
        self.mutation_max_val = config.get("mutation_max_val", 1.0)
    
    def mutate(self, offspring: np.ndarray, ga_instance: any) -> np.ndarray:
        # Make a copy to avoid modifying the original array
        mutated = offspring.copy()
        
        # Get gene space bounds
        if hasattr(ga_instance, 'gene_space'):
            if isinstance(ga_instance.gene_space, dict):
                low = ga_instance.gene_space.get('low', float('-inf'))
                high = ga_instance.gene_space.get('high', float('inf'))
            else:
                low, high = float('-inf'), float('inf')
        else:
            low, high = float('-inf'), float('inf')
        
        for idx in range(mutated.shape[0]):
            for j in range(mutated.shape[1]):
                if np.random.random() < self.probability:
                    # Generate random creep value
                    creep_val = np.random.uniform(self.mutation_min_val, self.mutation_max_val)
                    
                    # Calculate proposed new value
                    if np.random.random() < self.addition_probability:
                        new_val = mutated[idx, j] + creep_val
                    else:
                        new_val = mutated[idx, j] - creep_val
                    
                    # Only apply mutation if it keeps value within bounds
                    if low <= new_val <= high:
                        mutated[idx, j] = new_val
        
        return mutated


class DirectionalCreepMutation:
    """
    Directional Creep Mutation:
    For each solution, picks a general direction (increase/decrease) and then applies
    creep mutation in that direction with some variance. This helps maintain
    directional consistency in mutation steps for each solution.
    
    Config parameters:
      probability: Probability to apply mutation on each gene (default: 0.1)
      direction_bias: Likelihood of maintaining chosen direction (default: 0.8)
      mutation_min_val: Minimum creep value (default: 0.0)
      mutation_max_val: Maximum creep value (default: 1.0)
      variance_factor: How much individual genes can deviate from direction (default: 0.3)
    """
    def __init__(self, config: dict):
        self.probability = config.get("probability", 0.1)
        self.direction_bias = config.get("direction_bias", 0.8)
        self.mutation_min_val = config.get("mutation_min_val", 0.0)
        self.mutation_max_val = config.get("mutation_max_val", 1.0)
        #self.variance_factor = config.get("variance_factor", 0.3)
        
    def mutate(self, offspring: np.ndarray, ga_instance: any) -> np.ndarray:
        mutated = offspring.copy()
        n_solutions, n_genes = mutated.shape
        
        # Generate mutation mask
        mutation_mask = np.random.random((n_solutions, n_genes)) < self.probability
        
        # Choose primary direction for each solution
        solution_directions = np.random.choice([-1, 1], size=n_solutions)
        
        # Generate gene-specific directions with bias towards solution direction
        gene_directions = np.where(
            np.random.random((n_solutions, n_genes)) < self.direction_bias,
            solution_directions[:, np.newaxis],  # Main direction
            -solution_directions[:, np.newaxis]   # Opposite direction
        )
        
        # Generate random creep values
        creep_values = np.random.uniform(
            self.mutation_min_val,
            self.mutation_max_val,
            size=(n_solutions, n_genes)
        )
        
        # Apply directional creep where mutation mask is True
        mutated += mutation_mask * gene_directions * creep_values
        
        # Apply bounds if defined in ga_instance
        if hasattr(ga_instance, 'gene_space') and isinstance(ga_instance.gene_space, dict):
            low = ga_instance.gene_space.get('low', float('-inf'))
            high = ga_instance.gene_space.get('high', float('inf'))
            np.clip(mutated, low, high, out=mutated)

        print_offspring_stats(mutated, "After Directional mutation")
            
        return mutated


class OptimizedCreepMutation:
    """
    Optimized Creep Mutation:
    Vectorized implementation of creep mutation using numpy operations instead of loops.
    For each gene, with a given probability, a random value is generated between
    'mutation_min_val' and 'mutation_max_val' and either added or subtracted.
    
    Config parameters:
      probability: Probability to apply mutation on each gene (default: 0.1)
      addition_probability: Likelihood of adding the value (default: 0.5)
      mutation_min_val: Minimum creep value (default: 0.0)
      mutation_max_val: Maximum creep value (default: 1.0)
    """
    def __init__(self, config: dict):
        self.probability = config.get("probability", 0.1)
        self.addition_probability = config.get("addition_probability", 0.5)
        self.mutation_min_val = config.get("mutation_min_val", 0.0)
        self.mutation_max_val = config.get("mutation_max_val", 1.0)
    
    def mutate(self, offspring: np.ndarray, ga_instance: any) -> np.ndarray:
        mutated = offspring.copy()
        n_solutions, n_genes = mutated.shape
        
        # Generate mutation mask
        mutation_mask = np.random.random((n_solutions, n_genes)) < self.probability
        
        # Generate random creep values
        creep_values = np.random.uniform(
            self.mutation_min_val,
            self.mutation_max_val,
            size=(n_solutions, n_genes)
        )
        
        # Generate addition/subtraction mask
        add_mask = np.random.random((n_solutions, n_genes)) < self.addition_probability
        
        # Calculate changes (positive for addition, negative for subtraction)
        changes = creep_values * (2 * add_mask - 1)
        
        # Apply changes where mutation mask is True
        mutated += mutation_mask * changes
        
        # Apply bounds if defined in ga_instance
        if hasattr(ga_instance, 'gene_space') and isinstance(ga_instance.gene_space, dict):
            low = ga_instance.gene_space.get('low', float('-inf'))
            high = ga_instance.gene_space.get('high', float('inf'))
            np.clip(mutated, low, high, out=mutated)
            
        print_offspring_stats(mutated, "After Optimized mutation")

        return mutated


class CompositeMutation:
    """Applies multiple custom mutation operators in sequence"""
    def __init__(self, config: Dict[str, Any]):
        self.mutation_configs = config.get("mutations", [])
        self.mutations = []
        
        # Initialize all mutation operators
        for mut_config in self.mutation_configs.copy():  # Use copy to avoid modifying original
            mut_type = mut_config["type"]  # Don't pop, we might need config multiple times
            mutation = get_mutation_operator(mut_type, mut_config)
            self.mutations.append(mutation)
    
    def mutate(self, offspring: np.ndarray, ga_instance: Any) -> np.ndarray:
        """Apply each mutation operator in sequence"""
        mutated = offspring.copy()  # Work on a copy to avoid modifying original
        for mutation in self.mutations:
            if callable(mutation):
                mutated = mutation(mutated, ga_instance)
            else:
                # Handle built-in PyGAD mutation types
                ga_instance.mutation_type = mutation
                mutated = ga_instance.mutation(mutated)
        return mutated


class RandomSelectorMutation:
    """
    A mutation operator that selects one from a list of mutation operators randomly each generation,
    with optional selection probabilities.
    
    The configuration expects a key "mutations" that is a list of mutation configurations.
    Each individual mutation config should have at least a "type" field and optionally
    a "selection_probability" field.
    """
    def __init__(self, config: Dict[str, Any]):
        self.mutation_configs = config.get("mutations", [])
        self.mutations = []  # list of tuples: (mutation_callable, selection_probability)
        
        # Build the list of mutation operators
        for mut_config in self.mutation_configs:
            probability = mut_config.get("selection_probability", 1.0)
            mutation_callable = get_mutation_operator(mut_config["type"], mut_config)
            self.mutations.append((mutation_callable, probability))
        
        # Normalize the selection probabilities
        total = sum(prob for (_, prob) in self.mutations)
        if total == 0:
            # Avoid division by zero: assign equal probabilities if total is zero.
            self.weights = [1/len(self.mutations)] * len(self.mutations)
        else:
            self.weights = [prob/total for (_, prob) in self.mutations]
    
    def mutate(self, offspring: np.ndarray, ga_instance: Any) -> np.ndarray:
        """
        Randomly selects one mutation operator (based on its selection probability)
        and applies it to the offspring.
        """
        # Select index based on the normalized weights
        selected_index = np.random.choice(len(self.mutations), p=self.weights)
        mutation_func = self.mutations[selected_index][0]
        # Directly call the stored callable with offspring and ga_instance
        return mutation_func(offspring, ga_instance)


class PyGADMutationWrapper:
    """
    Wraps a PyGAD mutation function so that it can be called as a standalone mutation operator.
    It temporarily assigns the required attributes to the ga_instance.
    """
    def __init__(self, mutation_func: Callable, config: Dict[str, Any]):
        self.mutation_func = mutation_func
        self.probability = config.get("probability", 0.1)
        self.mutation_percent_genes = config.get("mutation_percent_genes", 10)
    
    def mutate(self, offspring: np.ndarray, ga_instance: Any) -> np.ndarray:
        # Save original attributes (if any)
        orig_prob = getattr(ga_instance, "mutation_probability", None)
        orig_percent = getattr(ga_instance, "mutation_percent_genes", None)
        
        # Set attributes required by the PyGAD mutation function
        ga_instance.mutation_probability = self.probability
        ga_instance.mutation_percent_genes = self.mutation_percent_genes
        
        # Call the PyGAD mutation function
        mutated = self.mutation_func(ga_instance, offspring)
        
        # Restore original values
        if orig_prob is not None:
            ga_instance.mutation_probability = orig_prob
        if orig_percent is not None:
            ga_instance.mutation_percent_genes = orig_percent
        
        return mutated
 

def get_mutation_operator(name: str, config: Dict[str, Any]) -> Callable:
    """Factory function for mutation operators.
    
    For built-in operator names (like 'random', 'swap', etc.) this wraps the pygad function into a callable.
    For custom ones, it returns the appropriate mutation callable.
    """
    
    builtin_operators = {
        "random": Mutation.random_mutation,
        "swap": Mutation.swap_mutation,
        "inversion": Mutation.inversion_mutation,
        "scramble": Mutation.scramble_mutation,
        "adaptive": Mutation.adaptive_mutation
    }
    
    if name in builtin_operators:
        # Wrap the pygad mutation function so that it is callable
        wrapper = PyGADMutationWrapper(builtin_operators[name], config)
        return wrapper.mutate
    elif name == "random_custom":
        mutation_instance = RandomMutation(config)
        return mutation_instance.mutate
    elif name == "spillover":
        mutation_instance = SpilloverMutation(config)
        return mutation_instance.mutate
    elif name == "composite":
        mutation_instance = CompositeMutation(config)
        return mutation_instance.mutate
    elif name == "random_selector":
        mutation_instance = RandomSelectorMutation(config)
        return mutation_instance.mutate
    elif name == "creep":
        mutation_instance = CreepMutation(config)
        return mutation_instance.mutate
    elif name == "distancing":
        mutation_instance = DistancingMutation(config)
        return mutation_instance.mutate
    elif name == "directional_creep":
        mutation_instance = DirectionalCreepMutation(config)
        return mutation_instance.mutate
    elif name == "optimized_creep":
        mutation_instance = OptimizedCreepMutation(config)
        return mutation_instance.mutate
    else:
        raise ValueError(f"Unknown mutation operator: {name}")
    