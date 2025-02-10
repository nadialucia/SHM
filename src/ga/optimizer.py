from pathlib import Path
import pygad
import yaml
from tqdm import tqdm
from datetime import datetime
from .operators.fitness import get_fitness_function
from .operators.crossover import get_crossover_operator
from .operators.mutation import get_mutation_operator
from .operators.selection import get_selection_operator

class GAOptimizer:
    def __init__(self, config: dict = None, config_path: Path = None, saved_run_path: Path = None):
        """Initialize optimizer either with new config or from saved run"""
        if saved_run_path:
            self._load_from_saved(saved_run_path)
        elif config is not None:
            self.config = config
            self._setup_output_dir()
        elif config_path is not None:
            self.config = self._load_config(config_path)
            self._setup_output_dir()
        else:
            raise ValueError("Provide a config dict, config_path, or saved_run_path")

    def _load_config(self, config_path: Path) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_output_dir(self):
        """Setup output directory structure"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(self.config.get("output_directory", "results")) / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config file
        with open(self.run_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)
       
    def save(self):
        """Save current state of optimization"""
        if not hasattr(self, 'ga_instance'):
            raise RuntimeError("No GA instance to save - run optimize() first")

        # Remove non-picklable attributes (e.g. callbacks) before saving
        if hasattr(self.ga_instance, "on_generation"):
            self.ga_instance.on_generation = None

        # Save GA instance
        self.ga_instance.save(filename=str(self.run_dir / "ga_instance"))
            
        return self.run_dir
    
    def _load_from_saved(self, saved_run_path: Path):
        """Load optimizer state from saved run"""
        # Load config
        config_path = saved_run_path / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"No config file found in {saved_run_path}")
            
        self.config = self._load_config(config_path)
        self.run_dir = saved_run_path
        
        # Load GA instance if exists
        ga_path = saved_run_path / "ga_instance"
        if ga_path.with_suffix('.pkl').exists():
            import pygad
            self.ga_instance = pygad.load(str(ga_path))
    
    @classmethod
    def from_saved(cls, run_path: Path):
        """Create optimizer instance from saved run"""
        return cls(saved_run_path=run_path)

    def _setup_ga(self):
        """Initialize GA with configuration"""
        # Get base config
        ga_config = self.config["genetic_algorithm"].copy()
        
        # Store original number of generations for continuing runs
        self.original_num_generations = ga_config["num_generations"]
        
        # Remove operator settings from ga_config
        operator_keys = ["crossover_type", "mutation_type", "parent_selection_type"]
        for key in operator_keys:
            ga_config.pop(key, None)
        
        self.ga_instance = pygad.GA(
            **ga_config,
            fitness_func=self.fitness_func,
            crossover_type=self.crossover_op,
            mutation_type=self.mutation_op,
            parent_selection_type=self.selection_op
        )
        
        # Attach the optimizer's run_dir to the GA instance so fitness functions know where to save
        self.ga_instance.run_dir = str(self.run_dir)

    def continue_optimization(self, additional_generations):
        """Continue optimization from current state"""
        if not hasattr(self, 'ga_instance'):
            raise RuntimeError("No GA instance to continue - run optimize() first")
        
        print(f"\nContinuing optimization for {additional_generations} more generations")
        print(f"Previous best fitness: {self.ga_instance.best_solution()[1]:.6f}")
        
        # Update number of generations
        #self.ga_instance.num_generations += additional_generations
        
        # Run additional generations
        self.ga_instance.run()
        
        return self._process_results()

    def _setup_fitness(self):
        fitness_config = self.config["optimization"]["fitness"]
        return get_fitness_function(
            name=fitness_config["type"],
            config=fitness_config.get("parameters", {})
        )
    
    def _setup_operators(self):
        """Setup all GA operators"""
        op_config = self.config["operators"]
        
        self.crossover_op = get_crossover_operator(
            op_config["crossover"]["type"],
            op_config["crossover"]
        )
        
        self.mutation_op = get_mutation_operator(
            op_config["mutation"]["type"],
            op_config["mutation"]
        )
        
        self.selection_op = get_selection_operator(
            op_config["selection"]["type"],
            op_config["selection"]
        )
        
    # Modify the optimize() method in GAOptimizer, replacing its current body with the following:
    def optimize(self, show_progress=True):
        """Run optimization"""
        self._setup_operators() # removing for now
        self._setup_ga()
        
        if show_progress:
            num_generations = self.config["genetic_algorithm"].get("num_generations", 100)
            progress_bar = tqdm(total=num_generations, desc="GA Generations", ncols=100)
            
            def on_generation(ga_instance):
                # Update progress bar with current best fitness
                best_solution = ga_instance.best_solution()
                if best_solution is not None:
                    best_fitness = best_solution[1]
                    progress_bar.set_description(
                        f"Gen {ga_instance.generations_completed}/{num_generations} | Best: {best_fitness:.3f}"
                    )
                progress_bar.update(1)
            
            self.ga_instance.on_generation = on_generation

        self.ga_instance.run()

        if show_progress:
            progress_bar.close()
            
        return self._process_results()
        
    def _generation_callback(self):
        """Save generation results"""
        pass

    def _process_results(self):
        """Process and save final results"""
        print("finished optimising")