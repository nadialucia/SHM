import yaml
import numpy as np
from pathlib import Path

class Beam:
    def __init__(self, config=None, E_vector=None):
        """Initialize beam with config file and/or E vector
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file. If None, default properties will be used.
        E_vector : array-like, optional
            Vector of elastic moduli for each segment. If None, uniform E vector will be created.
        """
        # Set default properties first
        self._set_default_properties()
        
        if config is not None:
            self._load_config(config)
        
        # Override E_vector if provided
        if E_vector is not None:
            self.E_vector = np.array(E_vector)
            self.n_segments = len(self.E_vector)
            self.E = np.mean(self.E_vector)  # Update reference E

    def _set_default_properties(self):
        """Set default properties"""
        # Default geometry
        self.length = 2.0
        self.width = 0.3
        self.height = 0.1
        
        # Default material properties
        self.E = 30e9  # Default elastic modulus (30 GPa)
        self.density = 2300.0
        self.poisson = 0.3
        
        # Default mesh properties
        self.n_segments = 10
        self.n_elements = 100
        self.n_mp = 21
        
        # Initialize default uniform E vector
        self.E_vector = np.array([self.E] * self.n_segments)

    def _load_config(self, config):
        """Load beam properties from config"""
            
        # Geometry
        self.width = config['geometry']['width']
        self.height = config['geometry']['height']
        self.length = config['geometry']['length']
        
        # Mesh
        self.n_segments = config['mesh']['n_p']
        self.n_elements = config['mesh']['n_e']
        self.n_mp = config['mesh']['n_mp']
        
        # Material
        self.E = config['material']['E']
        self.density = config['material']['rho']
        self.poisson = config['material']['nu']
        
        # Update uniform E vector unless it was already set by E_vector parameter
        if not hasattr(self, 'E_vector') or len(self.E_vector) != self.n_segments:
            self.E_vector = np.array([self.E] * self.n_segments)
        
    def apply_damage(self, damage_config):
        """
        Apply damage to beam elements based on damage configuration.
        
        Parameters:
        -----------
        damage_config : dict or list
            Either a single damage case dict with 'element' and 'factor',
            or a list of such dicts for multiple damage locations.
            Example:
            {
                'element': 4,
                'factor': 0.5
            }
            or
            [
                {'element': 4, 'factor': 0.5},
                {'element': 3, 'factor': 0.3}
            ]
        """
        # Reset E_vector to undamaged state
        self.E_vector = np.array([self.E] * self.n_segments)
        
        # Handle both single damage case and multiple damage cases
        if isinstance(damage_config, dict):
            damage_cases = [damage_config]
        else:
            damage_cases = damage_config
        
        # Apply each damage case
        for case in damage_cases:
            element = case.get('element')
            factor = case.get('factor')
            
            if element is None or factor is None:
                raise ValueError("Damage case must specify both 'element' and 'factor'")
            
            if not (0 <= element < self.n_segments):
                raise ValueError(f"Element index {element} out of range [0, {self.n_segments-1}]")
            
            if not (0 <= factor <= 1):
                raise ValueError(f"Damage factor {factor} must be between 0 and 1")
            
            # Apply damage factor to specified element
            self.E_vector[element] *= (1 - factor)
        
        
    def get_modal_properties(self, n_eigen, use_abaqus=False):
        """Calculate modal properties using current E vector"""
        if use_abaqus:
            from .modal_analysis_abaqus import calc_modal
        else:
            from .modal_analysis import calc_modal

        return calc_modal(
            elastic_modulus=self.E_vector,
            length=self.length,
            width=self.width,
            height=self.height,
            density=self.density,
            poisson=self.poisson,
            n_elements=self.n_elements,
            n_eigen=n_eigen,
            n_mp=self.n_mp
        )