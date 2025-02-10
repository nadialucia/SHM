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
        
        # Material
        self.E = config['material']['E']
        self.density = config['material']['rho']
        self.poisson = config['material']['nu']
        
        # Update uniform E vector unless it was already set by E_vector parameter
        if not hasattr(self, 'E_vector') or len(self.E_vector) != self.n_segments:
            self.E_vector = np.array([self.E] * self.n_segments)
        
    def apply_damage(self, element, factor):
        """Apply damage to specific element"""
        self.E_vector[element] *= (1 - factor)
        
    def get_modal_properties(self, n_eigen=4, use_abaqus=False):
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
            n_eigen=n_eigen
        )