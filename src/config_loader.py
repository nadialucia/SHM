import yaml
import os
from pathlib import Path

class ConfigLoader:
    def __init__(self):
        self.root_dir = Path(os.getcwd())
        self.config_dir = self.root_dir / "config"
    
    def _convert_to_float(self, value):
        if isinstance(value, str):
            if 'e' in value.lower():
                base, exp = value.lower().split('e')
                return float(base) * (10 ** float(exp))
            return float(value)
        return value
        
    def _convert_numeric_values(self, config):
        # Convert material properties
        config['material']['E'] = self._convert_to_float(config['material']['E'])
        config['material']['rho'] = self._convert_to_float(config['material']['rho'])
        config['material']['nu'] = self._convert_to_float(config['material']['nu'])
        
        # Convert geometry values
        config['geometry']['width'] = self._convert_to_float(config['geometry']['width'])
        config['geometry']['height'] = self._convert_to_float(config['geometry']['height'])
        config['geometry']['length'] = self._convert_to_float(config['geometry']['length'])
        
        # Convert damage values
        config['damage']['intactness'] = [self._convert_to_float(x) for x in config['damage']['intactness']]
        
        return config
            
    def load_beam_config(self):
        with open(self.config_dir / "beam_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            return self._convert_numeric_values(config)
            
    def load_damage_scenarios(self):
        with open(self.config_dir / "damage_scenarios.yaml", 'r') as f:
            return yaml.safe_load(f)