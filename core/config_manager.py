import json
import os

class ConfigManager:
    """
    Centralized Configuration Manager for Thesis experiments.
    Reads and writes to core/config.json.
    """
    _CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config.json')

    @classmethod
    def load_config(cls):
        """Loads the hyperparameter configuration from config.json."""
        if not os.path.exists(cls._CONFIG_FILE):
            return {}
        with open(cls._CONFIG_FILE, 'r') as f:
            return json.load(f)

    @classmethod
    def save_config(cls, new_config):
        """Saves the provided dictionary back to config.json safely."""
        with open(cls._CONFIG_FILE, 'w') as f:
            json.dump(new_config, f, indent=4)
