from abc import ABC, abstractmethod
import torch

class EvolutionaryAgent(ABC, torch.nn.Module):
    """
    Abstract Base Class for any agent in the Thermodynamic system.
    Enforces Interface Segregation.
    """
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_topology_info(self):
        """Returns details about current architecture (e.g., hidden size)."""
        pass

class Mutator(ABC):
    """
    Strategy Pattern for mutation logic.
    """
    @abstractmethod
    def mutate(self, agent: EvolutionaryAgent) -> EvolutionaryAgent:
        pass

class EnvironmentAdapter(ABC):
    """
    Adapter Pattern for Gym environments to ensure consistent interface.
    """
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def get_dims(self):
        pass

class ExperimentLogger(ABC):
    """
    Observer Pattern for logging experiment data.
    """
    @abstractmethod
    def log(self, episode, score, metrics):
        pass
    
    @abstractmethod
    def save(self):
        pass
