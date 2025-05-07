from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from environment import DataCenterEnv
from config import DataCenterConfig

class BaseAgent(ABC):
    def __init__(self, env: DataCenterEnv, config: DataCenterConfig):
        self.env = env
        self.config = config
    
    @abstractmethod
    def act(self, state: Tuple[int, int, int, int]) -> Tuple[int, int, int, int, int]:
        """Select an action for the given state."""
        pass
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Train the agent (if applicable)."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the agent's state."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the agent's state."""
        pass 