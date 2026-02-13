"""
Core Agent Abstraction.

This module defines the base Agent class that all specialized agents must inherit from.
It enforces a consistent interface for run() and state management.
"""
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel


class AgentState(BaseModel):
    """Base state for any agent."""
    name: str
    status: str = "idle"
    memory: dict[str, Any] = {}

S = TypeVar("S", bound=AgentState)

class Agent(ABC, Generic[S]):
    """
    Abstract Base Agent.
    
    Attributes:
        name (str): The name of the agent.
        state (S): The agent's internal state.
    """
    
    def __init__(self, name: str, state: S | None = None):
        self.name = name
        self.state = state or AgentState(name=name)
        
    @abstractmethod
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the agent's main logic.
        
        Args:
            context: Shared context/memory from the orchestrator.
            
        Returns:
            Dict containing the results of the execution.
        """
        pass

    def update_memory(self, key: str, value: Any) -> None:
        """Update the agent's internal memory."""
        self.state.memory[key] = value

    def log(self, message: str, level: str = "INFO") -> None:
        """Structured logging hook (to be integrated with Logger)."""
        print(f"[{level}] [{self.name}] {message}")
