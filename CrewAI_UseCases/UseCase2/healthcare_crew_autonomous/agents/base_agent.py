from typing import Any, Dict

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def receive(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receive a message dict and return a response dict.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
