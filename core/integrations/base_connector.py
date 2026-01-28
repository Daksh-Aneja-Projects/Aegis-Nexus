from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ExternalSystemConnector(ABC):
    """
    KORE "Headless" Abstraction Layer.
    Decouples logic from specific vendors (Workday, SAP, Oracle).
    Level 5: Autonomous agents use these standardized protocols.
    """
    
    @abstractmethod
    async def fetch_data(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize data from the external silo."""
        pass

    @abstractmethod
    async def execute_action(self, action: str, payload: Dict[str, Any]) -> bool:
        """Execute a sovereign action (headless)."""
        pass
    
    @abstractmethod
    async def rollback_action(self, transaction_id: str) -> bool:
        """Compensating transaction for Saga Pattern."""
        pass
