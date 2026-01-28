from typing import Dict, Type
from core.integrations.base_connector import ExternalSystemConnector
from core.integrations.workday_connector import WorkdayConnector
from core.integrations.sap_connector import SAPConnector

class ConnectorFactory:
    """
    Factory validation logic for KORE integrations.
    """
    
    _registry: Dict[str, Type[ExternalSystemConnector]] = {
        "workday": WorkdayConnector,
        "sap": SAPConnector
    }
    
    @classmethod
    def get_connector(cls, vendor: str) -> ExternalSystemConnector:
        connector_cls = cls._registry.get(vendor.lower())
        if not connector_cls:
            raise ValueError(f"Unknown vendor connector: {vendor}")
        return connector_cls()
