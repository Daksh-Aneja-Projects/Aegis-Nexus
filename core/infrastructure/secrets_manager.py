"""
Secrets Manager for Aegis Nexus
Production-grade secrets management with support for Vault, AWS Secrets Manager, and environment variables.

This module provides a unified interface for secrets retrieval across different backends,
with automatic detection of the available secrets provider based on environment.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from functools import lru_cache
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SecretsManager(ABC):
    """Abstract base class for secrets management backends."""
    
    @abstractmethod
    async def get_secret(self, secret_name: str, version: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret value by name."""
        pass
    
    @abstractmethod
    async def get_secret_json(self, secret_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a JSON-structured secret."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the secrets backend is available."""
        pass


class VaultSecretsManager(SecretsManager):
    """
    HashiCorp Vault secrets manager.
    
    Supports both token-based auth and Kubernetes service account auth.
    In K8s, uses the sidecar injector pattern where secrets are mounted as files.
    """
    
    def __init__(self):
        self.vault_addr = os.getenv("VAULT_ADDR", "http://vault:8200")
        self.vault_token = os.getenv("VAULT_TOKEN")
        self.vault_role = os.getenv("VAULT_ROLE", "aegis-nexus")
        self.vault_mount = os.getenv("VAULT_MOUNT", "secret")
        self.vault_path_prefix = os.getenv("VAULT_PATH_PREFIX", "aegis-nexus")
        self._client = None
        self._token_expires_at: Optional[datetime] = None
        
        # K8s service account token path
        self.k8s_token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
        
    async def _ensure_client(self):
        """Initialize or refresh the Vault client."""
        if self._client is not None:
            # Check if token needs refresh
            if self._token_expires_at and datetime.utcnow() < self._token_expires_at:
                return
        
        try:
            import hvac
            
            self._client = hvac.Client(url=self.vault_addr)
            
            # Try Kubernetes auth first (when running in K8s)
            if os.path.exists(self.k8s_token_path):
                with open(self.k8s_token_path, 'r') as f:
                    jwt = f.read()
                
                auth_response = self._client.auth.kubernetes.login(
                    role=self.vault_role,
                    jwt=jwt
                )
                
                if auth_response and 'auth' in auth_response:
                    self._client.token = auth_response['auth']['client_token']
                    lease_duration = auth_response['auth'].get('lease_duration', 3600)
                    self._token_expires_at = datetime.utcnow() + timedelta(seconds=lease_duration - 60)
                    logger.info("🔐 Vault: Authenticated via Kubernetes service account")
                    return
            
            # Fallback to token-based auth
            if self.vault_token:
                self._client.token = self.vault_token
                logger.info("🔐 Vault: Using token-based authentication")
            else:
                logger.warning("⚠️ Vault: No authentication method available")
                
        except ImportError:
            logger.warning("⚠️ hvac library not installed - Vault integration disabled")
            self._client = None
        except Exception as e:
            logger.error(f"❌ Vault initialization failed: {e}")
            self._client = None
    
    async def get_secret(self, secret_name: str, version: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret from Vault KV v2 engine."""
        await self._ensure_client()
        
        if not self._client:
            return None
        
        try:
            path = f"{self.vault_path_prefix}/{secret_name}"
            
            response = self._client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.vault_mount,
                version=int(version) if version else None
            )
            
            if response and 'data' in response and 'data' in response['data']:
                data = response['data']['data']
                # Return 'value' key if present, otherwise serialize
                return data.get('value', json.dumps(data))
                
        except Exception as e:
            logger.error(f"❌ Vault: Failed to retrieve secret '{secret_name}': {e}")
        
        return None
    
    async def get_secret_json(self, secret_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a JSON secret from Vault."""
        await self._ensure_client()
        
        if not self._client:
            return None
        
        try:
            path = f"{self.vault_path_prefix}/{secret_name}"
            
            response = self._client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.vault_mount,
                version=int(version) if version else None
            )
            
            if response and 'data' in response and 'data' in response['data']:
                return response['data']['data']
                
        except Exception as e:
            logger.error(f"❌ Vault: Failed to retrieve JSON secret '{secret_name}': {e}")
        
        return None
    
    async def health_check(self) -> bool:
        """Check Vault connection health."""
        await self._ensure_client()
        
        if not self._client:
            return False
        
        try:
            return self._client.is_authenticated()
        except:
            return False


class AWSSecretsManager(SecretsManager):
    """
    AWS Secrets Manager integration.
    
    Uses boto3 with automatic credential discovery (IAM roles, env vars, etc.)
    """
    
    def __init__(self):
        self.region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        self.secret_prefix = os.getenv("AWS_SECRET_PREFIX", "aegis-nexus")
        self._client = None
        
    def _ensure_client(self):
        """Initialize the boto3 Secrets Manager client."""
        if self._client is not None:
            return
        
        try:
            import boto3
            from botocore.config import Config
            
            self._client = boto3.client(
                'secretsmanager',
                region_name=self.region,
                config=Config(
                    retries={'max_attempts': 3, 'mode': 'standard'}
                )
            )
            logger.info(f"🔐 AWS Secrets Manager: Initialized in region {self.region}")
            
        except ImportError:
            logger.warning("⚠️ boto3 library not installed - AWS Secrets Manager disabled")
            self._client = None
        except Exception as e:
            logger.error(f"❌ AWS Secrets Manager initialization failed: {e}")
            self._client = None
    
    async def get_secret(self, secret_name: str, version: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret from AWS Secrets Manager."""
        self._ensure_client()
        
        if not self._client:
            return None
        
        try:
            full_name = f"{self.secret_prefix}/{secret_name}"
            
            kwargs = {'SecretId': full_name}
            if version:
                kwargs['VersionId'] = version
            
            response = self._client.get_secret_value(**kwargs)
            
            if 'SecretString' in response:
                return response['SecretString']
            elif 'SecretBinary' in response:
                import base64
                return base64.b64decode(response['SecretBinary']).decode('utf-8')
                
        except Exception as e:
            logger.error(f"❌ AWS SM: Failed to retrieve secret '{secret_name}': {e}")
        
        return None
    
    async def get_secret_json(self, secret_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve and parse a JSON secret from AWS Secrets Manager."""
        secret_string = await self.get_secret(secret_name, version)
        
        if secret_string:
            try:
                return json.loads(secret_string)
            except json.JSONDecodeError as e:
                logger.error(f"❌ AWS SM: Secret '{secret_name}' is not valid JSON: {e}")
        
        return None
    
    async def health_check(self) -> bool:
        """Check AWS Secrets Manager availability."""
        self._ensure_client()
        
        if not self._client:
            return False
        
        try:
            # Try to list secrets (just check we can connect)
            self._client.list_secrets(MaxResults=1)
            return True
        except Exception as e:
            logger.warning(f"⚠️ AWS SM health check failed: {e}")
            return False


class EnvSecretsManager(SecretsManager):
    """
    Environment variable-based secrets manager.
    
    Fallback for development environments. Secrets are read from environment variables
    with a configurable prefix (default: AEGIS_SECRET_).
    """
    
    def __init__(self):
        self.prefix = os.getenv("AEGIS_SECRET_PREFIX", "AEGIS_SECRET_")
        logger.info(f"🔐 Using environment variables for secrets (prefix: {self.prefix})")
    
    async def get_secret(self, secret_name: str, version: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret from environment variables."""
        if os.getenv("AEGIS_ENV") == "production":
            logger.error(f"⛔ SECURITY VIOLATION: Environment secrets fallback attempted for '{secret_name}' in production!")
            return None
            
        # Convert secret name to env var format: redis-url -> AEGIS_SECRET_REDIS_URL
        env_name = f"{self.prefix}{secret_name.upper().replace('-', '_').replace('/', '_')}"
        val = os.getenv(env_name)
        
        if val:
            # Mask secret in debug logs
            masked = val[:2] + "*" * (len(val) - 4) + val[-2:] if len(val) > 4 else "****"
            logger.debug(f"🔓 EnvSecrets: Retrieved '{secret_name}' (Value: {masked})")
            
        return val
    
    async def get_secret_json(self, secret_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve and parse a JSON secret from environment variables."""
        secret_string = await self.get_secret(secret_name, version)
        
        if secret_string:
            try:
                return json.loads(secret_string)
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Env secret '{secret_name}' is not valid JSON: {e}")
        
        return None
    
    async def health_check(self) -> bool:
        """Environment secrets are always available."""
        return True


class UnifiedSecretsManager:
    """
    Unified secrets manager that auto-detects the best available backend.
    
    Priority order:
    1. HashiCorp Vault (if VAULT_ADDR is set)
    2. AWS Secrets Manager (if running on AWS or AWS credentials available)
    3. Environment variables (fallback)
    """
    
    _instance = None
    _backend: Optional[SecretsManager] = None
    _backend_name: str = "unknown"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UnifiedSecretsManager, cls).__new__(cls)
        return cls._instance
    
    async def initialize(self) -> bool:
        """Initialize the secrets manager, auto-detecting the best backend."""
        
        # Try Vault first
        if os.getenv("VAULT_ADDR"):
            vault = VaultSecretsManager()
            if await vault.health_check():
                self._backend = vault
                self._backend_name = "vault"
                logger.info("✅ Secrets Manager: Using HashiCorp Vault")
                return True
            else:
                logger.warning("⚠️ Vault configured but health check failed")
        
        # Try AWS Secrets Manager
        if os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"):
            aws_sm = AWSSecretsManager()
            if await aws_sm.health_check():
                self._backend = aws_sm
                self._backend_name = "aws"
                logger.info("✅ Secrets Manager: Using AWS Secrets Manager")
                return True
            else:
                logger.warning("⚠️ AWS credentials present but health check failed")
        
        # Fallback to environment variables
        self._backend = EnvSecretsManager()
        self._backend_name = "env"
        logger.info("✅ Secrets Manager: Using environment variables (development mode)")
        return True
    
    async def get_secret(self, secret_name: str, version: Optional[str] = None) -> Optional[str]:
        """Get a secret from the active backend."""
        if not self._backend:
            await self.initialize()
        return await self._backend.get_secret(secret_name, version)
    
    async def get_secret_json(self, secret_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get a JSON secret from the active backend."""
        if not self._backend:
            await self.initialize()
        return await self._backend.get_secret_json(secret_name, version)
    
    async def health_check(self) -> Dict[str, Any]:
        """Get health status of the secrets manager."""
        if not self._backend:
            await self.initialize()
        
        is_healthy = await self._backend.health_check()
        
        return {
            "backend": self._backend_name,
            "healthy": is_healthy,
            "class": type(self._backend).__name__
        }
    
    @property
    def backend_name(self) -> str:
        """Get the name of the active backend."""
        return self._backend_name


# Global instance
_secrets_manager: Optional[UnifiedSecretsManager] = None


async def get_secrets_manager() -> UnifiedSecretsManager:
    """Get the global unified secrets manager instance."""
    global _secrets_manager
    
    if _secrets_manager is None:
        _secrets_manager = UnifiedSecretsManager()
        await _secrets_manager.initialize()
    
    return _secrets_manager


async def get_secret(secret_name: str, version: Optional[str] = None) -> Optional[str]:
    """Convenience function to get a secret."""
    manager = await get_secrets_manager()
    return await manager.get_secret(secret_name, version)


async def get_secret_json(secret_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Convenience function to get a JSON secret."""
    manager = await get_secrets_manager()
    return await manager.get_secret_json(secret_name, version)
