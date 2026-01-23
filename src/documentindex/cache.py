"""
Caching layer for DocumentIndex.

Supports multiple backends:
- Memory (default, for development)
- File system (for persistence)
- Redis (for distributed/production)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any, TypeVar, Generic
from datetime import datetime, timedelta
import json
import hashlib
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """A cached item with metadata"""
    key: str
    value: T
    created_at: datetime
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class CacheBackend(ABC):
    """Abstract cache backend"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL in seconds"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries"""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache (for development/testing)"""
    
    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
    
    async def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._cache[key]
            return None
        entry.hit_count += 1
        return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        # Evict if at capacity (remove least hit entry)
        if len(self._cache) >= self._max_size and key not in self._cache:
            if self._cache:
                min_key = min(self._cache.keys(), key=lambda k: self._cache[k].hit_count)
                del self._cache[min_key]
        
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at,
        )
    
    async def delete(self, key: str) -> None:
        self._cache.pop(key, None)
    
    async def exists(self, key: str) -> bool:
        entry = self._cache.get(key)
        if entry and entry.is_expired():
            del self._cache[key]
            return False
        return key in self._cache
    
    async def clear(self) -> None:
        self._cache.clear()
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "total_hits": sum(e.hit_count for e in self._cache.values()),
        }


class FileCache(CacheBackend):
    """File-based cache for persistence"""
    
    def __init__(self, cache_dir: str = ".cache/documentindex"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str) -> Path:
        # Hash key for filesystem-safe filename
        hashed = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._cache_dir / f"{hashed}.cache"
    
    async def get(self, key: str) -> Optional[Any]:
        path = self._get_path(key)
        if not path.exists():
            return None
        
        try:
            with open(path, 'rb') as f:
                entry: CacheEntry = pickle.load(f)
            
            if entry.is_expired():
                path.unlink()
                return None
            
            return entry.value
        except Exception as e:
            logger.warning(f"Cache read error for {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at,
        )
        
        path = self._get_path(key)
        try:
            with open(path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")
    
    async def delete(self, key: str) -> None:
        path = self._get_path(key)
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.warning(f"Cache delete error for {key}: {e}")
    
    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None
    
    async def clear(self) -> None:
        for path in self._cache_dir.glob("*.cache"):
            try:
                path.unlink()
            except Exception as e:
                logger.warning(f"Cache clear error for {path}: {e}")


class RedisCache(CacheBackend):
    """Redis-based cache for distributed/production use"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "docindex:",
        password: Optional[str] = None,
    ):
        self._prefix = prefix
        self._redis = None
        self._config = {
            "host": host,
            "port": port,
            "db": db,
            "password": password,
        }
    
    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as redis
                self._redis = redis.Redis(
                    **{k: v for k, v in self._config.items() if v is not None},
                    decode_responses=False
                )
            except ImportError:
                raise ImportError(
                    "redis is required for Redis cache. "
                    "Install it with: pip install redis"
                )
        return self._redis
    
    def _make_key(self, key: str) -> str:
        return f"{self._prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        r = await self._get_redis()
        data = await r.get(self._make_key(key))
        if data is None:
            return None
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Redis deserialize error for {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        r = await self._get_redis()
        try:
            data = pickle.dumps(value)
            if ttl:
                await r.setex(self._make_key(key), ttl, data)
            else:
                await r.set(self._make_key(key), data)
        except Exception as e:
            logger.warning(f"Redis write error for {key}: {e}")
    
    async def delete(self, key: str) -> None:
        r = await self._get_redis()
        await r.delete(self._make_key(key))
    
    async def exists(self, key: str) -> bool:
        r = await self._get_redis()
        return bool(await r.exists(self._make_key(key)))
    
    async def clear(self) -> None:
        r = await self._get_redis()
        keys = await r.keys(f"{self._prefix}*")
        if keys:
            await r.delete(*keys)


# ============================================================================
# Cache Manager (High-level interface)
# ============================================================================

@dataclass
class CacheConfig:
    """Cache configuration"""
    backend: str = "memory"  # "memory", "file", "redis"
    
    # Memory cache settings
    memory_max_size: int = 1000
    
    # File cache settings
    file_cache_dir: str = ".cache/documentindex"
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_prefix: str = "docindex:"
    
    # TTL defaults (in seconds)
    index_ttl: int = 86400 * 7  # 7 days for document indexes
    llm_response_ttl: int = 3600  # 1 hour for LLM responses
    search_result_ttl: int = 1800  # 30 minutes for search results


class CacheManager:
    """
    High-level cache manager for DocumentIndex.
    
    Caches:
    - Document indexes (keyed by doc_id or content hash)
    - LLM responses (keyed by prompt hash)
    - Search results (keyed by query + doc_id)
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._backend = self._create_backend()
    
    def _create_backend(self) -> CacheBackend:
        if self.config.backend == "memory":
            return MemoryCache(self.config.memory_max_size)
        elif self.config.backend == "file":
            return FileCache(self.config.file_cache_dir)
        elif self.config.backend == "redis":
            return RedisCache(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                prefix=self.config.redis_prefix,
            )
        else:
            raise ValueError(f"Unknown cache backend: {self.config.backend}")
    
    # -------------------------------------------------------------------------
    # Key generation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _hash_content(content: str) -> str:
        """Generate hash for content"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _index_key(self, doc_id: str) -> str:
        return f"index:{doc_id}"
    
    def _llm_key(self, prompt: str, model: str) -> str:
        prompt_hash = self._hash_content(prompt)
        return f"llm:{model}:{prompt_hash}"
    
    def _search_key(self, doc_id: str, query: str) -> str:
        query_hash = self._hash_content(query)
        return f"search:{doc_id}:{query_hash}"
    
    # -------------------------------------------------------------------------
    # Document Index caching
    # -------------------------------------------------------------------------
    
    async def get_index(self, doc_id: str) -> Optional[dict]:
        """Get cached document index"""
        return await self._backend.get(self._index_key(doc_id))
    
    async def set_index(self, doc_id: str, index_data: dict) -> None:
        """Cache document index"""
        await self._backend.set(
            self._index_key(doc_id),
            index_data,
            ttl=self.config.index_ttl,
        )
    
    async def has_index(self, doc_id: str) -> bool:
        """Check if index is cached"""
        return await self._backend.exists(self._index_key(doc_id))
    
    async def delete_index(self, doc_id: str) -> None:
        """Delete cached index"""
        await self._backend.delete(self._index_key(doc_id))
    
    # -------------------------------------------------------------------------
    # LLM Response caching
    # -------------------------------------------------------------------------
    
    async def get_llm_response(self, prompt: str, model: str) -> Optional[str]:
        """Get cached LLM response"""
        return await self._backend.get(self._llm_key(prompt, model))
    
    async def set_llm_response(self, prompt: str, model: str, response: str) -> None:
        """Cache LLM response"""
        await self._backend.set(
            self._llm_key(prompt, model),
            response,
            ttl=self.config.llm_response_ttl,
        )
    
    # -------------------------------------------------------------------------
    # Search Result caching
    # -------------------------------------------------------------------------
    
    async def get_search_result(self, doc_id: str, query: str) -> Optional[Any]:
        """Get cached search result"""
        return await self._backend.get(self._search_key(doc_id, query))
    
    async def set_search_result(self, doc_id: str, query: str, result: Any) -> None:
        """Cache search result"""
        await self._backend.set(
            self._search_key(doc_id, query),
            result,
            ttl=self.config.search_result_ttl,
        )
    
    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    
    async def clear_all(self) -> None:
        """Clear all cached data"""
        await self._backend.clear()
    
    async def get(self, key: str) -> Optional[Any]:
        """Direct access to cache backend"""
        return await self._backend.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Direct access to cache backend"""
        await self._backend.set(key, value, ttl)
