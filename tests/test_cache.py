"""
Tests for caching functionality.
"""

import pytest
import asyncio
import tempfile
import os

from documentindex.cache import (
    CacheConfig,
    CacheManager,
    MemoryCache,
    FileCache,
)


class TestMemoryCache:
    """Tests for in-memory cache"""
    
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        cache = MemoryCache()
        await cache.set("key1", "value1")
        
        result = await cache.get("key1")
        assert result == "value1"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        cache = MemoryCache()
        result = await cache.get("nonexistent")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_delete(self):
        cache = MemoryCache()
        await cache.set("key1", "value1")
        await cache.delete("key1")
        
        result = await cache.get("key1")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_exists(self):
        cache = MemoryCache()
        
        assert await cache.exists("key1") is False
        
        await cache.set("key1", "value1")
        assert await cache.exists("key1") is True
    
    @pytest.mark.asyncio
    async def test_clear(self):
        cache = MemoryCache()
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        
        await cache.clear()
        
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        cache = MemoryCache()
        await cache.set("key1", "value1", ttl=1)  # 1 second TTL
        
        # Should exist immediately
        assert await cache.get("key1") == "value1"
        
        # Wait for expiration
        await asyncio.sleep(1.5)
        
        # Should be expired
        assert await cache.get("key1") is None
    
    @pytest.mark.asyncio
    async def test_max_size_eviction(self):
        cache = MemoryCache(max_size=3)
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")
        await cache.set("key4", "value4")  # Should trigger eviction
        
        # One of the first three keys should be evicted
        total = sum(1 for k in ["key1", "key2", "key3", "key4"] 
                   if await cache.get(k) is not None)
        assert total == 3
    
    def test_get_stats(self):
        cache = MemoryCache(max_size=100)
        stats = cache.get_stats()
        
        assert "size" in stats
        assert "max_size" in stats
        assert stats["max_size"] == 100
    
    @pytest.mark.asyncio
    async def test_complex_values(self):
        cache = MemoryCache()
        
        # Store dict
        await cache.set("dict_key", {"nested": {"data": [1, 2, 3]}})
        result = await cache.get("dict_key")
        assert result["nested"]["data"] == [1, 2, 3]
        
        # Store list
        await cache.set("list_key", [1, 2, 3, 4, 5])
        result = await cache.get("list_key")
        assert result == [1, 2, 3, 4, 5]


class TestFileCache:
    """Tests for file-based cache"""
    
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(tmpdir)
            await cache.set("key1", "value1")
            
            result = await cache.get("key1")
            assert result == "value1"
    
    @pytest.mark.asyncio
    async def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write with one cache instance
            cache1 = FileCache(tmpdir)
            await cache1.set("key1", "value1")
            
            # Read with another cache instance
            cache2 = FileCache(tmpdir)
            result = await cache2.get("key1")
            assert result == "value1"
    
    @pytest.mark.asyncio
    async def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(tmpdir)
            await cache.set("key1", "value1")
            await cache.delete("key1")
            
            result = await cache.get("key1")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(tmpdir)
            await cache.set("key1", "value1")
            await cache.set("key2", "value2")
            
            await cache.clear()
            
            assert await cache.get("key1") is None
            assert await cache.get("key2") is None
    
    @pytest.mark.asyncio
    async def test_complex_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FileCache(tmpdir)
            
            data = {
                "string": "test",
                "number": 42,
                "list": [1, 2, 3],
                "nested": {"a": "b"},
            }
            await cache.set("complex", data)
            
            result = await cache.get("complex")
            assert result == data


class TestCacheManager:
    """Tests for high-level cache manager"""
    
    @pytest.mark.asyncio
    async def test_memory_backend(self):
        config = CacheConfig(backend="memory")
        manager = CacheManager(config)
        
        await manager.set("test", "value")
        result = await manager.get("test")
        assert result == "value"
    
    @pytest.mark.asyncio
    async def test_file_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CacheConfig(backend="file", file_cache_dir=tmpdir)
            manager = CacheManager(config)
            
            await manager.set("test", "value")
            result = await manager.get("test")
            assert result == "value"
    
    @pytest.mark.asyncio
    async def test_index_caching(self):
        config = CacheConfig(backend="memory")
        manager = CacheManager(config)
        
        index_data = {"doc_id": "test123", "structure": []}
        
        await manager.set_index("test123", index_data)
        result = await manager.get_index("test123")
        
        assert result == index_data
    
    @pytest.mark.asyncio
    async def test_has_index(self):
        config = CacheConfig(backend="memory")
        manager = CacheManager(config)
        
        assert await manager.has_index("nonexistent") is False
        
        await manager.set_index("test123", {"data": "test"})
        assert await manager.has_index("test123") is True
    
    @pytest.mark.asyncio
    async def test_delete_index(self):
        config = CacheConfig(backend="memory")
        manager = CacheManager(config)
        
        await manager.set_index("test123", {"data": "test"})
        await manager.delete_index("test123")
        
        assert await manager.has_index("test123") is False
    
    @pytest.mark.asyncio
    async def test_llm_response_caching(self):
        config = CacheConfig(backend="memory")
        manager = CacheManager(config)
        
        prompt = "What is 2+2?"
        model = "gpt-4"
        response = "4"
        
        await manager.set_llm_response(prompt, model, response)
        result = await manager.get_llm_response(prompt, model)
        
        assert result == response
    
    @pytest.mark.asyncio
    async def test_search_result_caching(self):
        config = CacheConfig(backend="memory")
        manager = CacheManager(config)
        
        doc_id = "test123"
        query = "revenue"
        results = [{"node_id": "0001", "score": 0.9}]
        
        await manager.set_search_result(doc_id, query, results)
        cached = await manager.get_search_result(doc_id, query)
        
        assert cached == results
    
    @pytest.mark.asyncio
    async def test_clear_all(self):
        config = CacheConfig(backend="memory")
        manager = CacheManager(config)
        
        await manager.set_index("test1", {})
        await manager.set_index("test2", {})
        
        await manager.clear_all()
        
        assert await manager.has_index("test1") is False
        assert await manager.has_index("test2") is False
    
    def test_invalid_backend(self):
        config = CacheConfig(backend="invalid")
        with pytest.raises(ValueError):
            CacheManager(config)


class TestCacheConfig:
    """Tests for cache configuration"""
    
    def test_default_config(self):
        config = CacheConfig()
        assert config.backend == "memory"
        assert config.memory_max_size == 1000
        assert config.index_ttl > 0
    
    def test_custom_config(self):
        config = CacheConfig(
            backend="file",
            file_cache_dir="/tmp/test_cache",
            index_ttl=3600,
        )
        assert config.backend == "file"
        assert config.index_ttl == 3600
