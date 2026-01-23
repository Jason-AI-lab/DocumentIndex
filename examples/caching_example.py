"""
Caching Example: Using different cache backends.

This example demonstrates:
1. Memory cache (development)
2. File cache (persistence)
3. Redis cache (production)
4. Caching document indexes
5. Caching search results
"""

import asyncio
import tempfile
from documentindex import (
    CacheConfig,
    CacheManager,
    MemoryCache,
    FileCache,
    DocumentIndexer,
    NodeSearcher,
    IndexerConfig,
    LLMConfig,
)


SAMPLE_DOC = """
Annual Report 2024

Financial Summary

Total Revenue: $10 billion
Net Income: $1.5 billion  
Operating Margin: 25%

Our growth was driven by strong demand in key markets.

Risk Factors

Market volatility continues to present challenges.
Supply chain disruptions may affect operations.
"""


async def memory_cache_example():
    """Using in-memory cache"""
    print("=" * 60)
    print("EXAMPLE 1: Memory Cache")
    print("=" * 60)
    
    # Create memory cache configuration
    config = CacheConfig(
        backend="memory",
        memory_max_size=1000,  # Max items
        index_ttl=86400 * 7,   # 7 days for indexes
        llm_response_ttl=3600,  # 1 hour for LLM responses
        search_result_ttl=1800, # 30 minutes for searches
    )
    
    cache = CacheManager(config)
    
    # Store and retrieve data
    await cache.set("test_key", {"data": "value"})
    result = await cache.get("test_key")
    print(f"Stored and retrieved: {result}")
    
    # Index caching
    await cache.set_index("doc123", {"structure": []})
    has_index = await cache.has_index("doc123")
    print(f"Index cached: {has_index}")
    
    # LLM response caching
    await cache.set_llm_response("What is 2+2?", "gpt-4", "4")
    llm_result = await cache.get_llm_response("What is 2+2?", "gpt-4")
    print(f"LLM response cached: {llm_result}")
    
    print("\nMemory cache is ideal for development and testing")


async def file_cache_example():
    """Using file-based cache"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: File Cache")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as cache_dir:
        # Create file cache configuration
        config = CacheConfig(
            backend="file",
            file_cache_dir=cache_dir,
        )
        
        cache = CacheManager(config)
        
        # Store data
        await cache.set("persistent_key", {"data": "This persists to disk"})
        print(f"Data stored in: {cache_dir}")
        
        # Retrieve data
        result = await cache.get("persistent_key")
        print(f"Retrieved: {result}")
        
        # Create a new cache instance (simulating restart)
        cache2 = CacheManager(config)
        result2 = await cache2.get("persistent_key")
        print(f"Data survives restart: {result2 is not None}")
    
    print("\nFile cache is ideal for single-server persistence")


async def redis_cache_example():
    """Using Redis cache (requires Redis server)"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Redis Cache (Configuration)")
    print("=" * 60)
    
    # Create Redis cache configuration
    config = CacheConfig(
        backend="redis",
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        redis_password=None,  # Set if Redis requires auth
        redis_prefix="docindex:",  # Key prefix
    )
    
    print("Redis configuration:")
    print(f"  Host: {config.redis_host}")
    print(f"  Port: {config.redis_port}")
    print(f"  DB: {config.redis_db}")
    print(f"  Prefix: {config.redis_prefix}")
    
    print("\nRedis cache is ideal for:")
    print("  - Distributed systems")
    print("  - Multiple server instances")
    print("  - High-performance caching")
    print("  - Production environments")
    
    # Uncomment to use (requires Redis server running)
    # cache = CacheManager(config)
    # await cache.set("redis_key", {"data": "value"})
    # result = await cache.get("redis_key")
    # print(f"Retrieved from Redis: {result}")


async def indexer_with_cache_example():
    """Using cache with document indexer"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Indexer with Cache")
    print("=" * 60)
    
    # Create cache manager
    cache_config = CacheConfig(backend="memory")
    cache = CacheManager(cache_config)
    
    # Create indexer with cache
    indexer_config = IndexerConfig(
        llm_config=LLMConfig(model="gpt-4o"),
        use_cache=True,
    )
    indexer = DocumentIndexer(indexer_config, cache_manager=cache)
    
    # First indexing (no cache hit)
    print("First indexing (cache miss)...")
    doc_index = await indexer.index(
        text=SAMPLE_DOC,
        doc_name="annual_report",
        doc_id="report_2024",
    )
    print(f"  Nodes: {doc_index.get_node_count()}")
    
    # Cache the index
    await cache.set_index(doc_index.doc_id, doc_index.to_dict(include_text=True))
    print("  Index cached")
    
    # Check cache
    has_cached = await cache.has_index("report_2024")
    print(f"  Cache verified: {has_cached}")


async def search_with_cache_example():
    """Using cache with search"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Search with Cache")
    print("=" * 60)
    
    # Create cache
    cache_config = CacheConfig(backend="memory")
    cache = CacheManager(cache_config)
    
    # Create indexer and index document
    indexer_config = IndexerConfig(
        llm_config=LLMConfig(model="gpt-4o"),
    )
    indexer = DocumentIndexer(indexer_config)
    
    doc_index = await indexer.index(
        text=SAMPLE_DOC,
        doc_name="test_doc",
    )
    
    # Create searcher with cache
    searcher = NodeSearcher(doc_index, cache_manager=cache)
    
    query = "financial performance"
    
    # First search (cache miss)
    print(f"First search: '{query}'")
    results1 = await searcher.find_related_nodes(query)
    print(f"  Found: {len(results1)} nodes")
    
    # Second search (cache hit)
    print(f"Second search: '{query}'")
    results2 = await searcher.find_related_nodes(query)
    print(f"  Found: {len(results2)} nodes (from cache)")


async def cache_patterns_example():
    """Common caching patterns"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Common Caching Patterns")
    print("=" * 60)
    
    cache = CacheManager(CacheConfig(backend="memory"))
    
    # Pattern 1: Check-then-compute
    print("\n1. Check-then-compute pattern:")
    
    async def get_or_compute(key, compute_fn):
        result = await cache.get(key)
        if result is not None:
            print("    Cache hit!")
            return result
        
        print("    Cache miss, computing...")
        result = await compute_fn()
        await cache.set(key, result, ttl=3600)
        return result
    
    async def expensive_computation():
        await asyncio.sleep(0.1)  # Simulate work
        return {"computed": "value"}
    
    result1 = await get_or_compute("compute_key", expensive_computation)
    result2 = await get_or_compute("compute_key", expensive_computation)
    
    # Pattern 2: Cache invalidation
    print("\n2. Cache invalidation pattern:")
    
    await cache.set_index("doc1", {"version": 1})
    print("    Stored version 1")
    
    # Document updated, invalidate cache
    await cache.delete_index("doc1")
    print("    Cache invalidated")
    
    await cache.set_index("doc1", {"version": 2})
    print("    Stored version 2")
    
    # Pattern 3: TTL-based expiration
    print("\n3. TTL-based expiration:")
    print("    - index_ttl: 7 days (stable data)")
    print("    - llm_response_ttl: 1 hour (may change with prompt)")
    print("    - search_result_ttl: 30 min (query results)")


async def main():
    """Run caching examples"""
    print("\n" + "#" * 60)
    print("DocumentIndex Caching Examples")
    print("#" * 60)
    
    await memory_cache_example()
    await file_cache_example()
    await redis_cache_example()
    await indexer_with_cache_example()
    await search_with_cache_example()
    await cache_patterns_example()
    
    print("\n" + "=" * 60)
    print("Caching examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
