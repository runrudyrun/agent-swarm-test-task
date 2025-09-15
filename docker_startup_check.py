#!/usr/bin/env python3
"""
Docker startup check script for vector store validation and re-indexing.
Leverages existing project code instead of duplicating logic.
"""

import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_vector_store_compatibility():
    """Check if vector store is compatible with current embedding configuration."""
    try:
        from langchain_chroma import Chroma
        from rag.config import RAGConfig, get_embeddings
        
        logger.info("Checking vector store compatibility...")
        
        # Ensure cache directory exists before trying to use embeddings
        cache_dir = Path("/app/data/.cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set proper cache directory for embeddings to avoid permission issues
        os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
        os.environ['HF_HOME'] = str(cache_dir)
        
        vectorstore = Chroma(
            persist_directory=RAGConfig.VECTOR_STORE_PATH,
            embedding_function=get_embeddings(),
            collection_name=RAGConfig.COLLECTION_NAME
        )
        
        # Test if we can access the collection
        count = vectorstore._collection.count()
        logger.info(f"Vector store loaded successfully with {count} documents")
        if count == 0:
            logger.info("Vector store is present but empty; treating as incompatible to trigger ingestion")
            return False, "EMPTY"
        return True, count
        
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Vector store check failed: {error_msg}")
        
        # Check if it's a dimension mismatch error
        if "dimension" in error_msg.lower():
            logger.info("Detected embedding dimension mismatch")
            return False, "DIMENSION_MISMATCH"
        elif "permission" in error_msg.lower() or "cache" in error_msg.lower():
            logger.info("Detected cache/permission issue with embeddings")
            return False, "CACHE_ERROR"
        else:
            logger.info("Vector store has other compatibility issues")
            return False, "OTHER_ERROR"

def reindex_if_needed():
    """Re-index documents if needed using existing RAG infrastructure."""
    try:
        # Set up cache directory - try primary location first, fallback to home
        cache_dir = Path('/app/data/.cache')
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = str(cache_dir)
        except PermissionError:
            # Fallback to home directory cache
            home_cache = Path.home() / ".cache" / "infinitepay"
            home_cache.mkdir(parents=True, exist_ok=True)
            cache_path = str(home_cache)
        
        # Set proper cache directory for embeddings to avoid permission issues
        os.environ['TRANSFORMERS_CACHE'] = cache_path
        os.environ['HF_HOME'] = cache_path
        
        # Check current embedding provider
        embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER", "local")
        
        # Decide strategy based on availability of raw data
        raw_data_path = Path("./data/raw")
        has_raw = raw_data_path.exists() and any(raw_data_path.glob("*.json"))

        if embeddings_provider == "openai" and os.getenv("OPENAI_API_KEY") and has_raw:
            logger.info("Re-indexing with OpenAI embeddings using existing raw data...")
            from rag.reindex_openai import reindex_with_openai
            reindex_with_openai()
        else:
            # Fresh ingestion path (works both first-time and refresh)
            logger.info("Running fresh ingestion with current embeddings provider...")
            from rag.ingest import ingest_infinitepay_content
            import asyncio
            result = asyncio.run(ingest_infinitepay_content())
            logger.info(f"Ingestion completed with {result} document chunks")
        
        return True
        
    except Exception as e:
        logger.error(f"Re-indexing failed: {e}")
        return False

def main():
    """Main startup check logic."""
    logger.info("🚀 Starting InfinitePay Agent Swarm API startup checks...")
    
    # Ensure base data directory exists first
    base_data_dir = Path("/app/data")
    try:
        base_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Base data directory created/verified: {base_data_dir}")
    except Exception as e:
        logger.warning(f"⚠️  Could not create base data directory: {e}")
    
    # Ensure cache directory exists with proper permissions
    cache_dir = Path("/app/data/.cache")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Cache directory created/verified: {cache_dir}")
    except Exception as e:
        logger.warning(f"⚠️  Could not create cache directory: {e}")
        # Try alternative cache location in home directory
        home_cache = Path.home() / ".cache" / "infinitepay"
        try:
            home_cache.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Using alternative cache directory: {home_cache}")
            # Update environment variables
            os.environ['TRANSFORMERS_CACHE'] = str(home_cache)
            os.environ['HF_HOME'] = str(home_cache)
        except Exception as home_e:
            logger.error(f"❌ Could not create any cache directory: {home_e}")
    
    vector_store_path = Path("./data/chroma")
    
    # Case 1: Vector store exists - check compatibility
    if vector_store_path.exists() and any(vector_store_path.glob("*.sqlite3")):
        logger.info("📊 Vector store found, checking compatibility...")
        
        is_compatible, result = check_vector_store_compatibility()
        
        if is_compatible:
            logger.info(f"✅ Vector store is compatible with current embeddings")
        else:
            logger.info("🔄 Vector store incompatible - needs re-indexing")
            
            if result == "CACHE_ERROR":
                logger.info("⚠️  Cache/permission issue detected - attempting re-indexing anyway")
                # For cache errors, try re-indexing directly without clearing
                if reindex_if_needed():
                    logger.info("✅ Successfully re-indexed vector store despite cache issues")
                else:
                    logger.warning("⚠️  Could not re-index due to cache issues, starting API without RAG")
            else:
                logger.info("🗑️  Clearing existing vector store...")
                
                # Clear existing vector store
                import shutil
                try:
                    shutil.rmtree(vector_store_path)
                except PermissionError as e:
                    logger.warning(f"Permission denied clearing vector store: {e}. Attempting to continue...")
                    # Try to clear just the database file instead
                    try:
                        db_file = vector_store_path / "chroma.sqlite3"
                        if db_file.exists():
                            db_file.unlink()
                            logger.info("Cleared vector store database file")
                    except Exception as inner_e:
                        logger.error(f"Could not clear vector store: {inner_e}")
                        return False
                
                # Re-index if possible
                if reindex_if_needed():
                    logger.info("✅ Successfully re-indexed vector store")
                else:
                    logger.warning("⚠️  Could not re-index, starting API without RAG")
    
    # Case 2: No vector store - check for raw data to index
    else:
        logger.info("🆕 No vector store found - checking for existing data...")
        
        if reindex_if_needed():
            logger.info("✅ Successfully created vector store from existing data")
        else:
            logger.info("📭 No existing data found - API will work without RAG")
            logger.info("🌐 To enable RAG, run: python -m rag.ingest")
    
    logger.info("🎯 Startup checks completed - starting API server...")
    return 0

if __name__ == "__main__":
    sys.exit(main())