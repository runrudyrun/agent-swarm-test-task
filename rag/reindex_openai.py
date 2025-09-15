"""Re-index existing documents with OpenAI embeddings."""

import json
import logging
import os
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from .config import RAGConfig, get_embeddings, get_chroma_settings

logger = logging.getLogger(__name__)


def load_existing_documents(data_dir: str = "./data/raw") -> List[Document]:
    """Load existing scraped documents from JSON files."""
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Data directory {data_dir} does not exist")
        return documents
    
    # Load all JSON files from the raw data directory
    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Create a Document from the JSON data
            content = data.get('content', '')
            if content:
                # Create metadata
                metadata = {
                    'source': data.get('url', ''),
                    'title': data.get('title', ''),
                    'description': data.get('description', ''),
                    'scraped_at': data.get('scraped_at', '')
                }
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
                logger.info(f"Loaded document from {json_file}: {data.get('title', 'Unknown')}")
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents from {data_dir}")
    return documents


def reindex_with_openai():
    """Re-index documents with OpenAI embeddings."""
    logger.info("Starting re-indexing with OpenAI embeddings...")
    
    # Clear existing vector store
    vector_store_path = Path(RAGConfig.VECTOR_STORE_PATH)
    if vector_store_path.exists():
        logger.info(f"Clearing existing vector store at {vector_store_path}")
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
                raise
    
    # Load existing documents
    documents = load_existing_documents()
    if not documents:
        logger.error("No documents found to re-index")
        return
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAGConfig.CHUNK_SIZE,
        chunk_overlap=RAGConfig.CHUNK_OVERLAP,
        length_function=len,
    )
    
    # Split documents into chunks
    logger.info("Splitting documents into chunks...")
    splits = text_splitter.split_documents(documents)
    logger.info(f"Created {len(splits)} document chunks")
    
    # Get OpenAI embeddings
    logger.info("Getting OpenAI embeddings...")
    embeddings = get_embeddings()
    
    # Create new vector store with OpenAI embeddings
    logger.info("Creating vector store with OpenAI embeddings...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=RAGConfig.VECTOR_STORE_PATH,
        collection_name=RAGConfig.COLLECTION_NAME,
        client_settings=get_chroma_settings(),
    )
    
    # Persist the vector store
    vectorstore.persist()
    
    logger.info(f"Successfully re-indexed {len(splits)} document chunks with OpenAI embeddings")
    logger.info(f"Vector store saved to {RAGConfig.VECTOR_STORE_PATH}")
    
    # Verify the vector store
    try:
        count = vectorstore._collection.count()
        logger.info(f"Vector store contains {count} documents")
    except Exception as e:
        logger.warning(f"Could not verify document count: {e}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run re-indexing
    reindex_with_openai()