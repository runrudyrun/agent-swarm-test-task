"""RAG ingestion pipeline for InfinitePay content."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from .config import RAGConfig, get_embeddings

logger = logging.getLogger(__name__)


class WebContentFetcher:
    """Fetch and clean web content."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        
    async def fetch_url(self, url: str) -> Optional[dict]:
        """Fetch and parse a single URL."""
        try:
            logger.info(f"Fetching: {url}")
            response = await self.client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Get title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else url
            
            # Get meta description
            description = ""
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                description = meta_desc.get('content', '')
            
            return {
                'url': url,
                'title': title_text,
                'description': description,
                'content': content,
                'scraped_at': asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML."""
        # Try to find main content areas
        main_content = ""
        
        # Look for common content containers
        content_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.content',
            '.main-content',
            '#content',
            '#main-content'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = elements[0].get_text(separator=' ', strip=True)
                break
        
        # Fallback to body content if no main content found
        if not main_content:
            body = soup.find('body')
            if body:
                main_content = body.get_text(separator=' ', strip=True)
        
        return main_content
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class DocumentProcessor:
    """Process fetched content into documents."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAGConfig.CHUNK_SIZE,
            chunk_overlap=RAGConfig.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
    
    def create_documents(self, fetched_data: List[dict]) -> List[Document]:
        """Create LangChain documents from fetched data."""
        documents = []
        
        for data in fetched_data:
            if not data or not data.get('content'):
                continue
            
            # Create a comprehensive text for the document
            full_text = f"""Title: {data['title']}
URL: {data['url']}
Description: {data['description']}

Content:
{data['content']}"""
            
            # Create metadata
            metadata = {
                'source': data['url'],
                'title': data['title'],
                'description': data['description'],
                'scraped_at': data.get('scraped_at', 0)
            }
            
            # Split into chunks
            chunks = self.text_splitter.split_text(full_text)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = i
                chunk_metadata['total_chunks'] = len(chunks)
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
        
        return documents


class VectorStoreManager:
    """Manage vector store operations."""
    
    def __init__(self):
        self.embeddings = get_embeddings()
        self.persist_directory = RAGConfig.VECTOR_STORE_PATH
        
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """Create a new vector store from documents."""
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        # Create directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=RAGConfig.COLLECTION_NAME
        )
        
        return vectorstore
    
    def load_vectorstore(self) -> Optional[Chroma]:
        """Load existing vector store."""
        try:
            vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=RAGConfig.COLLECTION_NAME
            )
            
            # Test if collection exists
            try:
                vectorstore._collection.count()
                return vectorstore
            except Exception:
                return None
                
        except Exception:
            return None


async def ingest_infinitepay_content():
    """Main ingestion function."""
    logger.info("Starting InfinitePay content ingestion")
    
    # Initialize components
    fetcher = WebContentFetcher()
    processor = DocumentProcessor()
    vector_manager = VectorStoreManager()
    
    try:
        # Fetch all URLs
        fetched_data = []
        for url in RAGConfig.INFINITEPAY_URLS:
            data = await fetcher.fetch_url(url)
            if data:
                fetched_data.append(data)
            else:
                logger.warning(f"Failed to fetch: {url}")
        
        logger.info(f"Successfully fetched {len(fetched_data)} URLs")
        
        # Save raw data
        raw_data_path = Path("data/raw")
        raw_data_path.mkdir(parents=True, exist_ok=True)
        
        for data in fetched_data:
            filename = data['url'].replace('https://', '').replace('/', '_') + '.json'
            filepath = raw_data_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Process into documents
        documents = processor.create_documents(fetched_data)
        logger.info(f"Created {len(documents)} document chunks")
        
        # Create vector store
        vectorstore = vector_manager.create_vectorstore(documents)
        logger.info("Vector store created successfully")
        
        # Save URL sources
        sources_path = Path("data/sources")
        sources_path.mkdir(parents=True, exist_ok=True)
        
        with open(sources_path / "urls.txt", "w", encoding="utf-8") as f:
            f.write("InfinitePay Content Sources\n")
            f.write("=" * 50 + "\n\n")
            for data in fetched_data:
                f.write(f"Title: {data['title']}\n")
                f.write(f"URL: {data['url']}\n")
                f.write(f"Description: {data['description']}\n")
                f.write("-" * 30 + "\n")
        
        return len(documents)
        
    finally:
        await fetcher.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = asyncio.run(ingest_infinitepay_content())
    print(f"Ingestion completed. Created {result} document chunks.")