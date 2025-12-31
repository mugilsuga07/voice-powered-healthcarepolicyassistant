"""
Document Ingestion Script
Processes PDF policy documents and stores embeddings in ChromaDB.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import fitz  # PyMuPDF
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
COLLECTION_NAME = "insurance_policies"


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text with page numbers from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of page dictionaries with content and metadata
    """
    logger.info(f"Extracting text from: {pdf_path.name}")
    
    doc = fitz.open(str(pdf_path))
    pages = []
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "content": text,
                "page": page_num,
                "source": pdf_path.name,
                "source_path": str(pdf_path)
            })
    
    doc.close()
    logger.info(f"  Extracted {len(pages)} pages from {pdf_path.name}")
    return pages


def chunk_documents(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split pages into semantic chunks with metadata.
    
    Args:
        pages: List of page dictionaries
        
    Returns:
        List of chunk dictionaries with content and metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len
    )
    
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["content"])
        
        for i, split in enumerate(splits):
            if split.strip():  # Skip empty chunks
                chunks.append({
                    "content": split.strip(),
                    "metadata": {
                        "source": page["source"],
                        "page": page["page"],
                        "chunk_index": i,
                        "total_chunks_in_page": len(splits)
                    }
                })
    
    return chunks


def ingest_documents(
    policies_dir: Path,
    persist_dir: Path,
    force_reingest: bool = False
) -> Dict[str, Any]:
    """
    Ingest all PDF documents from the policies directory.
    
    Args:
        policies_dir: Directory containing PDF files
        persist_dir: Directory to persist ChromaDB
        force_reingest: If True, delete existing data and re-ingest
        
    Returns:
        Dictionary with ingestion statistics
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    # Find all PDF files
    pdf_files = list(policies_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {policies_dir}")
        return {"status": "no_files", "documents": 0, "chunks": 0}
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    
    # Extract and chunk all documents
    all_chunks = []
    document_stats = []
    
    for pdf_file in pdf_files:
        pages = extract_text_from_pdf(pdf_file)
        chunks = chunk_documents(pages)
        all_chunks.extend(chunks)
        
        document_stats.append({
            "file": pdf_file.name,
            "pages": len(pages),
            "chunks": len(chunks)
        })
        
        logger.info(f"  Created {len(chunks)} chunks from {pdf_file.name}")
    
    if not all_chunks:
        logger.warning("No chunks extracted from documents")
        return {"status": "no_chunks", "documents": len(pdf_files), "chunks": 0}
    
    # Prepare for embedding
    texts = [chunk["content"] for chunk in all_chunks]
    metadatas = [chunk["metadata"] for chunk in all_chunks]
    
    # Create embeddings and store in ChromaDB
    logger.info(f"Creating embeddings for {len(texts)} chunks...")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create or update vector store
    if force_reingest and persist_dir.exists():
        import shutil
        shutil.rmtree(persist_dir)
        logger.info("Cleared existing vector store")
    
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    vectorstore = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir)
    )
    
    logger.info(f"✅ Successfully ingested {len(texts)} chunks into ChromaDB")
    
    # Save ingestion manifest
    manifest = {
        "ingested_at": datetime.now().isoformat(),
        "total_documents": len(pdf_files),
        "total_chunks": len(all_chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "documents": document_stats
    }
    
    manifest_path = persist_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Manifest saved to {manifest_path}")
    
    return {
        "status": "success",
        "documents": len(pdf_files),
        "chunks": len(all_chunks),
        "manifest": manifest
    }


def main():
    """Main entry point for document ingestion."""
    project_root = get_project_root()
    policies_dir = project_root / "data" / "policies"
    persist_dir = project_root / "data" / "chroma"
    
    # Check if policies directory exists
    if not policies_dir.exists():
        policies_dir.mkdir(parents=True, exist_ok=True)
        logger.error(f"Created empty policies directory: {policies_dir}")
        logger.error("Please add PDF files to this directory and run again.")
        return
    
    # Run ingestion
    try:
        result = ingest_documents(
            policies_dir=policies_dir,
            persist_dir=persist_dir,
            force_reingest=True  # Always fresh ingest for now
        )
        
        if result["status"] == "success":
            print("\n" + "="*50)
            print("✅ INGESTION COMPLETE")
            print("="*50)
            print(f"Documents processed: {result['documents']}")
            print(f"Chunks created: {result['chunks']}")
            print(f"Vector store: {persist_dir}")
            print("="*50)
        else:
            print(f"\n⚠️ Ingestion incomplete: {result['status']}")
            
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


if __name__ == "__main__":
    main()

