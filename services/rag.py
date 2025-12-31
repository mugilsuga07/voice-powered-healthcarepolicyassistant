"""
RAG (Retrieval-Augmented Generation) Service
Handles document retrieval and grounded answer generation.
"""

import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "insurance_policies"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.3


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with metadata."""
    content: str
    source: str
    page: int
    score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "score": self.score
        }


@dataclass
class RAGResult:
    """Result from RAG pipeline."""
    answer: str
    chunks: List[RetrievedChunk] = field(default_factory=list)
    latency_ms: int = 0
    success: bool = True
    error: Optional[str] = None
    was_refused: bool = False
    refusal_reason: Optional[str] = None


def clean_query(query: str) -> str:
    """
    Clean the query by removing conversational filler.
    This improves retrieval accuracy.
    """
    import re
    
    # Common conversational fillers to remove
    fillers = [
        r"^hey[,.]?\s*",
        r"^hi[,.]?\s*",
        r"^hello[,.]?\s*",
        r"^um+[,.]?\s*",
        r"^uh+[,.]?\s*",
        r"^so[,.]?\s*",
        r"^okay[,.]?\s*",
        r"^ok[,.]?\s*",
        r"^\w+\s+here[,.]?\s*",  # "Madonna here", "John here"
        r"^my name is \w+[,.]?\s*",
        r"^this is \w+[,.]?\s*",
        r"^i have a question[,.]?\s*",
        r"^i wanted to ask[,.]?\s*",
        r"^i need to know[,.]?\s*",
        r"^can you tell me[,.]?\s*",
        r"^could you tell me[,.]?\s*",
        r"^i would like to know[,.]?\s*",
    ]
    
    cleaned = query.strip()
    for filler in fillers:
        cleaned = re.sub(filler, "", cleaned, flags=re.IGNORECASE)
    
    # Clean up any double spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    if cleaned != query:
        logger.info(f"Query cleaned: '{query}' -> '{cleaned}'")
    
    return cleaned if cleaned else query


class RAGService:
    """
    Retrieval-Augmented Generation service for policy questions.
    """
    
    # Medical/out-of-scope topics to refuse
    BLOCKED_TOPICS = [
        "diagnos", "symptom", "treatment", "medication", "dosage",
        "prescri", "side effect", "drug interact", "medical advice",
        "should i take", "is it safe to", "cure", "disease",
        "prognosis", "therapy", "clinical", "doctor recommend"
    ]
    
    SYSTEM_PROMPT = """You are a helpful assistant for healthcare insurance call-center agents. 
You answer questions about insurance policy details based on the provided policy documents.

IMPORTANT RULES:
1. Extract relevant information from the provided excerpts, even if the format is imperfect
2. Policy excerpts may come from summary pages, example scenarios, or glossaries - use ALL available information
3. If you see dollar amounts, percentages, or coverage details, include them in your answer
4. NEVER provide medical advice - only administrative/policy information
5. Always cite the source document and page
6. Only say "I couldn't find this information" if there is truly NO relevant data in the excerpts

Be helpful and extract any relevant policy details you can find."""

    ANSWER_PROMPT = """Based on the following policy document excerpts, answer the agent's question.

POLICY EXCERPTS:
{context}

AGENT'S QUESTION: {question}

Instructions:
- Look for dollar amounts, percentages, and coverage details in the excerpts
- Information may appear in tables, examples, or summaries - use it all
- Cite the source document and page for each piece of information
- If multiple plans show different values, list them all
- Only say you couldn't find information if there is truly nothing relevant"""

    def __init__(self, persist_dir: Optional[Path] = None):
        """Initialize the RAG service."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        # Set persist directory
        if persist_dir is None:
            project_root = Path(__file__).parent.parent
            persist_dir = project_root / "data" / "chroma"
        
        self.persist_dir = persist_dir
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=500
        )
        
        # Load vector store if it exists
        self.vectorstore = None
        if persist_dir.exists():
            try:
                self.vectorstore = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=self.embeddings,
                    persist_directory=str(persist_dir)
                )
                logger.info(f"Loaded vector store from {persist_dir}")
            except Exception as e:
                logger.warning(f"Could not load vector store: {e}")
    
    def _is_medical_question(self, query: str) -> tuple[bool, str]:
        """
        Check if the query is asking for medical advice.
        
        Returns:
            Tuple of (is_blocked, reason)
        """
        query_lower = query.lower()
        
        for topic in self.BLOCKED_TOPICS:
            if topic in query_lower:
                return True, f"Query appears to be asking about medical topics ({topic})"
        
        return False, ""
    
    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format retrieved chunks into context string."""
        if not chunks:
            return "No relevant policy information found."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Source: {chunk.source}, Page {chunk.page}]\n{chunk.content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def retrieve(self, query: str, plan_filter: Optional[str] = None) -> List[RetrievedChunk]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: The search query
            plan_filter: Optional filename to filter results by specific plan
            
        Returns:
            List of retrieved chunks with scores, sorted by relevance (highest first)
        """
        if self.vectorstore is None:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            # Clean the query to remove conversational filler
            cleaned_query = clean_query(query)
            
            # Build filter if plan is specified
            search_kwargs = {"k": TOP_K * 2}  # Get more results to filter from
            
            if plan_filter:
                search_kwargs["filter"] = {"source": plan_filter}
                logger.info(f"Filtering by plan: {plan_filter}")
            
            # Search with scores using cleaned query
            results = self.vectorstore.similarity_search_with_score(
                cleaned_query,
                **search_kwargs
            )
            
            chunks = []
            for doc, score in results:
                # Convert distance to similarity (lower distance = higher similarity)
                # ChromaDB returns L2 distance, so we invert it
                similarity = 1 / (1 + score)
                
                if similarity >= SIMILARITY_THRESHOLD:
                    chunks.append(RetrievedChunk(
                        content=doc.page_content,
                        source=doc.metadata.get("source", "Unknown"),
                        page=doc.metadata.get("page", 0),
                        score=round(similarity, 3)
                    ))
            
            # Sort by score (highest first) and limit to TOP_K
            chunks.sort(key=lambda x: x.score, reverse=True)
            chunks = chunks[:TOP_K]
            
            logger.info(f"Retrieved {len(chunks)} relevant chunks (plan_filter={plan_filter})")
            return chunks
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def generate_answer(
        self,
        query: str,
        chunks: List[RetrievedChunk]
    ) -> str:
        """
        Generate a grounded answer based on retrieved chunks.
        
        Args:
            query: The user's question
            chunks: Retrieved document chunks
            
        Returns:
            Generated answer string
        """
        context = self._format_context(chunks)
        
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=self.ANSWER_PROMPT.format(
                context=context,
                question=query
            ))
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def query(self, query: str, plan_filter: Optional[str] = None) -> RAGResult:
        """
        Process a query through the full RAG pipeline.
        
        Args:
            query: The user's question (should be sanitized)
            plan_filter: Optional filename to filter results by specific plan
            
        Returns:
            RAGResult with answer, chunks, and metadata
        """
        start_time = time.time()
        
        try:
            # Check for medical/blocked topics
            is_blocked, reason = self._is_medical_question(query)
            if is_blocked:
                latency_ms = int((time.time() - start_time) * 1000)
                logger.warning(f"Query refused: {reason}")
                return RAGResult(
                    answer="I can only assist with administrative policy questions. For medical questions or advice, please direct the caller to their healthcare provider.",
                    chunks=[],
                    latency_ms=latency_ms,
                    success=True,
                    was_refused=True,
                    refusal_reason=reason
                )
            
            # Check if vector store is available
            if self.vectorstore is None:
                latency_ms = int((time.time() - start_time) * 1000)
                return RAGResult(
                    answer="Policy documents have not been ingested yet. Please run the document ingestion script first.",
                    chunks=[],
                    latency_ms=latency_ms,
                    success=False,
                    error="Vector store not initialized"
                )
            
            # Retrieve relevant chunks (with optional plan filter)
            chunks = self.retrieve(query, plan_filter=plan_filter)
            
            # Generate answer
            if not chunks:
                if plan_filter:
                    answer = f"I couldn't find relevant information in the selected plan ({plan_filter}). Try selecting 'All Plans' or rephrase your question."
                else:
                    answer = "I couldn't find relevant information in the policy documents for this question. Please try rephrasing or check with a supervisor."
            else:
                answer = self.generate_answer(query, chunks)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return RAGResult(
                answer=answer,
                chunks=chunks,
                latency_ms=latency_ms,
                success=True
            )
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"RAG query error: {e}")
            return RAGResult(
                answer="An error occurred while processing your question. Please try again.",
                chunks=[],
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )


# Global service instance
_rag_service = None


def get_rag_service() -> RAGService:
    """Get or create the global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        logger.info("Creating new RAGService instance...")
        _rag_service = RAGService()
    return _rag_service


def reset_rag_service():
    """Reset the RAG service (useful for re-initialization)."""
    global _rag_service
    _rag_service = None
    logger.info("RAG service reset")


def query_rag(query: str, plan_filter: Optional[str] = None) -> RAGResult:
    """
    Convenience function to query the RAG pipeline.
    
    Args:
        query: The user's question (should be sanitized)
        plan_filter: Optional filename to filter results by specific plan
        
    Returns:
        RAGResult with answer and metadata
    """
    logger.info(f"query_rag called with query='{query[:50]}...', plan_filter={plan_filter}")
    service = get_rag_service()
    
    if service.vectorstore is None:
        logger.error("Vector store is None! Documents may not be ingested.")
        return RAGResult(
            answer="The policy documents have not been loaded. Please restart the application or run document ingestion.",
            chunks=[],
            latency_ms=0,
            success=False,
            error="Vector store not loaded"
        )
    
    result = service.query(query, plan_filter=plan_filter)
    logger.info(f"query_rag result: success={result.success}, chunks={len(result.chunks)}, answer_len={len(result.answer)}")
    return result

