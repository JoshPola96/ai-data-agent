# app/core/dependencies.py

"""
Global Service Locator
Instantiates singletons to avoid circular imports
"""

from app.core.vectorstore import VectorStore
from app.core.session import SessionStore
from app.services.llm import LLMService
from app.services.ingest import IngestionService
from app.services.retrieval import HybridRetriever
from app.services.chart import ChartService

# Initialize Global Instances
vector_store = VectorStore()
session_store = SessionStore()
llm_service = LLMService()
chart_service = ChartService()

# Services that depend on the stores
ingestion_service = IngestionService(vector_store, session_store)
retriever = HybridRetriever(vector_store)
