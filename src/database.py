"""
Vector database module using Qdrant and local HuggingFace embeddings.
Handles vector collection initialization, payload indexing, document ingestion, and filtered searches.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config import settings
from src.parser import DocumentChunk


class VectorStoreManager:
    """Manages document embeddings and vector storage in Qdrant."""

    def __init__(self):
        # 1. Initialize local embedding model (Runs locally for $0)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name
        )

        # 2. Connect to Qdrant (Cloud URL if configured, otherwise local disk storage)
        if settings.qdrant_url and settings.qdrant_api_key:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key
            )
        else:
            local_storage_path = Path(settings.raw_data_dir).parent / "qdrant_storage"
            self.client = QdrantClient(path=str(local_storage_path))

        self.collection_name = settings.collection_name
        self._ensure_collection_and_indexes()

    def _ensure_collection_and_indexes(self):
        """Creates the collection and ensures payload indexes exist for metadata filtering."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=384,
                    distance=rest.Distance.COSINE
                )
            )

        # Create payload indexes required by Qdrant for filtering
        indexes_to_create = [
            ("ticker", rest.PayloadSchemaType.KEYWORD),
            ("fiscal_period", rest.PayloadSchemaType.KEYWORD),
            ("type", rest.PayloadSchemaType.KEYWORD),
            ("fiscal_year", rest.PayloadSchemaType.INTEGER),
        ]

        for field_name, field_schema in indexes_to_create:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema
                )
            except Exception:
                # Index may already exist
                pass

    def insert_chunks(self, chunks: List[DocumentChunk]):
        """Embeds and inserts structured document chunks into Qdrant."""
        if not chunks:
            return

        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Generate embeddings
        vectors = self.embeddings.embed_documents(texts)

        # Prepare Qdrant points
        points = []
        for i, (text, vector, meta) in enumerate(zip(texts, vectors, metadatas)):
            payload = {**meta, "page_content": text}
            points.append(
                rest.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                )
            )

        # Upload in batches
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search_filtered(
        self,
        query: str,
        ticker: str,
        fiscal_year: int,
        fiscal_period: str,
        limit: int = 4
    ) -> List[Document]:
        """
        Executes a similarity search strictly constrained by ticker, year, and quarter.
        """
        query_vector = self.embeddings.embed_query(query)

        # Construct strict metadata filter
        conditions = [
            rest.FieldCondition(
                key="ticker",
                match=rest.MatchValue(value=ticker.upper())
            ),
            rest.FieldCondition(
                key="fiscal_year",
                match=rest.MatchValue(value=fiscal_year)
            ),
            rest.FieldCondition(
                key="fiscal_period",
                match=rest.MatchValue(value=fiscal_period.upper())
            )
        ]

        query_filter = rest.Filter(must=conditions)

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit
            )
            points = response.points
        else:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit
            )

        documents = []
        for hit in points:
            content = hit.payload.get("page_content", "")
            meta = {k: v for k, v in hit.payload.items() if k != "page_content"}
            documents.append(Document(page_content=content, metadata=meta))

        return documents