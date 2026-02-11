import logging
import os
from pathlib import Path
from typing import Any

import chromadb  # type: ignore
from chromadb.api.models.Collection import Collection  # type: ignore

from .schemas import LectureChunk, QueryFilter

logger = logging.getLogger(__name__)

COLLECTION_NAME = "lecture_chunks"
DEFAULT_PERSIST_PATH = ".chroma_data"


class ChromaVectorStore:
    _persist_path: str
    _client: Any

    def __init__(self, persist_path: str | None = None):
        self._persist_path = persist_path or os.getenv(
            "CHROMA_PERSIST_PATH", DEFAULT_PERSIST_PATH
        )

        Path(self._persist_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self._persist_path)

        logger.info(f"Initialized ChromaDB client at {self._persist_path}")

    def create_or_get_collection(self) -> Collection:
        try:
            collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Lecture transcript chunks for RAG"},
            )

            logger.info(
                f"Collection '{COLLECTION_NAME}' ready with {collection.count()} vectors"
            )

            return collection
        except Exception as e:
            logger.error(f"Failed to create/get collection: {e}")
            raise

    def upsert_chunks(
        self,
        course_id: int,
        lecture_id: int,
        chunks: list[LectureChunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            logger.warning(
                f"No chunks provided for course_id={course_id}, lecture_id={lecture_id}"
            )
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})"
            )

        collection = self.create_or_get_collection()

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_metadata() for chunk in chunks]

        try:
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            logger.info(
                f"Upserted {len(chunks)} chunks for course_id={course_id}, lecture_id={lecture_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to upsert chunks for course_id={course_id}, lecture_id={lecture_id}: {e}"
            )
            raise

    def query_by_lecture(
        self,
        query_embedding: list[float],
        filter: QueryFilter,
        top_k: int = 5,
    ) -> dict[str, list[Any]]:
        collection = self.create_or_get_collection()

        where_clause = filter.to_where_clause()

        if where_clause is None:
            logger.warning("Query filter is empty - retrieving from all lectures")

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
            )

            logger.info(
                f"Retrieved {len(results['documents'][0])} results for filter={where_clause}, top_k={top_k}"
            )

            return results
        except Exception as e:
            logger.error(f"Failed to query with filter={where_clause}: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def delete_lecture_chunks(self, course_id: int, lecture_id: int) -> None:
        collection = self.create_or_get_collection()

        # lecture_id is globally unique; single-operator where satisfies Chroma syntax
        where_clause: dict[str, Any] = {"lecture_id": {"$eq": lecture_id}}

        try:
            collection.delete(where=where_clause)

            logger.info(
                f"Deleted chunks for course_id={course_id}, lecture_id={lecture_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to delete chunks for course_id={course_id}, lecture_id={lecture_id}: {e}"
            )
            raise

    def get_collection_stats(self) -> dict[str, int]:
        collection = self.create_or_get_collection()

        return {"total_vectors": collection.count()}
