import glob
import json
import uuid

from qdrant_client import QdrantClient, models

from app.config import settings
from app.services.retrieval.embedding import (
    embed_documents,
    get_embedding_dim,
    get_active_model_type,
)


TARGET_COLLECTION = settings.QDRANT_COLLECTION
EXPECTED_DIMENSION = 768
EXPECTED_MODEL = "huggingface:sentence-transformers:BAAI/bge-base-en-v1.5"
BATCH_SIZE = 64


def main():
    print("=" * 70)
    print("KnowledgeMesh — Regenerated JSON → Qdrant migration")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Validate active embedding configuration
    # ---------------------------------------------------------------
    active_dimension = get_embedding_dim()
    active_model = get_active_model_type()

    if active_dimension != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"Active embedding dimension is {active_dimension}, "
            f"expected {EXPECTED_DIMENSION}."
        )

    if active_model != EXPECTED_MODEL:
        raise RuntimeError(
            f"Active embedding model is {active_model}, "
            f"expected {EXPECTED_MODEL}."
        )

    print(f"Embedding model = {active_model}")
    print(f"Embedding dim   = {active_dimension}")
    print(f"Batch size      = {BATCH_SIZE}")

    # ---------------------------------------------------------------
    # 2. Load validated regenerated JSON
    # ---------------------------------------------------------------
    files = sorted(
        glob.glob(r".\processed_data\true\*.json")
        + glob.glob(r".\processed_data\noisy\*.json")
    )

    documents = []
    pairs = set()

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        document_id = data["document_id"]
        filename = data["filename"]
        source_type = data["source_type"]
        content_hash = data["content_hash"]
        ingestion_timestamp = data["ingested_at"]
        chunks = data["chunks"]

        for chunk in chunks:
            chunk_id = int(chunk["chunk_id"])
            text = chunk["text"]

            key = (document_id, chunk_id)

            if key in pairs:
                raise RuntimeError(
                    f"Duplicate document/chunk pair: {key}"
                )

            pairs.add(key)

            documents.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "source_type": source_type,
                    "content_hash": content_hash,
                    "ingested_at": ingestion_timestamp,
                    "chunk_id": chunk_id,
                    "text": text,
                    "total_chunks": len(chunks),
                }
            )

    unique_documents = len(
        set(d["document_id"] for d in documents)
    )

    print()
    print(f"JSON files       = {len(files)}")
    print(f"Documents        = {unique_documents}")
    print(f"Chunks           = {len(documents)}")
    print(f"Unique pairs     = {len(pairs)}")

    if len(files) != 12:
        raise RuntimeError(
            f"Expected 12 JSON files, found {len(files)}."
        )

    if unique_documents != 12:
        raise RuntimeError(
            f"Expected 12 document IDs, found {unique_documents}."
        )

    if len(documents) != 1101:
        raise RuntimeError(
            f"Expected 1101 chunks, found {len(documents)}."
        )

    if len(pairs) != 1101:
        raise RuntimeError(
            f"Expected 1101 unique pairs, found {len(pairs)}."
        )

    # ---------------------------------------------------------------
    # 3. Validate Qdrant target
    # ---------------------------------------------------------------
    client = QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
    )

    collection = client.get_collection(TARGET_COLLECTION)

    actual_dimension = collection.config.params.vectors.size

    print()
    print(f"Collection       = {TARGET_COLLECTION}")
    print(f"Points before    = {collection.points_count}")
    print(f"Dimension        = {actual_dimension}")
    print(
        f"Distance         = "
        f"{collection.config.params.vectors.distance}"
    )

    if actual_dimension != EXPECTED_DIMENSION:
        raise RuntimeError(
            f"Qdrant dimension {actual_dimension} "
            f"!= {EXPECTED_DIMENSION}."
        )

    # ---------------------------------------------------------------
    # 4. Batch embedding + upsert
    # ---------------------------------------------------------------
    total = len(documents)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = documents[start:end]

        print()
        print(
            f"[{end:04d}/{total:04d}] "
            f"Embedding chunks {start}-{end - 1}..."
        )

        texts = [d["text"] for d in batch]

        embeddings = embed_documents(texts)

        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding count {len(embeddings)} "
                f"!= batch size {len(batch)}."
            )

        points = []

        for document, vector in zip(batch, embeddings):
            if len(vector) != EXPECTED_DIMENSION:
                raise RuntimeError(
                    f"Vector dimension {len(vector)} "
                    f"!= {EXPECTED_DIMENSION}."
                )

            document_id = document["document_id"]
            chunk_id = document["chunk_id"]

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{document_id}-{chunk_id}",
                )
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": document["text"],
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "total_chunks": document["total_chunks"],
                        "source": document["filename"],
                        "source_type": document["source_type"],
                        "content_hash": document["content_hash"],
                        "ingested_at": document["ingested_at"],
                        "embedding_model": EXPECTED_MODEL,
                        "embedding_dimension": len(vector),
                        "character_count": len(document["text"]),
                    },
                )
            )

        client.upsert(
            collection_name=TARGET_COLLECTION,
            points=points,
            wait=True,
        )

        print(f"        Upserted {len(points)} points.")

    # ---------------------------------------------------------------
    # 5. Final collection verification
    # ---------------------------------------------------------------
    collection = client.get_collection(TARGET_COLLECTION)

    print()
    print("=" * 70)
    print("Migration complete")
    print("=" * 70)
    print(f"Collection       = {TARGET_COLLECTION}")
    print(f"Points after     = {collection.points_count}")
    print(f"Expected         = {total}")

    if collection.points_count != total:
        raise RuntimeError(
            f"Migration verification failed: "
            f"expected {total} points, "
            f"got {collection.points_count}."
        )

    print()
    print("STATUS: PASS")
    print("Old enterprise_rag collection was NOT modified.")


if __name__ == "__main__":
    main()
