"""
KnowledgeMesh universal document ingestion pipeline.

Pipeline:
    Parse
      ↓
    Chunk
      ↓
    Save processed JSON
      ↓
    Generate Hugging Face embeddings
      ↓
    Validate embedding dimensions
      ↓
    Create Qdrant points
      ↓
    Upsert into Qdrant

Supported source types:
    true
    noisy
    general

CLI:
    python -m app.ingestion.processor DATA
    python -m app.ingestion.processor DATA --wipe
    python -m app.ingestion.processor DATA/true_data true
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings

from app.services.retrieval.embedding import (
    embed_texts,
    get_embedding_dim,
    get_active_model_type,
)

from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

logfire.configure(service_name="enterprise-ingestion-service")

PROCESSED_DATA_DIR = "processed_data"

DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 150


# ---------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------

qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def get_timestamp() -> str:
    """Return a UTC ISO-8601 timestamp."""

    return datetime.now(timezone.utc).isoformat()


def create_document_id(
    file_path: str,
    filename: str,
    source_type: str,
) -> str:
    """
    Generate a deterministic document ID.

    The same file path + filename + source type produces
    the same document ID.
    """

    identity = f"{os.path.abspath(file_path)}::{filename}::{source_type}"

    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def create_content_hash(text: str) -> str:
    """Generate a deterministic hash of document content."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_processed_locally(
    data: dict,
    source_type: str,
    filename: str,
) -> str:
    """
    Save parsed + chunked metadata as JSON.

    Output:
        processed_data/
            <source_type>/
                <filename>.json
    """

    folder = os.path.join(
        PROCESSED_DATA_DIR,
        source_type,
    )

    os.makedirs(
        folder,
        exist_ok=True,
    )

    safe_filename = os.path.basename(filename)

    destination = os.path.join(
        folder,
        f"{safe_filename}.json",
    )

    with open(
        destination,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return destination


# ---------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------


def extract_text(
    file_path: str,
    filename: str,
) -> Optional[str]:
    """Select the appropriate parser based on file extension."""

    extension = filename.lower().rsplit(".", 1)[-1]

    with logfire.span(
        "📄 Document Parsing",
        filename=filename,
        extension=extension,
    ):
        if extension == "pdf":
            return parse_pdf(file_path)

        if extension in ("html", "htm"):
            return parse_html(file_path)

        if extension == "txt":
            return parse_text(file_path)

        if extension in ("docx", "pptx"):
            from app.ingestion.loaders.office import parse_office

            return parse_office(file_path)

        logfire.warning(
            "Unsupported file type — skipping",
            filename=filename,
            extension=extension,
        )

        return None


# ---------------------------------------------------------------------
# Embedding / Qdrant validation
# ---------------------------------------------------------------------


def validate_collection_dimension() -> int:
    """
    Validate that the active embedding model dimension matches
    the configured Qdrant collection.

    Returns:
        Expected Qdrant vector dimension.
    """

    collection_name = settings.QDRANT_COLLECTION

    print(f"🔎 Validating Qdrant collection: {collection_name}")

    collection_info = qdrant_client.get_collection(collection_name)

    vectors_config = collection_info.config.params.vectors

    expected_dimension = vectors_config.size

    actual_dimension = get_embedding_dim()

    print(f"   Qdrant dimension : {expected_dimension}")
    print(f"   Model dimension  : {actual_dimension}")

    if actual_dimension != expected_dimension:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"model '{get_active_model_type()}' produces "
            f"{actual_dimension} dimensions, but Qdrant "
            f"collection '{collection_name}' expects "
            f"{expected_dimension} dimensions."
        )

    print("   ✅ Embedding dimension validated")

    return expected_dimension


# ---------------------------------------------------------------------
# Single file processing
# ---------------------------------------------------------------------


def process_file(
    file_path: str,
    filename: str,
    source_type: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict:
    """
    Complete ingestion pipeline for ONE document.
    """

    document_id = create_document_id(
        file_path=file_path,
        filename=filename,
        source_type=source_type,
    )

    started_at = get_timestamp()

    result = {
        "filename": filename,
        "source_type": source_type,
        "document_id": document_id,
        "embedding_model": None,
        "embedding_dimension": None,
        "status": "failed",
        "chunks": 0,
        "vectors": 0,
        "processed_path": None,
        "error": None,
        "started_at": started_at,
        "completed_at": None,
    }

    print(f"\n📄 Processing: {filename}")

    with logfire.span(
        "📦 Processing File",
        filename=filename,
        source_type=source_type,
        document_id=document_id,
    ):
        try:
            # ---------------------------------------------------------
            # 1. Extract
            # ---------------------------------------------------------

            print("   1️⃣ Extracting text...")

            full_text = extract_text(
                file_path=file_path,
                filename=filename,
            )

            if not full_text or not full_text.strip():
                logfire.warning(
                    "No text extracted — skipping document",
                    filename=filename,
                )

                print("   ⚠️ No text extracted — skipped")

                result["status"] = "skipped"
                result["completed_at"] = get_timestamp()

                return result

            full_text = full_text.strip()

            print(f"   ✅ Extracted {len(full_text):,} characters")

            content_hash = create_content_hash(full_text)

            # ---------------------------------------------------------
            # 2. Chunk
            # ---------------------------------------------------------

            print("   2️⃣ Chunking document...")

            with logfire.span(
                "✂️ Chunk Document",
                filename=filename,
            ):
                chunks = chunk_text(
                    full_text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            if not chunks:
                logfire.warning(
                    "No chunks generated — skipping document",
                    filename=filename,
                )

                print("   ⚠️ No chunks generated — skipped")

                result["status"] = "skipped"
                result["completed_at"] = get_timestamp()

                return result

            print(f"   ✅ Generated {len(chunks)} chunks")

            # ---------------------------------------------------------
            # 3. Build processed metadata
            # ---------------------------------------------------------

            print("   3️⃣ Building metadata...")

            ingestion_timestamp = get_timestamp()

            embedding_model = get_active_model_type()
            embedding_dimension = get_embedding_dim()

            processed_data = {
                "document_id": document_id,
                "filename": filename,
                "source_type": source_type,
                "content_hash": content_hash,
                "ingested_at": ingestion_timestamp,
                "embedding": {
                    "model": embedding_model,
                    "dimension": embedding_dimension,
                },
                "chunking": {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "total_chunks": len(chunks),
                },
                "chunks": [
                    {
                        "chunk_id": index,
                        "text": chunk,
                        "character_count": len(chunk),
                    }
                    for index, chunk in enumerate(chunks)
                ],
            }

            result["embedding_model"] = embedding_model
            result["embedding_dimension"] = embedding_dimension

            # ---------------------------------------------------------
            # 4. Save processed JSON
            # ---------------------------------------------------------

            print("   4️⃣ Saving processed JSON...")

            processed_path = save_processed_locally(
                processed_data,
                source_type,
                filename,
            )

            result["processed_path"] = processed_path

            logfire.info(
                "💾 Saved processed document",
                path=processed_path,
                chunks=len(chunks),
            )

            print(f"   ✅ Saved: {processed_path}")

            # ---------------------------------------------------------
            # 5. Generate embeddings
            # ---------------------------------------------------------

            print(f"   5️⃣ Generating embeddings for {len(chunks)} chunks...")

            with logfire.span(
                "🧠 Generate Embeddings",
                chunks=len(chunks),
                model=embedding_model,
                dimension=embedding_dimension,
            ):
                embeddings = embed_texts(chunks)

            if len(embeddings) != len(chunks):
                raise RuntimeError(
                    "Embedding count does not match "
                    "chunk count: "
                    f"{len(embeddings)} embeddings for "
                    f"{len(chunks)} chunks."
                )

            print(f"   ✅ Generated {len(embeddings)} embeddings")

            # ---------------------------------------------------------
            # 6. Validate embedding dimensions
            # ---------------------------------------------------------

            print("   6️⃣ Validating vector dimensions...")

            expected_dimension = validate_collection_dimension()

            invalid_dimensions = [
                len(vector)
                for vector in embeddings
                if len(vector) != expected_dimension
            ]

            if invalid_dimensions:
                raise RuntimeError(
                    "Embedding dimension mismatch: "
                    f"Qdrant expects {expected_dimension}, "
                    f"but received dimensions such as "
                    f"{invalid_dimensions[:5]}."
                )

            print("   ✅ Vector dimensions valid")

            # ---------------------------------------------------------
            # 7. Build Qdrant points
            # ---------------------------------------------------------

            print(f"   7️⃣ Building {len(chunks)} Qdrant points...")

            with logfire.span(
                "📍 Build Qdrant Points",
                points=len(chunks),
            ):
                points: List[models.PointStruct] = []

                for index, (
                    chunk,
                    vector,
                ) in enumerate(zip(chunks, embeddings)):
                    point_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{document_id}-{index}",
                        )
                    )

                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "text": chunk,
                                "document_id": document_id,
                                "chunk_id": index,
                                "total_chunks": len(chunks),
                                "source": filename,
                                "source_type": source_type,
                                "content_hash": content_hash,
                                "ingested_at": ingestion_timestamp,
                                "embedding_model": embedding_model,
                                "embedding_dimension": len(vector),
                                "character_count": len(chunk),
                            },
                        )
                    )

            print(f"   ✅ Built {len(points)} points")

            # ---------------------------------------------------------
            # 8. Upsert into Qdrant
            # ---------------------------------------------------------

            print("   8️⃣ Upserting into Qdrant...")

            try:
                with logfire.span(
                    "📡 Qdrant Upsert",
                    collection=settings.QDRANT_COLLECTION,
                    points=len(points),
                ):
                    qdrant_client.upsert(
                        collection_name=(settings.QDRANT_COLLECTION),
                        points=points,
                    )

            except Exception as qdrant_exc:
                print("\n🚨 QDRANT UPSERT FAILED")

                print(f"   Type: {type(qdrant_exc).__name__}")

                print(f"   Error: {qdrant_exc}")

                raise

            print(f"   ✅ Upserted {len(points)} vectors")

            # ---------------------------------------------------------
            # 9. Success
            # ---------------------------------------------------------

            result["status"] = "success"
            result["chunks"] = len(chunks)
            result["vectors"] = len(points)
            result["completed_at"] = get_timestamp()

            logfire.info(
                "✅ Document successfully indexed",
                filename=filename,
                document_id=document_id,
                chunks=len(chunks),
                vectors=len(points),
                model=embedding_model,
                dimension=embedding_dimension,
            )

            print(f"   🎉 SUCCESS: {filename}")

            return result

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

            result["status"] = "failed"
            result["error"] = error_message
            result["completed_at"] = get_timestamp()

            logfire.error(
                "❌ Document processing failed",
                filename=filename,
                document_id=document_id,
                error=error_message,
            )

            print(f"\n❌ FAILED: {filename}")

            print(f"   Error: {error_message}")

            return result


# ---------------------------------------------------------------------
# Directory processing
# ---------------------------------------------------------------------


def process_directory(
    dir_path: str,
    source_type: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict:
    """
    Process every supported file inside a directory.

    Returns an ingestion summary.
    """

    summary = {
        "directory": dir_path,
        "source_type": source_type,
        "embedding_model": None,
        "embedding_dimension": None,
        "total_files": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "total_chunks": 0,
        "total_vectors": 0,
        "files": [],
    }

    print(f"\n📂 Scanning directory: {dir_path}")

    with logfire.span(
        "📂 Scanning Directory",
        path=dir_path,
        source_type=source_type,
    ):
        if not os.path.isdir(dir_path):
            logfire.error(
                "Directory does not exist",
                path=dir_path,
            )

            return summary

        files = sorted(
            f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))
        )

        summary["total_files"] = len(files)

        logfire.info(
            "Files discovered",
            count=len(files),
            directory=dir_path,
        )

        print(f"   📄 Files discovered: {len(files)}")

        for filename in files:
            file_path = os.path.join(
                dir_path,
                filename,
            )

            result = process_file(
                file_path=file_path,
                filename=filename,
                source_type=source_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            summary["files"].append(result)

            if result["status"] == "success":
                summary["successful"] += 1

                summary["total_chunks"] += result["chunks"]

                summary["total_vectors"] += result["vectors"]

                if summary["embedding_model"] is None:
                    summary["embedding_model"] = result["embedding_model"]

                if summary["embedding_dimension"] is None:
                    summary["embedding_dimension"] = result["embedding_dimension"]

            elif result["status"] == "skipped":
                summary["skipped"] += 1

            else:
                summary["failed"] += 1

        logfire.info(
            "📊 Directory processing completed",
            total_files=summary["total_files"],
            successful=summary["successful"],
            failed=summary["failed"],
            skipped=summary["skipped"],
            chunks=summary["total_chunks"],
            vectors=summary["total_vectors"],
            model=summary["embedding_model"],
            dimension=summary["embedding_dimension"],
        )

    return summary


# ---------------------------------------------------------------------
# Source type detection
# ---------------------------------------------------------------------


def detect_source_type(
    directory_name: str,
) -> str:
    """
    Infer source type from a directory name.

    Examples:
        true_data  → true
        noisy_data → noisy
        documents  → general
    """

    normalized = directory_name.lower()

    if "true" in normalized:
        return "true"

    if "noisy" in normalized:
        return "noisy"

    return "general"


# ---------------------------------------------------------------------
# Qdrant collection management
# ---------------------------------------------------------------------


def ensure_collection() -> None:
    """
    Ensure that the configured Qdrant collection exists.

    If it already exists, validate its dimension against the
    active embedding model.
    """

    collection_name = settings.QDRANT_COLLECTION

    print("\n" + "=" * 60)

    print("🔍 DEBUG: ensure_collection()")

    print("=" * 60)

    print(f"Collection : {collection_name}")

    print(f"Qdrant URL : {settings.QDRANT_URL}")

    try:
        # -------------------------------------------------------------
        # 1. Embedding dimension
        # -------------------------------------------------------------

        print("\n1️⃣ Getting embedding dimension...")

        dimension = get_embedding_dim()

        print(f"   ✅ Dimension: {dimension}")

        # -------------------------------------------------------------
        # 2. Collection existence
        # -------------------------------------------------------------

        print("\n2️⃣ Checking collection_exists()...")

        exists = qdrant_client.collection_exists(collection_name)

        print(f"   ✅ Exists: {exists}")

        # -------------------------------------------------------------
        # 3. Existing collection
        # -------------------------------------------------------------

        if exists:
            print("\n3️⃣ Calling get_collection()...")

            collection_info = qdrant_client.get_collection(collection_name)

            print("   ✅ get_collection() succeeded")

            existing_dimension = collection_info.config.params.vectors.size

            print(f"   Existing dimension: {existing_dimension}")

            # ---------------------------------------------------------
            # 4. Dimension validation
            # ---------------------------------------------------------

            print("\n4️⃣ Comparing dimensions...")

            if existing_dimension != dimension:
                raise RuntimeError(
                    "Qdrant collection dimension mismatch: "
                    f"collection '{collection_name}' has "
                    f"{existing_dimension} dimensions, but "
                    f"the active model "
                    f"'{get_active_model_type()}' "
                    f"produces {dimension} dimensions. "
                    "Use --wipe to recreate the collection "
                    "before ingesting with a different model."
                )

            print("   ✅ Dimensions match")

            logfire.info(
                "Qdrant collection already exists",
                collection=collection_name,
                dimension=existing_dimension,
            )

            print("\n✅ ensure_collection() completed")

            return

        # -------------------------------------------------------------
        # 5. Create collection
        # -------------------------------------------------------------

        print("\n3️⃣ Collection does not exist.")

        print("4️⃣ Creating Qdrant collection...")

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            ),
        )

        print("   ✅ create_collection() succeeded")

        logfire.info(
            "✅ Qdrant collection created",
            collection=collection_name,
            dimension=dimension,
            distance="COSINE",
        )

        print("\n✅ ensure_collection() completed")

    except Exception as exc:
        print("\n" + "=" * 60)

        print("🚨 EXACT FAILURE INSIDE ensure_collection()")

        print("=" * 60)

        print(f"Exception type: {type(exc).__name__}")

        print(f"Exception     : {exc}")

        print("=" * 60)

        raise


def wipe_collection() -> None:
    """
    Delete the configured Qdrant collection.

    The collection will be recreated before ingestion.
    """

    collection_name = settings.QDRANT_COLLECTION

    print(f"\n⚠️ Checking collection for wipe: {collection_name}")

    with logfire.span(
        "⚠️ Wiping Qdrant Collection",
        collection=collection_name,
    ):
        if qdrant_client.collection_exists(collection_name):
            print("   Deleting collection...")

            qdrant_client.delete_collection(collection_name=collection_name)

            logfire.warning(
                "Qdrant collection deleted",
                collection=collection_name,
            )

            print("   ✅ Collection deleted")

        else:
            logfire.info(
                "Collection did not exist — nothing to wipe",
                collection=collection_name,
            )

            print("   ℹ️ Collection did not exist")


# ---------------------------------------------------------------------
# Universal ingestion
# ---------------------------------------------------------------------


def run_universal_ingestion(
    base_dir: str,
    explicit_source_type: Optional[str] = None,
    wipe: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Dict:
    """
    Universal ingestion entry point.
    """

    overall_summary = {
        "base_directory": base_dir,
        "collection": settings.QDRANT_COLLECTION,
        "embedding_model": None,
        "embedding_dimension": None,
        "wipe": wipe,
        "started_at": get_timestamp(),
        "directories": [],
        "total_files": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "total_chunks": 0,
        "total_vectors": 0,
    }

    print("\n" + "=" * 60)

    print("🚀 UNIVERSAL INGESTION")

    print("=" * 60)

    print(f"📁 Base directory : {base_dir}")

    print(f"📦 Collection     : {settings.QDRANT_COLLECTION}")

    print(f"🤗 Model          : {get_active_model_type()}")

    print(f"📐 Dimension      : {get_embedding_dim()}")

    print(f"🧹 Wipe           : {wipe}")

    with logfire.span(
        "🚀 Universal Ingestion Started",
        base_directory=base_dir,
        wipe=wipe,
    ):
        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"Path does not exist: {base_dir}")

        # -------------------------------------------------------------
        # 1. Optional wipe
        # -------------------------------------------------------------

        if wipe:
            wipe_collection()

        # -------------------------------------------------------------
        # 2. Ensure Qdrant collection
        # -------------------------------------------------------------

        ensure_collection()

        overall_summary["embedding_model"] = get_active_model_type()

        overall_summary["embedding_dimension"] = get_embedding_dim()

        print("\n✅ Qdrant initialization complete")

        # -------------------------------------------------------------
        # 3. Detect subdirectories
        # -------------------------------------------------------------

        subdirectories = sorted(
            directory
            for directory in os.listdir(base_dir)
            if os.path.isdir(
                os.path.join(
                    base_dir,
                    directory,
                )
            )
        )

        # -------------------------------------------------------------
        # 4. Flat directory
        # -------------------------------------------------------------

        if not subdirectories:
            if explicit_source_type:
                source_type = explicit_source_type

            else:
                source_type = detect_source_type(
                    os.path.basename(os.path.normpath(base_dir))
                )

            logfire.info(
                "Processing flat directory",
                directory=base_dir,
                source_type=source_type,
            )

            summary = process_directory(
                dir_path=base_dir,
                source_type=source_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            overall_summary["directories"].append(summary)

        # -------------------------------------------------------------
        # 5. Process subdirectories
        # -------------------------------------------------------------

        else:
            for subdirectory in subdirectories:
                source_type = (
                    explicit_source_type
                    if explicit_source_type
                    else detect_source_type(subdirectory)
                )

                directory_path = os.path.join(
                    base_dir,
                    subdirectory,
                )

                summary = process_directory(
                    dir_path=directory_path,
                    source_type=source_type,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                overall_summary["directories"].append(summary)

        # -------------------------------------------------------------
        # 6. Aggregate
        # -------------------------------------------------------------

        for summary in overall_summary["directories"]:
            overall_summary["total_files"] += summary["total_files"]

            overall_summary["successful"] += summary["successful"]

            overall_summary["failed"] += summary["failed"]

            overall_summary["skipped"] += summary["skipped"]

            overall_summary["total_chunks"] += summary["total_chunks"]

            overall_summary["total_vectors"] += summary["total_vectors"]

        overall_summary["completed_at"] = get_timestamp()

        # -------------------------------------------------------------
        # 7. Observability
        # -------------------------------------------------------------

        logfire.info(
            "🎯 Universal ingestion completed",
            total_files=overall_summary["total_files"],
            successful=overall_summary["successful"],
            failed=overall_summary["failed"],
            skipped=overall_summary["skipped"],
            chunks=overall_summary["total_chunks"],
            vectors=overall_summary["total_vectors"],
            model=overall_summary["embedding_model"],
            dimension=overall_summary["embedding_dimension"],
        )

    return overall_summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    wipe_requested = "--wipe" in sys.argv

    clean_args = [argument for argument in sys.argv[1:] if argument != "--wipe"]

    target_dir = clean_args[0] if len(clean_args) >= 1 else "DATA"

    explicit_type = clean_args[1] if len(clean_args) >= 2 else None

    if not os.path.exists(target_dir):
        print(f"❌ Error: path '{target_dir}' does not exist.")

        sys.exit(1)

    try:
        summary = run_universal_ingestion(
            base_dir=target_dir,
            explicit_source_type=explicit_type,
            wipe=wipe_requested,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        )

        print("\n" + "=" * 60)

        print("🚀 INGESTION COMPLETE")

        print("=" * 60)

        print(f"📁 Directory : {summary['base_directory']}")

        print(f"📦 Collection: {summary['collection']}")

        print(f"🤗 Model     : {summary['embedding_model']}")

        print(f"📐 Dimension : {summary['embedding_dimension']}")

        print(f"📄 Files     : {summary['total_files']}")

        print(f"✅ Successful: {summary['successful']}")

        print(f"⚠️ Skipped   : {summary['skipped']}")

        print(f"❌ Failed    : {summary['failed']}")

        print(f"✂️ Chunks    : {summary['total_chunks']}")

        print(f"🧠 Vectors   : {summary['total_vectors']}")

        print("=" * 60)

        if summary["failed"] > 0:
            sys.exit(1)

    except Exception as exc:
        logfire.error(
            "🔥 Ingestion job failed",
            error=f"{type(exc).__name__}: {exc}",
        )

        print(f"\n❌ Ingestion failed: {type(exc).__name__}: {exc}")

        sys.exit(1)
