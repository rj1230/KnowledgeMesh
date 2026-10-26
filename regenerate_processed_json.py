from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.chunking.splitter import chunk_text
from app.ingestion.loaders.pdf import parse_pdf


ROOT = Path(__file__).resolve().parent
SOURCE_DIRS = [
    ROOT / "DATA" / "true_data",
    ROOT / "DATA" / "noisy_data",
]

OUTPUT_DIRS = {
    "true_data": ROOT / "processed_data" / "true",
    "noisy_data": ROOT / "processed_data" / "noisy",
}

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process_pdf(pdf_path: Path, source_type: str) -> dict:
    print(f"\nProcessing: {pdf_path.name}")

    text = parse_pdf(str(pdf_path))

    if not text.strip():
        raise RuntimeError(f"No text extracted from {pdf_path}")

    chunks = chunk_text(
        text,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        raise RuntimeError(f"No chunks generated for {pdf_path}")

    document_id = content_hash(text)[:24]

    result = {
        "document_id": document_id,
        "filename": pdf_path.name,
        "source_type": source_type,
        "content_hash": content_hash(text),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": "huggingface:sentence-transformers:BAAI/bge-base-en-v1.5",
        "embedding_dimension": 768,
        "chunking": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
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

    return result


def main() -> None:
    total = 0

    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            print(f"Skipping missing directory: {source_dir}")
            continue

        source_type = source_dir.name

        output_dir = OUTPUT_DIRS[source_type]
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted(source_dir.glob("*.pdf"))

        print(f"\n{'=' * 80}")
        print(f"{source_type}: {len(pdf_files)} PDF files")
        print(f"{'=' * 80}")

        for pdf_path in pdf_files:
            result = process_pdf(pdf_path, source_type)

            output_path = output_dir / f"{pdf_path.name}.json"

            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(
                f"  chunks={result['chunking']['total_chunks']}"
                f" | chars={sum(c['character_count'] for c in result['chunks'])}"
                f" | document_id={result['document_id']}"
            )

            total += 1

    print(f"\nRegenerated {total} PDF JSON files.")
    print("Qdrant was NOT accessed.")
    print("Embedding generation was NOT performed.")


if __name__ == "__main__":
    main()
