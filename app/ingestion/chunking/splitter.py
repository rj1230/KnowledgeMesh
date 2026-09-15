from typing import List
import logfire


def _split_oversized_text(text: str, chunk_size: int) -> List[str]:
    """
    Split a single piece of text that is larger than chunk_size.

    Preference order:
    1. Sentence boundaries
    2. Word boundaries
    3. Hard character split as final fallback

    This guarantees that no returned chunk exceeds chunk_size.
    """
    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    remaining = text

    while len(remaining) > chunk_size:
        candidate = remaining[:chunk_size]

        # Prefer splitting at the last sentence boundary.
        sentence_positions = [
            candidate.rfind(". "),
            candidate.rfind("? "),
            candidate.rfind("! "),
            candidate.rfind(".\n"),
            candidate.rfind("?\n"),
            candidate.rfind("!\n"),
        ]

        split_at = max(sentence_positions)

        # Only use the sentence boundary if it is reasonably far
        # into the chunk. Otherwise look for a word boundary.
        if split_at < int(chunk_size * 0.5):
            split_at = candidate.rfind(" ")

        # If there is no useful boundary, hard split.
        if split_at < 1:
            split_at = chunk_size
        else:
            # Include punctuation but remove surrounding whitespace later.
            split_at += 1

        piece = remaining[:split_at].strip()

        if piece:
            chunks.append(piece)

        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def _add_overlap(
    chunks: List[str],
    chunk_overlap: int,
    chunk_size: int,
) -> List[str]:
    """
    Add character overlap between adjacent chunks.

    Example:

        Chunk A: [.................]
                       overlap
        Chunk B:          [.................]

    The overlap is taken from the end of the previous chunk.

    The final chunks are always guaranteed to remain <= chunk_size.
    """
    if not chunks or chunk_overlap <= 0:
        return chunks

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    overlapped: List[str] = []

    for index, chunk in enumerate(chunks):
        if index == 0:
            overlapped.append(chunk)
            continue

        previous = overlapped[-1]

        overlap_text = previous[-chunk_overlap:].strip()

        if not overlap_text:
            overlapped.append(chunk)
            continue

        # We cannot simply prepend overlap because that could push
        # the chunk above chunk_size.
        available_space = chunk_size - len(chunk)

        if available_space <= 0:
            overlapped.append(chunk)
            continue

        overlap_text = overlap_text[:available_space]

        combined = f"{overlap_text} {chunk}".strip()

        # Final safety guarantee.
        if len(combined) > chunk_size:
            combined = combined[:chunk_size].rstrip()

        overlapped.append(combined)

    return overlapped


def chunk_text(
    text: str,
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
) -> List[str]:
    """
    Production-oriented paragraph-aware text chunker.

    Features:
    - Splits primarily on paragraphs.
    - Keeps chunks <= chunk_size.
    - Splits oversized paragraphs safely.
    - Supports configurable overlap.
    - Preserves semantic boundaries where possible.
    - Emits Logfire tracing information.

    Args:
        text:
            Full extracted document text.

        chunk_size:
            Maximum number of characters allowed in each chunk.

        chunk_overlap:
            Number of characters of context carried between chunks.

    Returns:
        List[str]:
            Clean, non-empty text chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    with logfire.span(
        "✂️ Text Chunking",
        text_length=len(text),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ):
        if not text or not text.strip():
            logfire.info("Empty text received — no chunks generated.")
            return []

        # Normalize line endings.
        normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split on blank lines.
        raw_paragraphs = normalized_text.split("\n\n")

        paragraphs: List[str] = []

        for paragraph in raw_paragraphs:
            cleaned = " ".join(paragraph.split())

            if cleaned:
                paragraphs.append(cleaned)

        if not paragraphs:
            logfire.info("No valid paragraphs found.")
            return []

        base_chunks: List[str] = []
        current_chunk = ""

        for paragraph in paragraphs:

            # Handle paragraphs larger than chunk_size separately.
            if len(paragraph) > chunk_size:
                if current_chunk:
                    base_chunks.append(current_chunk.strip())
                    current_chunk = ""

                oversized_chunks = _split_oversized_text(
                    paragraph,
                    chunk_size,
                )

                base_chunks.extend(oversized_chunks)
                continue

            # Start a new chunk.
            if not current_chunk:
                current_chunk = paragraph
                continue

            # Try adding the next paragraph.
            candidate = f"{current_chunk}\n\n{paragraph}"

            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                base_chunks.append(current_chunk.strip())
                current_chunk = paragraph

        # Flush final chunk.
        if current_chunk.strip():
            base_chunks.append(current_chunk.strip())

        # Add overlap.
        final_chunks = _add_overlap(
            base_chunks,
            chunk_overlap=chunk_overlap,
            chunk_size=chunk_size,
        )

        # Final validation.
        valid_chunks = [
            chunk.strip()
            for chunk in final_chunks
            if chunk and chunk.strip()
        ]

        oversized_count = sum(
            1 for chunk in valid_chunks if len(chunk) > chunk_size
        )

        if oversized_count:
            # This should never happen because of the safeguards above.
            raise RuntimeError(
                f"Chunking invariant violated: "
                f"{oversized_count} chunks exceed chunk_size={chunk_size}."
            )

        logfire.info(
            "✅ Chunking completed",
            original_characters=len(text),
            paragraphs=len(paragraphs),
            chunks=len(valid_chunks),
            max_chunk_length=max(
                (len(chunk) for chunk in valid_chunks),
                default=0,
            ),
        )

        return valid_chunks