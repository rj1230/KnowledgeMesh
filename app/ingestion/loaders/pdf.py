import logfire
import pymupdf


def parse_pdf(file_path: str) -> str:
    """
    Extract PDF text locally using PyMuPDF.

    PyMuPDF is used as the primary extractor because the previous
    pypdf/pdfplumber pipeline produced PDF text with corrupted word
    boundaries on some documents.
    """
    with logfire.span("PDF Parsing (local)", filename=file_path):
        try:
            doc = pymupdf.open(file_path)
            total_pages = len(doc)
            logfire.info(f"PDF has {total_pages} pages.")

            text_parts: list[str] = []

            for page in doc:
                text = page.get_text("text") or ""
                if text.strip():
                    text_parts.append(text)

            doc.close()

            full_text = "\n".join(text_parts)

            if not full_text.strip():
                logfire.warning(
                    f"No text extracted from {file_path}. "
                    "File may be fully image-based."
                )
            else:
                logfire.info(
                    f"Extracted {len(full_text)} characters from {file_path}."
                )

            return full_text

        except Exception as e:
            logfire.error(f"PDF Parse Failed for {file_path}: {e}")
            raise
