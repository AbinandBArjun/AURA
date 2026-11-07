import tiktoken
from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_index: int
    text: str
    token_count: int
    page_start: int
    page_end: int
    section: str = "body"
    retrievable: bool = True


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        encoding_name: str = "cl100k_base",
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk_pages(self, pages: list[dict]) -> list[TextChunk]:
        """
        Accepts either original pages or section-annotated segments.

        Annotated segments must contain:
        page_number, text, section, retrievable.
        """

        chunks = []
        current_tokens = []
        current_pages = []
        current_section = None
        current_retrievable = True
        chunk_index = 0

        def flush():
            nonlocal chunk_index, current_tokens, current_pages

            if not current_tokens:
                return

            chunks.append(
                TextChunk(
                    chunk_index=chunk_index,
                    text=self.encoding.decode(current_tokens),
                    token_count=len(current_tokens),
                    page_start=min(current_pages),
                    page_end=max(current_pages),
                    section=current_section or "body",
                    retrievable=current_retrievable,
                )
            )

            chunk_index += 1

        for segment in pages:
            page_number = segment["page_number"]
            text = segment["text"].strip()

            if not text:
                continue

            section = segment.get("section", "body")
            retrievable = segment.get("retrievable", True)

            # Never mix different sections or retrieval policies.
            if current_tokens and (
                section != current_section
                or retrievable != current_retrievable
            ):
                flush()
                current_tokens = []
                current_pages = []

            current_section = section
            current_retrievable = retrievable

            page_tokens = self.encoding.encode(text)

            for token in page_tokens:
                current_tokens.append(token)
                current_pages.append(page_number)

                if len(current_tokens) >= self.chunk_size:
                    flush()

                    if self.chunk_overlap:
                        current_tokens = current_tokens[
                            -self.chunk_overlap:
                        ]
                        current_pages = current_pages[
                            -self.chunk_overlap:
                        ]
                    else:
                        current_tokens = []
                        current_pages = []

        flush()

        return chunks
