import tiktoken
from dataclasses import dataclass


@dataclass
class TextChunk:
    chunk_index: int
    text: str
    token_count: int
    page_start: int
    page_end: int


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
        Convert extracted PDF pages into overlapping token-based chunks.

        Each page must contain:
            {
                "page_number": 1,
                "text": "Extracted page text..."
            }
        """

        chunks = []
        current_tokens = []
        current_pages = []
        chunk_index = 0

        for page in pages:
            page_number = page["page_number"]
            page_text = page["text"].strip()

            if not page_text:
                continue

            page_tokens = self.encoding.encode(page_text)

            for token in page_tokens:
                current_tokens.append(token)
                current_pages.append(page_number)

                if len(current_tokens) >= self.chunk_size:
                    chunk_text = self.encoding.decode(current_tokens)

                    chunks.append(
                        TextChunk(
                            chunk_index=chunk_index,
                            text=chunk_text,
                            token_count=len(current_tokens),
                            page_start=min(current_pages),
                            page_end=max(current_pages),
                        )
                    )

                    chunk_index += 1

                    if self.chunk_overlap:
                        overlap_tokens = current_tokens[-self.chunk_overlap:]
                        overlap_pages = current_pages[-self.chunk_overlap:]
                    else:
                        overlap_tokens = []
                        overlap_pages = []

                    current_tokens = list(overlap_tokens)
                    current_pages = list(overlap_pages)

        if current_tokens:
            chunks.append(
                TextChunk(
                    chunk_index=chunk_index,
                    text=self.encoding.decode(current_tokens),
                    token_count=len(current_tokens),
                    page_start=min(current_pages),
                    page_end=max(current_pages),
                )
            )

        return chunks
