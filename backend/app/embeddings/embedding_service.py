from sentence_transformers import SentenceTransformer


class EmbeddingService:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_SIZE = 384

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_NAME)

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return vector.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Texts cannot contain empty values.")

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        return vectors.tolist()
