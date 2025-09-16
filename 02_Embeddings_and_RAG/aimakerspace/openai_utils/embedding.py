from dotenv import load_dotenv
import openai
from typing import List
import os
import asyncio


class EmbeddingModel:
    def __init__(self, embeddings_model_name: str = "text-embedding-3-small", batch_size: int = 1024):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. Please set it to your OpenAI API key."
            )
        # Configure legacy OpenAI SDK (v0.x)
        openai.api_key = self.openai_api_key
        self.embeddings_model_name = embeddings_model_name
        self.batch_size = batch_size

    async def async_get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        batches = [list_of_text[i:i + self.batch_size] for i in range(0, len(list_of_text), self.batch_size)]

        async def process_batch(batch):
            # Legacy async embeddings API
            embedding_response = await openai.Embedding.acreate(
                input=batch, model=self.embeddings_model_name
            )
            return [embeddings["embedding"] for embeddings in embedding_response["data"]]

        results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        return [embedding for batch_result in results for embedding in batch_result]

    async def async_get_embedding(self, text: str) -> List[float]:
        embedding = await openai.Embedding.acreate(
            input=text, model=self.embeddings_model_name
        )
        return embedding["data"][0]["embedding"]

    def get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
        embedding_response = openai.Embedding.create(
            input=list_of_text, model=self.embeddings_model_name
        )
        return [embeddings["embedding"] for embeddings in embedding_response["data"]]

    def get_embedding(self, text: str) -> List[float]:
        embedding = openai.Embedding.create(
            input=text, model=self.embeddings_model_name
        )
        return embedding["data"][0]["embedding"]


if __name__ == "__main__":
    embedding_model = EmbeddingModel()
    print(asyncio.run(embedding_model.async_get_embedding("Hello, world!")))
    print(
        asyncio.run(
            embedding_model.async_get_embeddings(["Hello, world!", "Goodbye, world!"])
        )
    )
