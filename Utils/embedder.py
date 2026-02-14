from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from Utils.util import clean_generations

class Embedder:
    """ """

    def __init__(self, model_name: str):
        if model_name in ["./LLM_Models/Sentence_Embedding_Models/all-mpnet-base-v2"]:
            self.model = SentenceTransformer(model_name)
        elif model_name in ["bge-small-en-v1.5"]:
            self.model = TextEmbedding(
                model_name="BAAI/bge-small-en-v1.5", providers=["CUDAExecutionProvider"]
            )
        else:
            raise ValueError("Invalid model name")

        self.model_name = model_name

    def _embed(self, text_inputs: List[str]) -> np.ndarray:
        """
        Embeds the input text.
        """

        if self.model_name in ["./LLM_Models/Sentence_Embedding_Models/all-mpnet-base-v2"]:
            return self.model.encode(text_inputs)
        elif self.model_name in ["bge-small-en-v1.5"]:
            return np.array(
                list(self.model.embed(text_inputs))
            )  # original returns generatorj

    def embed_individual_generations(
        self, individual_generations: np.ndarray, clean: bool
    ) -> np.ndarray:
        """
        Embeds individual generations from multiple prompts or models.

        Args:
            individual_generations (np.ndarray): Array of shape (n_samples, n_prompts) containing text generations.
            clean (bool): Whether to clean the generations before embedding.

        Returns:
            np.ndarray: Array of shape (n_samples, n_prompts, embedding_dim) containing the embeddings.
        """

        n_samples, n_models = individual_generations.shape
        if clean:
            cleaned_generations = np.array(clean_generations(individual_generations))
            embeddings = self._embed(cleaned_generations.flatten())
        else:
            embeddings = self._embed(individual_generations.flatten())
        embeddings = embeddings.reshape(n_samples, n_models, -1)
        return embeddings

    def embed_dataset(self, dataset: List[str]) -> np.ndarray:
        """
        Embeds the input text.
        """
        inputs = [sample["embedding_input"] for sample in dataset]
        return self._embed(inputs)