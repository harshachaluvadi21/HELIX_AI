import hashlib
import numpy as np
import google.generativeai as genai
from typing import List
from langchain_core.embeddings import Embeddings
import config

class GeminiEmbeddings(Embeddings):
    """
    Custom LangChain-compatible embedding wrapper around the Google Gemini API 
    using 'models/text-embedding-004'. Contains a deterministic fallback vectorizer
    to support offline operations and testing with missing API keys.
    """
    def __init__(self, api_key: str = "", model_name: str = "models/text-embedding-004"):
        self.model_name = model_name
        self.api_key = api_key or config.GEMINI_API_KEY
        self.active = bool(self.api_key)
        
        if self.active:
            try:
                genai.configure(api_key=self.api_key)
            except Exception as e:
                print(f"Error configuring Gemini Embeddings API: {e}. Falling back to deterministic embeddings.")
                self.active = False

    def _get_fallback_embedding(self, text: str) -> List[float]:
        """
        Generates a 768-dimension deterministic vector based on text content.
        Guarantees that identical text yields identical embeddings, enabling offline search!
        """
        # Create a deterministically seeded generator
        sha256 = hashlib.sha256(text.encode('utf-8')).hexdigest()
        seed = int(sha256[:8], 16)
        rng = np.random.default_rng(seed)
        
        # Standard size 768
        vector = rng.standard_normal(768).tolist()
        
        # L2 Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = [float(v / norm) for v in vector]
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of document strings."""
        if not self.active:
            return [self._get_fallback_embedding(t) for t in texts]
            
        try:
            # Google Generative AI allows batch embedding requests
            # For massive lists, we partition them into chunks of 100
            embeddings = []
            chunk_size = 100
            
            for i in range(0, len(texts), chunk_size):
                batch = texts[i : i + chunk_size]
                result = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type="retrieval_document"
                )
                
                # Check formatting of the response
                if "embedding" in result:
                    embeddings.extend(result["embedding"])
                else:
                    # Some versions return list of dicts or nested structure
                    embeddings.extend([emb for emb in result.get("embeddings", [])])
                    
            # Double check lengths match
            if len(embeddings) == len(texts):
                return embeddings
            else:
                raise ValueError("Embedding count mismatch from API.")
                
        except Exception as e:
            print(f"Gemini API embedding call failed: {e}. Defaulting to fallback vectorizer.")
            return [self._get_fallback_embedding(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string."""
        if not self.active:
            return self._get_fallback_embedding(text)
            
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            
            if "embedding" in result:
                return result["embedding"]
            elif "embeddings" in result and len(result["embeddings"]) > 0:
                return result["embeddings"][0]
            else:
                raise ValueError("Missing embedding data in response.")
                
        except Exception as e:
            print(f"Gemini API query embedding failed: {e}. Defaulting to fallback.")
            return self._get_fallback_embedding(text)
