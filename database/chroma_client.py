import os
import re
import chromadb
from typing import Dict, List, Tuple, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from services.embedding_service import GeminiEmbeddings
import config

# Map language names to LangChain splitter languages
LANGCHAIN_LANG_MAP = {
    "Python": Language.PYTHON,
    "Java": Language.JAVA,
    "JavaScript": Language.JS,
    "TypeScript": Language.TS,
    "C++": Language.CPP,
    "Go": Language.GO,
    "PHP": Language.PHP,
    "HTML": Language.HTML,
}

class ChromaDBClient:
    def __init__(self, persist_dir: Optional[str] = None):
        """Initializes the ChromaDB persistent or in-memory client."""
        self.persist_dir = persist_dir or config.CHROMA_DB_DIR
        
        try:
            # Persistent client
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            print(f"ChromaDB initialized at {self.persist_dir}")
        except Exception as e:
            # Fallback to ephemeral in-memory client (extremely robust for sandboxed/windows tests)
            print(f"Could not initialize persistent ChromaDB: {e}. Falling back to Ephemeral Client.")
            self.client = chromadb.EphemeralClient()
            
        self.embeddings = GeminiEmbeddings()

    def _get_safe_collection_name(self, repo_id: str) -> str:
        """
        Formats a unique repo_id into a safe ChromaDB collection name:
        - 3 to 63 characters
        - Starts and ends with alphanumeric
        - Contains only alphanumeric, underscores, hyphens, or dots
        - No consecutive dots
        """
        # Remove unsafe characters
        name = re.sub(r'[^a-zA-Z0-9_\-\.]', '', repo_id)
        # Prevent consecutive dots
        name = re.sub(r'\.+', '.', name)
        # Ensure it fits length constraints
        name = name[:60]
        # Pad if too short
        if len(name) < 3:
            name = f"repo-{name}" if name else "repo-collection"
        # Ensure it starts/ends with alphanumeric
        if not name[0].isalnum():
            name = "r" + name[1:]
        if not name[-1].isalnum():
            name = name[:-1] + "1"
        return name.lower()

    def get_or_create_collection(self, repo_id: str):
        """Fetches or creates a ChromaDB collection for a specific repository."""
        collection_name = self._get_safe_collection_name(repo_id)
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def delete_collection(self, repo_id: str) -> bool:
        """Deletes a ChromaDB collection for a repository."""
        try:
            collection_name = self._get_safe_collection_name(repo_id)
            self.client.delete_collection(name=collection_name)
            return True
        except Exception:
            return False

    def get_splitter_for_language(self, language: str) -> RecursiveCharacterTextSplitter:
        """Returns a LangChain splitter optimized for a specific programming language."""
        lang_enum = LANGCHAIN_LANG_MAP.get(language)
        
        # Define chunk parameters
        chunk_size = 1200
        chunk_overlap = 150
        
        if lang_enum:
            return RecursiveCharacterTextSplitter.from_language(
                language=lang_enum,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

    def ingest_file(self, repo_id: str, file_record: Dict[str, Any]) -> int:
        """
        Chunks a file according to its language, extracts embeddings, 
        and indexes it inside the repository's Chroma collection.
        """
        filepath = file_record["filepath"]
        abs_path = file_record["absolute_path"]
        language = file_record["language"]
        
        if not os.path.exists(abs_path):
            return 0
            
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            if not content.strip():
                return 0
                
            # Split document
            splitter = self.get_splitter_for_language(language)
            chunks = splitter.split_text(content)
            
            if not chunks:
                return 0
                
            collection = self.get_or_create_collection(repo_id)
            
            ids = []
            documents = []
            metadatas = []
            
            # Simple line range estimator per chunk
            lines = content.split('\n')
            total_lines = len(lines)
            
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{filepath}_chunk_{idx}"
                
                # Approximate lines matching this chunk
                start_line = 1
                end_line = total_lines
                
                # Let's search inside the text for dynamic line tracing
                try:
                    first_few_chars = chunk[:50].strip()
                    if first_few_chars:
                        for l_idx, line in enumerate(lines):
                            if first_few_chars in line:
                                start_line = l_idx + 1
                                break
                    last_few_chars = chunk[-50:].strip()
                    if last_few_chars:
                        for l_idx in range(len(lines) - 1, -1, -1):
                            if last_few_chars in lines[l_idx]:
                                end_line = l_idx + 1
                                break
                except Exception:
                    pass
                
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    "filepath": filepath,
                    "language": language,
                    "start_line": start_line,
                    "end_line": max(start_line, end_line),
                    "repo_id": repo_id
                })
            
            # Generate embeddings
            embeddings_list = self.embeddings.embed_documents(documents)
            
            # Ingest to ChromaDB
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
            return len(chunks)
            
        except Exception as e:
            print(f"Error ingesting file {filepath} to ChromaDB: {e}")
            return 0

    def query_repository(self, repo_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic code search in the collection of a specific repository.
        Returns matching chunks sorted by cosine similarity.
        """
        try:
            collection = self.get_or_create_collection(repo_id)
            query_embedding = self.embeddings.embed_query(query)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit
            )
            
            formatted_results = []
            if results and results.get("documents"):
                # Extract parts
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                ids = results["ids"][0]
                distances = results.get("distances", [[0] * len(docs)])[0]
                
                for idx in range(len(docs)):
                    # Cosine distance to similarity: 1 - distance
                    similarity = 1.0 - distances[idx]
                    formatted_results.append({
                        "id": ids[idx],
                        "code": docs[idx],
                        "metadata": metas[idx],
                        "similarity": round(float(similarity), 4)
                    })
            return formatted_results
        except Exception as e:
            print(f"Error querying ChromaDB repository {repo_id}: {e}")
            return []
