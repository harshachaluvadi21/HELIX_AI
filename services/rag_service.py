from typing import Dict, List, Any, Tuple, Optional
from database.chroma_client import ChromaDBClient
from database.mongodb_client import MongoDBClient
from services.llm_service import LLMService

class RAGService:
    def __init__(self, chroma_client: Optional[ChromaDBClient] = None, mongodb_client: Optional[MongoDBClient] = None):
        """Initializes the RAG Orchestrator Service."""
        self.chroma = chroma_client or ChromaDBClient()
        self.db = mongodb_client or MongoDBClient()
        self.llm = LLMService()

    def query_with_rag(self, 
                       repo_id: str, 
                       chat_id: str, 
                       query: str, 
                       persona: str = "Industry Engineer",
                       model_preference: str = "gemini") -> Tuple[str, List[Dict[str, Any]]]:
        """
        Executes a complete Conversational RAG pipeline:
        1. Retrieves semantic code context chunks from ChromaDB.
        2. Retrieves the last few chat messages from MongoDB memory.
        3. Formulates a rich prompt with code snippets and persona instructions.
        4. Invokes the LLMService for generative response.
        5. Persists the exchange in MongoDB chat collections.
        
        Returns:
            Tuple: (Assistant response string, List of retrieved sources metadata)
        """
        # Step 1: Semantic Search
        # Retrieve top 5 matching code snippets
        sources = self.chroma.query_repository(repo_id=repo_id, query=query, limit=5)
        
        # Step 2: Extract chat history context
        chat_history = ""
        chat_session = self.db.get_chat(chat_id)
        if chat_session and chat_session.get("messages"):
            # Use last 5 messages for memory to prevent overflow
            history_messages = chat_session["messages"][-5:]
            history_parts = []
            for msg in history_messages:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role_label}: {msg['content']}")
            chat_history = "\n".join(history_parts)

        # Step 3: Format the context of code snippets
        code_context_parts = []
        for s_idx, src in enumerate(sources):
            meta = src["metadata"]
            code_block = (
                f"--- SOURCE FILE {s_idx + 1} ---\n"
                f"File: {meta.get('filepath')}\n"
                f"Line range: {meta.get('start_line')} to {meta.get('end_line')}\n"
                f"Language: {meta.get('language')}\n"
                f"Code snippet:\n"
                f"```\n"
                f"{src['code']}\n"
                f"```\n"
            )
            code_context_parts.append(code_block)
            
        code_context = "\n".join(code_context_parts) if code_context_parts else "No relevant code snippets found in vector database."

        # Step 4: Construct the augmented RAG prompt
        system_instruction = (
            "You are a helpful AI Assistant with deep access to the source code repository.\n"
            "Analyse the provided code snippets carefully to answer the user's questions.\n"
            "Format your answer with clear sections, bullet points, and markdown tables if helpful.\n"
            "When mentioning files or lines, use format: [filename:Lstart-end](file:///filepath).\n"
            "If the code snippets do not contain the answer, rely on your general programming knowledge "
            "but clearly state that the specific answer was not found in the indexed code snippets."
        )

        rag_prompt = (
            f"=== PERSISTENT CHAT HISTORY ===\n"
            f"{chat_history}\n\n"
            f"=== RETRIEVED SOURCE CODE CONTEXT ===\n"
            f"{code_context}\n\n"
            f"=== USER QUERY ===\n"
            f"{query}\n\n"
            f"Please formulate your response according to your active developer persona profile."
        )

        # Step 5: Save User Message in MongoDB
        self.db.add_chat_message(chat_id, "user", query)

        # Step 6: Invoke LLM Generation
        response_text = self.llm.generate(
            prompt=rag_prompt,
            system_instruction=system_instruction,
            persona=persona,
            model_preference=model_preference,
            temperature=0.2
        )

        # Step 7: Save Assistant Message in MongoDB
        self.db.add_chat_message(chat_id, "assistant", response_text)

        return response_text, sources
