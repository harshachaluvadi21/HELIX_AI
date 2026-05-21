from typing import Dict, List, Any, Optional
from services.llm_service import LLMService
from database.mongodb_client import MongoDBClient

class DocumentationGenerator:
    def __init__(self, mongodb_client: Optional[MongoDBClient] = None):
        """Initializes the AI Documentation Generator Module."""
        self.db = mongodb_client or MongoDBClient()
        self.llm = LLMService()

    def generate_readme(self, repo_id: str, parsed_repo: Dict[str, Any], persona: str = "Industry Engineer") -> str:
        """
        Asks the LLM to analyze the workspace structure and generate 
        a high-impact, professional SaaS-style README.md.
        """
        files = parsed_repo.get("files", [])
        
        # Assemble short summary of codebase
        file_list = []
        for file in files[:30]:  # Limit details to keep prompt efficient
            file_list.append(f"- `{file['filepath']}` ({file['language']}, LOC: {file['lines']})")
        flat_files = "\n".join(file_list)
        
        languages_str = ", ".join([f"{k} ({v}%)" for k, v in parsed_repo.get("languages", {}).items()])

        prompt = (
            f"You are a World-Class Technical Writer and AI SaaS Designer. "
            f"Generate a master-level, production-ready, beautiful README.md for the following repository:\n\n"
            f"=== REPOSITORY STATISTICS ===\n"
            f"- Total files: {parsed_repo.get('total_files')}\n"
            f"- Lines of Code (LOC): {parsed_repo.get('total_loc')}\n"
            f"- Languages density: {languages_str}\n\n"
            f"=== DIRECTORY TREE ===\n"
            f"{flat_files}\n\n"
            f"Please write a comprehensive README in markdown. Include standard open-source badges (shields.io style), "
            f"a catchy banner title, a detailed project description, a premium Features list with emojis, "
            f"a clean ascii/markdown directory structure tree, a complete local installation & setup guide, "
            f"usage examples, and a professional API Reference or architecture outline. "
            f"Make the visual layout outstanding."
        )

        response_text = self.llm.generate(
            prompt=prompt,
            system_instruction="You are a professional technical document architect.",
            persona=persona,
            model_preference="gemini",
            temperature=0.3
        )
        
        # Save README generation in database analysis run
        analysis_record = {
            "repo_id": repo_id,
            "mode": "README Generation",
            "score": 95,
            "report": response_text,
            "created_at": None
        }
        self.db.save_analysis(analysis_record)

        return response_text

    def generate_api_documentation(self, repo_id: str, file_record: Dict[str, Any], persona: str = "Industry Engineer") -> str:
        """
        Inspects a specific code file and generates developer-centric inline docs,
        describing all methods, classes, signatures, and imports.
        """
        filepath = file_record["filepath"]
        abs_path = file_record["absolute_path"]
        
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            return f"Error reading file content for documentation: {e}"

        prompt = (
            f"You are a Senior Developer. Create a highly professional, exhaustive developer "
            f"API Documentation for the following source file:\n\n"
            f"File: `{filepath}`\n"
            f"Language: {file_record.get('language')}\n\n"
            f"=== SOURCE CODE ===\n"
            f"```\n"
            f"{content[:5000]}\n"  # Grab up to 5000 chars to cover standard files
            f"```\n\n"
            f"Please structure your API documentation in Markdown:\n"
            f"1. **Module Overview**: Purpose, namespaces, dependencies.\n"
            f"2. **Classes & Interfaces**: Summarize constructor parameters, inheritances.\n"
            f"3. **Method & Function References**: Detail signature, parameters, types, return structures, exceptions, and side effects.\n"
            f"4. **Usage Snippet**: Provide a neat, copy-paste-ready example showing how to import and use the module."
        )

        response_text = self.llm.generate(
            prompt=prompt,
            system_instruction="You write highly technical, structured, precise api doc guides.",
            persona=persona,
            model_preference="gemini",
            temperature=0.1
        )
        
        return response_text
