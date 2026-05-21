import re
import os
from typing import Dict, List, Any, Optional
from services.llm_service import LLMService
from database.mongodb_client import MongoDBClient

class ArchitectureAnalyzer:
    def __init__(self, mongodb_client: Optional[MongoDBClient] = None):
        """Initializes the System Architecture Analyzer."""
        self.db = mongodb_client or MongoDBClient()
        self.llm = LLMService()

    def parse_imports(self, filepath: str) -> List[str]:
        """
        Parses source files to discover dependencies / import statements.
        Supports standard Python, JavaScript, TypeScript, and Java syntax.
        """
        dependencies = []
        if not os.path.exists(filepath):
            return dependencies
            
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            # Python imports
            py_imports = re.findall(r'^(?:from\s+([\w\.]+)\s+import|import\s+([\w\.,\s]+))', content, re.MULTILINE)
            for imp in py_imports:
                for part in imp:
                    if part:
                        dependencies.append(part.strip().split('.')[0])
                        
            # JS/TS imports
            js_imports = re.findall(r'(?:import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]|require\([\'"]([^\'"]+)[\'"]\))', content)
            for imp in js_imports:
                for part in imp:
                    if part:
                        # Extract clean filename
                        dependencies.append(part.split('/')[-1])
                        
            # Java imports
            java_imports = re.findall(r'^import\s+([\w\.]+);', content, re.MULTILINE)
            for imp in java_imports:
                dependencies.append(imp.strip().split('.')[-1])
                
        except Exception:
            pass
            
        return sorted(list(set(dependencies)))

    def map_repository_architecture(self, repo_id: str, parsed_repo: Dict[str, Any], persona: str = "Industry Engineer") -> Dict[str, Any]:
        """
        Asks the LLM to inspect the directory tree and file dependencies
        to map out layers and generate a system architecture report.
        """
        files = parsed_repo.get("files", [])
        
        # Compile a flat file tree representation for the LLM
        file_tree = []
        for file in files[:50]:  # Limit list to prevent token bloating
            file_tree.append(f"- {file['filepath']} ({file['language']}, LOC: {file['lines']})")
            
        flat_tree = "\n".join(file_tree)

        # Trace dependencies on top files
        import_matrix = {}
        for file in files[:10]:
            imports = self.parse_imports(file["absolute_path"])
            if imports:
                # Filter out standard system libraries
                filtered_imports = [imp for imp in imports if len(imp) > 2]
                if filtered_imports:
                    import_matrix[file["filepath"]] = filtered_imports

        dependency_summary = ""
        for src_file, deps in import_matrix.items():
            dependency_summary += f"- `{src_file}` relies on: {', '.join([f'`{d}`' for d in deps])}\n"

        prompt = (
            f"You are a Principal Software Architect. Analyze this repository's structure and import relations:\n\n"
            f"=== DIRECTORY TREE MAP ===\n"
            f"{flat_tree}\n\n"
            f"=== FILE IMPORT RELATIONSHIPS ===\n"
            f"{dependency_summary}\n\n"
            f"Please map the software architecture. "
            f"Provide your report in clean Markdown detailing:\n"
            f"1. **Core Architecture Paradigm**: (e.g. Layered/MVC, Clean Architecture, Hexagonal, Event-Driven, or Monolithic Scripting).\n"
            f"2. **Functional Boundary Layers**: Categorize folders/files into clear layers (e.g. UI/Presentation, Services/Business Logic, Database/Persistence, Utilities).\n"
            f"3. **Component Interaction Flow Diagram**: Render a valid **Mermaid.js graph** (e.g., `graph TD` or `sequenceDiagram`) demonstrating how data flows through the files.\n"
            f"4. **Dependency Coupling Health**: Critique circular dependencies, structural coupling risks, and compliance with SOLID patterns.\n"
            f"5. **Onboarding Roadmap**: A step-by-step guide for a new engineer to understand the system layout."
        )

        response_text = self.llm.generate(
            prompt=prompt,
            system_instruction="You are an expert generative AI software architect. Detail system structures thoroughly.",
            persona=persona,
            model_preference="gemini",
            temperature=0.2
        )

        analysis_record = {
            "repo_id": repo_id,
            "mode": "Architecture Analysis",
            "score": 90,
            "report": response_text,
            "created_at": None
        }
        
        self.db.save_analysis(analysis_record)

        return {
            "report": response_text,
            "import_matrix": import_matrix
        }
