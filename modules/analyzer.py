import re
from typing import Dict, List, Any, Optional
from services.llm_service import LLMService
from database.mongodb_client import MongoDBClient

class RepositoryAnalyzer:
    def __init__(self, mongodb_client: Optional[MongoDBClient] = None):
        """Initializes the Repository Code Analyzer."""
        self.db = mongodb_client or MongoDBClient()
        self.llm = LLMService()

    def calculate_static_complexity(self, code_content: str) -> int:
        """
        Approximates code complexity based on branching constructs:
        e.g., if, for, while, switch, case, try-except, catch.
        """
        # Search patterns for control flows
        branch_patterns = [
            r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\bcatch\b', 
            r'\bexcept\b', r'\bcase\b', r'\b\&\&\b', r'\b\|\|\b'
        ]
        score = 1  # Base complexity
        for pattern in branch_patterns:
            score += len(re.findall(pattern, code_content))
        return score

    def generate_heatmap_data(self, parsed_repo: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Compiles a list of files with coordinates mapping size, static complexity,
        and estimated maintenance effort for rendering heatmaps in the UI.
        """
        heatmap_nodes = []
        files = parsed_repo.get("files", [])
        
        for file in files:
            abs_path = file["absolute_path"]
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                content = ""
                
            complexity = self.calculate_static_complexity(content)
            loc = file["lines"]
            size_kb = round(file["size_bytes"] / 1024, 2)
            
            # Simple heuristic for risk score: high size + high complexity = higher risk
            risk_score = round(min(100.0, (complexity * 1.5) + (loc / 10.0)), 1)
            
            heatmap_nodes.append({
                "filepath": file["filepath"],
                "language": file["language"],
                "loc": loc,
                "size_kb": size_kb,
                "complexity": complexity,
                "risk_score": risk_score
            })
            
        # Return top 25 files sorted by risk score to avoid rendering clutter
        return sorted(heatmap_nodes, key=lambda x: x["risk_score"], reverse=True)[:25]

    def compute_code_quality_score(self, repo_id: str, parsed_repo: Dict[str, Any], persona: str = "Industry Engineer") -> Dict[str, Any]:
        """
        Asks the LLM to inspect the core codebase structure and 
        assign an aggregate Code Quality scorecard.
        """
        # Focus on top 5 files to prevent context window bloating
        files = sorted(parsed_repo.get("files", []), key=lambda x: x["lines"], reverse=True)
        top_files = files[:5]
        
        code_context = []
        for file in top_files:
            abs_path = file["absolute_path"]
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                
                content_lines = []
                current_len = 0
                for line in lines:
                    if current_len + len(line) > 30000:
                        break
                    content_lines.append(line)
                    current_len += len(line)
                
                content = "".join(content_lines)
                code_context.append(f"File: {file['filepath']}\n```\n{content}\n```")
            except Exception:
                pass
                
        files_summary = "\n\n".join(code_context)

        prompt = (
            f"You are a Senior Systems Analyst. Perform a deep code quality scan on the following files:\n\n"
            f"{files_summary}\n\n"
            f"Based on this structure, generate a comprehensive Code Quality scorecard.\n"
            f"Format your response EXACTLY as a Markdown dashboard containing:\n"
            f"1. A central rating table out of 100 for:\n"
            f"   - **Readability & Style**\n"
            f"   - **Maintainability & Refactoring**\n"
            f"   - **Error Safety & Robustness**\n"
            f"   - **Performance Efficiency**\n"
            f"2. A final calculated **Global Code Quality Score** (weighted average).\n"
            f"3. Three major actionable improvement items with line references.\n"
            f"4. A summary of code patterns observed."
        )

        response_text = self.llm.generate(
            prompt=prompt,
            system_instruction="You are a strict codebase auditor. Rate code based on best practices.",
            persona=persona,
            model_preference="gemini",
            temperature=0.1
        )

        # Parse a numeric score out of the response if possible, else default to 85
        score_match = re.search(r"Global Code Quality Score:\s*(\d+)", response_text, re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 82

        analysis_record = {
            "repo_id": repo_id,
            "mode": "Code Quality Analysis",
            "score": score,
            "report": response_text,
            "created_at": None  # Will be set inside db.save_analysis
        }
        
        self.db.save_analysis(analysis_record)

        return {
            "score": score,
            "report": response_text
        }
