import re
from typing import Dict, List, Any, Optional
from services.llm_service import LLMService
from database.mongodb_client import MongoDBClient

class SecurityAuditor:
    def __init__(self, mongodb_client: Optional[MongoDBClient] = None):
        """Initializes the Repository Security Auditor."""
        self.db = mongodb_client or MongoDBClient()
        self.llm = LLMService()
        
        # Static regex patterns for secret detections
        self.secret_patterns = {
            "Google/Gemini API Key": r'\b(AIzaSy[a-zA-Z0-9_\-]{33})\b',
            "Groq API Key": r'\b(gsk_[a-zA-Z0-9]{50})\b',
            "Generic API Key / Secret": r'\b(?:api|secret|private|token|key|auth|password|pwd)\b\s*=\s*[\'"]([a-zA-Z0-9_\-\.\=\+]{8,64})[\'"]',
            "MongoDB Connection URL": r'\b(mongodb(?:\+srv)?:\/\/[a-zA-Z0-9_\-\.\:]+(?:\:[a-zA-Z0-9_\-\.\:]+)?@[a-zA-Z0-9_\-\.\:]+(?:\/[a-zA-Z0-9_\-\.\:]+)?)\b',
            "Generic AWS/Secret Key": r'\b(?:AKIA[0-9A-Z]{16})\b',
            "Private RSA Key Banner": r'-----BEGIN\s+RSA\s+PRIVATE\s+KEY-----'
        }

    def scan_file_for_secrets(self, filepath: str, abs_path: str) -> List[Dict[str, Any]]:
        """
        Scans a single file using static regex rules to identify 
        leaked secrets, credentials, or API keys.
        """
        findings = []
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f):
                    for label, pattern in self.secret_patterns.items():
                        matches = re.findall(pattern, line, re.IGNORECASE)
                        if matches:
                            # Avoid flagging placeholders
                            matched_val = matches[0]
                            if any(placeholder in matched_val.lower() for placeholder in ["your_", "my_", "test_", "placeholder", "key_here", "key-here"]):
                                continue
                                
                            # Censor the leaked secret for display safety
                            censored = matched_val[:4] + "..." + matched_val[-4:] if len(matched_val) > 8 else "********"
                            findings.append({
                                "category": "Hardcoded Secret Leaks",
                                "severity": "CRITICAL",
                                "filepath": filepath,
                                "line_number": line_idx + 1,
                                "snippet": line.strip(),
                                "explanation": f"Leaked {label} detected in plain text: `{censored}`.",
                                "fix": "Remove hardcoded secret. Load the credential from a secure environment variable or a secret vault instead."
                            })
        except Exception:
            pass
        return findings

    def run_owasp_audit(self, repo_id: str, parsed_repo: Dict[str, Any], persona: str = "Security Engineer") -> Dict[str, Any]:
        """
        Aggregates static secret scans and performs a Generative AppSec
        OWASP Top 10 code audit on the core files of the repository.
        """
        files = parsed_repo.get("files", [])
        
        # 1. Run Static Secret Scanner on all files
        findings = []
        for file in files:
            file_findings = self.scan_file_for_secrets(file["filepath"], file["absolute_path"])
            if file_findings:
                findings.extend(file_findings)
                
        # 2. Invoke Generative OWASP scan on top 5 files
        sorted_files = sorted(files, key=lambda x: x["lines"], reverse=True)
        top_files = sorted_files[:5]
        
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
            f"You are a Lead Application Security Officer and OWASP Auditor. "
            f"Review these repository source files for vulnerabilities:\n\n"
            f"{files_summary}\n\n"
            f"Check strictly for:\n"
            f"- SQL Injections / Query Interlopations\n"
            f"- Cross-Site Scripting (XSS)\n"
            f"- Missing Authentication / Weak JWT verification\n"
            f"- Insecure Deserialization\n"
            f"- Path Traversal Risks\n\n"
            f"Structure your response in Markdown, detailing:\n"
            f"1. **Executive Security Dashboard**: Overall security posture grade (A to F), total vulnerabilities found.\n"
            f"2. **Detailed Vulnerability Matrix**: Table detailing File, Severity (Critical/High/Medium/Low), Category, and OWASP ID.\n"
            f"3. **Vulnerability Breakthrough Cards**: For each vulnerability, list the exact Line reference, Code snippet, the Attack vector explaination, and the Remediated code fix block."
        )

        response_text = self.llm.generate(
            prompt=prompt,
            system_instruction="You are a strict, expert AppSec vulnerability scanner.",
            persona=persona,
            model_preference="gemini",
            temperature=0.1
        )

        # Basic risk scoring: base score of 100, subtract points for vulnerabilities
        critical_count = len([f for f in findings if f["severity"] == "CRITICAL"])
        deduction = (critical_count * 15)
        if "CRITICAL" in response_text or "🔴 CRITICAL" in response_text:
            deduction += 20
        if "HIGH" in response_text or "🔴 HIGH" in response_text:
            deduction += 15
            
        score = max(35, 100 - deduction)

        analysis_record = {
            "repo_id": repo_id,
            "mode": "Security Audit",
            "score": score,
            "report": response_text,
            "findings": findings,
            "created_at": None
        }
        
        self.db.save_analysis(analysis_record)

        return {
            "score": score,
            "report": response_text,
            "static_findings": findings
        }

    def review_pull_request_diff(self, repo_id: str, diff_text: str, persona: str = "Industry Engineer") -> Dict[str, Any]:
        """
        Performs a detailed AI Review on a GitHub Pull Request unified diff.
        Analyzes risks, duplicate structures, complexity, and performance anomalies in the diff blocks.
        """
        if not diff_text.strip():
            return {
                "score": 100,
                "report": "### 📝 Pull Request Review\n\nNo modifications or file diffs were found in this Pull Request."
            }

        prompt = (
            f"You are a Lead Pull Request Auditor and Senior Reviewer. "
            f"Inspect the following Unified Pull Request Diff carefully:\n\n"
            f"```diff\n"
            f"{diff_text[:8000]}\n"  # Grab up to 8000 characters of diff to fit context
            f"```\n\n"
            f"Compile a high-impact, professional code review report including:\n"
            f"1. **PR Executive Summary**: Overall assessment of the change size, risk profile, and code health impact.\n"
            f"2. **Risk Meter**: (🟢 LOW / 🟡 MEDIUM / 🔴 HIGH) and rationale.\n"
            f"3. **Vulnerabilities & Key Code Issues**: Spot any security flaws, secret leaks, or performance bottlenecks introduced in the '+' lines.\n"
            f"4. **Optimization Suggestions**: Specific refactoring or algorithmic updates with copy-pasteable blocks.\n"
            f"5. **Ready-to-Use GitHub Comments**: Concrete quotes/reviews that can be directly pasted into the GitHub PR thread."
        )

        response_text = self.llm.generate(
            prompt=prompt,
            system_instruction="You are a meticulous, constructive, expert pull request reviewer.",
            persona=persona,
            model_preference="gemini",
            temperature=0.2
        )

        # Deduce a rating score from findings
        risk_score = 90
        if "🔴 HIGH" in response_text or "HIGH RISK" in response_text.upper():
            risk_score = 65
        elif "🟡 MEDIUM" in response_text or "MEDIUM RISK" in response_text.upper():
            risk_score = 80

        analysis_record = {
            "repo_id": repo_id,
            "mode": "PR Code Review",
            "score": risk_score,
            "report": response_text,
            "created_at": None
        }
        
        self.db.save_analysis(analysis_record)

        return {
            "score": risk_score,
            "report": response_text
        }
