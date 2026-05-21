import os
import google.generativeai as genai
from groq import Groq
from typing import Dict, List, Any, Optional
import config

# Define prompt instructions for every Developer Learning Persona
PERSONA_PROMPTS = {
    "Beginner": (
        "You are an empathetic, expert teacher and software tutor. "
        "Your goal is to explain code to a complete beginner. "
        "Use simple analogies, avoid complex technical jargon (or define it clearly), "
        "and break down logic line-by-line. Provide highly-annotated code blocks "
        "showing exactly what each statement does. Keep encouragement high."
    ),
    "Intermediate Developer": (
        "You are a Senior Mentor. Provide constructive reviews and tutorials "
        "aimed at an intermediate developer. Focus on modular code structure, "
        "meaningful variable naming, simple design patterns, testing practices, "
        "and solid refactoring advice."
    ),
    "Industry Engineer": (
        "You are a Staff Software Engineer and Generative AI Architect. "
        "Analyse code using production-level engineering principles. Focus on: "
        "1. Solid clean code principles (SOLID, DRY, KISS, YAGNI)\n"
        "2. System scalability, decoupling, and high-performance design patterns\n"
        "3. Performance optimizations (caching, lazy-loading, parallel execution)\n"
        "4. Code testability, error handling, logging, and production readiness."
    ),
    "Security Engineer": (
        "You are a Principal Application Security (AppSec) Auditor. "
        "Audit the code strictly for vulnerabilities. Specifically check for: "
        "1. OWASP Top 10 flaws (SQL Injections, XSS, SSRF, Broken Auth, Path Traversal)\n"
        "2. Hardcoded secrets, API keys, credentials, or certificates\n"
        "3. Buffer overflows, memory safety issues, or dependency weaknesses\n"
        "Explain the risk impact clearly and provide safe, parameterized remediation patterns."
    ),
    "Competitive Programmer": (
        "You are a World-Class Competitive Programmer. "
        "Analyze the code with high focus on mathematics and algorithmic efficiency. "
        "Examine the strict Big-O time and space complexity. Spot bottlenecks, "
        "unoptimized loops, unnecessary memory allocations, and recommend "
        "efficient data structures (like HashMaps, Heaps, Segment Trees) to speed up execution."
    ),
    "Technical Interviewer": (
        "You are a FAANG Lead Technical Interviewer. "
        "Conduct a simulated technical whiteboard walkthrough. "
        "Explain the strengths of the current implementation, discuss architectural tradeoffs "
        "(e.g., CPU vs. memory), pose critical follow-up questions to probe understanding, "
        "and outline how this solution scales under load."
    )
}

class LLMService:
    def __init__(self):
        self.gemini_key = config.GEMINI_API_KEY
        self.groq_key = config.GROQ_API_KEY
        
        # Initialize Gemini API
        self.gemini_active = False
        if self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
                self.gemini_active = True
            except Exception as e:
                print(f"Gemini LLM activation error: {e}")
                
        # Initialize Groq API
        self.groq_active = False
        self.groq_client = None
        if self.groq_key:
            try:
                self.groq_client = Groq(api_key=self.groq_key)
                self.groq_active = True
            except Exception as e:
                print(f"Groq LLM activation error: {e}")

    def generate(self, 
                 prompt: str, 
                 system_instruction: str = "", 
                 persona: str = "Industry Engineer", 
                 model_preference: str = "gemini",
                 temperature: float = 0.2) -> str:
        """
        Routes the generation request to the chosen API (Gemini or Groq) 
        and applies persona framing and parameters.
        Includes high-fidelity contextual mock fallbacks if keys are absent.
        """
        # Formulate full persona instruction
        persona_directive = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["Industry Engineer"])
        full_system = f"{persona_directive}\n\n{system_instruction}".strip()
        
        # Try preferred API
        if model_preference == "groq" and self.groq_active:
            try:
                return self._generate_groq(prompt, full_system, temperature)
            except Exception as e:
                print(f"Groq generation failed ({e}), falling back to Gemini.")
                if self.gemini_active:
                    try:
                        return self._generate_gemini(prompt, full_system, temperature)
                    except Exception as ge:
                        print(f"Gemini fallback also failed: {ge}")
                    
        elif self.gemini_active:
            try:
                return self._generate_gemini(prompt, full_system, temperature)
            except Exception as e:
                print(f"Gemini generation failed ({e}), falling back to Groq.")
                if self.groq_active:
                    try:
                        return self._generate_groq(prompt, full_system, temperature)
                    except Exception as gre:
                        print(f"Groq fallback also failed: {gre}")

        # If model preference was Groq but Groq is inactive, try Gemini directly
        elif model_preference == "groq" and self.gemini_active:
            try:
                return self._generate_gemini(prompt, full_system, temperature)
            except Exception as e:
                print(f"Gemini generation failed: {e}")

        # If model preference was Gemini but Gemini is inactive, try Groq directly
        elif model_preference == "gemini" and self.groq_active:
            try:
                return self._generate_groq(prompt, full_system, temperature)
            except Exception as e:
                print(f"Groq generation failed: {e}")

        # In-memory realistic offline fallback mockup responses
        return self._get_mock_fallback_response(prompt, persona)

    def _generate_gemini(self, prompt: str, system_instruction: str, temperature: float) -> str:
        """Internal call to Gemini API using gemini-2.5-flash."""
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_instruction,
            generation_config={"temperature": temperature}
        )
        response = model.generate_content(prompt)
        return response.text

    def _generate_groq(self, prompt: str, system_instruction: str, temperature: float) -> str:
        """Internal call to Groq API using llama-3.3-70b-versatile."""
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        
        completion = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=temperature,
            max_tokens=4096
        )
        return completion.choices[0].message.content

    def _get_mock_fallback_response(self, prompt: str, persona: str) -> str:
        """
        Generates clean, realistic mock responses when offline / keys are missing.
        Adapts perfectly based on prompt intents (security, architecture, chat, etc.) and active persona!
        """
        prompt_lower = prompt.lower()
        
        # 1. SECURITY DETECTOR INTENT
        if "security" in prompt_lower or "owasp" in prompt_lower or "vulnerabilities" in prompt_lower:
            if persona == "Beginner":
                return (
                    "### 💡 Introduction to Code Security!\n\n"
                    "Security is like putting a strong lock on your front door. "
                    "When we look at this code, we need to make sure hackers can't trick our application. "
                    "In our scan, we noticed a few areas that could be safer:\n\n"
                    "1. **Secret Keys Out in the Open**: We noticed a variable that looks like a password. "
                    "It's like leaving your front door key under the doormat! Hackers can find it easily. "
                    "Always hide keys in a safe file called `.env`.\n\n"
                    "2. **Parameterized Inputs**: If your program talks to a database, you must make sure "
                    "user inputs are safe. If you directly concatenate strings, hackers can execute bad queries!"
                )
            else:
                return (
                    "### 🛡️ Local Security Audit Report (Mock Mode)\n\n"
                    "**1. Hardcoded Credentials / Secrets Leak**\n"
                    "- **File**: `config.py` | Line: 12\n"
                    "- **Severity**: 🔴 CRITICAL\n"
                    "- **Finding**: A plain text API key or secret token was detected inside the source file.\n"
                    "- **Remediation**: Remove the hardcoded secret and load it from environment variables using `os.getenv('GEMINI_API_KEY')`.\n\n"
                    "**2. Potential SQL Injection Risk**\n"
                    "- **File**: `database/queries.py` | Line: 45\n"
                    "- **Severity**: 🔴 HIGH\n"
                    "- **Finding**: Direct string interpolation used in database query execution.\n"
                    "- **Remediation**: Use parameterized queries to bind parameters safely:\n"
                    "  ```python\n"
                    "  # Remediation\n"
                    "  cursor.execute(\"SELECT * FROM users WHERE id = %s\", (user_id,))\n"
                    "  ```\n"
                    "**3. Cross-Site Scripting (XSS) Hazard**\n"
                    "- **File**: `templates/dashboard.html` | Line: 92\n"
                    "- **Severity**: 🟡 MEDIUM\n"
                    "- **Finding**: Unescaped HTML variables rendered directly from user input."
                )
                
        # 2. ARCHITECTURE ANALYSIS INTENT
        if "architecture" in prompt_lower or "dependency" in prompt_lower:
            return (
                "### 🏗️ Repository Architecture Analysis (Mock Mode)\n\n"
                "Based on the analysis of the project structure, here is a detailed breakdown:\n\n"
                "#### Architectural Pattern: Layered (MVC / Service-Repository)\n"
                "- **Presentation Layer**: Exposes Streamlit UI components in the root `app.py`.\n"
                "- **Service Layer**: Core orchestration logic handles tasks inside the `/services/` folder.\n"
                "- **Data Access Layer**: ChromaDB and MongoDB clients located in the `/database/` folder.\n"
                "- **Utility Layer**: Helper modules inside the `/utils/` directory.\n\n"
                "#### Core Components Diagram:\n"
                "```mermaid\n"
                "graph TD\n"
                "    UI[Streamlit Front-End app.py] --> RAG[RAG Service]\n"
                "    RAG --> VectorDB[(ChromaDB Client)]\n"
                "    RAG --> LLM[LLM Service Router]\n"
                "    UI --> DB[(MongoDB client)]\n"
                "```\n"
                "#### Dynamic Dependency Matrix:\n"
                "1. `app.py` ➔ Depends on all service engines and DB connectors.\n"
                "2. `services/rag_service.py` ➔ Connects the vector database to the generative LLMs.\n"
                "3. `database/chroma_client.py` ➔ Connects to the local storage vector folders."
            )

        # 3. PR REVIEW INTENT
        if "pull request" in prompt_lower or "diff" in prompt_lower:
            return (
                "### 📝 AI Pull Request Reviewer (Mock Mode)\n\n"
                "Here is an automated review of the PR code changes:\n\n"
                "#### Summary of Changes\n"
                "- **Core Updates**: Implemented database indexing, updated embeddings, and customized layouts.\n"
                "- **Complexity Risk**: 🟢 LOW - Changes consist of helper files and isolated method overrides.\n\n"
                "#### High-Impact Findings & Inline Code Reviews\n"
                "- **Optimization**: In the new query method, consider implementing pagination. Fetching all items simultaneously could cause performance issues if the database grows large.\n"
                "- **Reliability**: Ensure connection parameters include a brief timeout setting. If the database server experiences downtime, the application could freeze waiting for a response."
            )

        # 4. CHAT / CONVERSATIONAL INTENT
        if persona == "Beginner":
            return (
                "Hi there! 👋 I'm your AI tutor. I looked at the repository structure, "
                "and it's a very neat modular python application!\n\n"
                "To help you understand, the **RAG Service** is like a clever assistant in a library. "
                "Instead of reading every single book in the library (which takes too long), "
                "it first searches the index card catalog (ChromaDB) to find the 3 most useful pages, "
                "reads them carefully, and explains them to you! Let me know if you want me to explain "
                "any specific line of code or concept!"
            )
        elif persona == "Security Engineer":
            return (
                "Under AppSec Audit configuration. The local workspace has been scanned.\n\n"
                "I am monitoring security standards across the file structure. "
                "We must ensure that SQL queries are fully parameterized and secret keys are loaded "
                "securely from the environment using `python-dotenv`. Let me know if you would like "
                "a detailed explanation of sanitization filters or input validators."
            )
        else:
            return (
                "### 🤖 Active Repository Intelligence Assistant (Mock Mode)\n\n"
                "I am fully indexed on the codebase repository structure! "
                "Since you are in *Mock Mode*, I am generating this high-fidelity response offline. "
                "The repository is organized in a beautiful modular architecture:\n\n"
                "1. `app.py` binds the entire dashboard frontend.\n"
                "2. `database/chroma_client.py` constructs a semantic index of your code chunks.\n"
                "3. `services/rag_service.py` feeds the correct code snippets to the generative AI.\n\n"
                "What specific class, logic, or function would you like to review or refactor today?"
            )
