import os
import zipfile
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Supported extension mapping
EXTENSION_MAP = {
    ".py": "Python",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".cpp": "C++",
    ".h": "C++",
    ".hpp": "C++",
    ".cc": "C++",
    ".go": "Go",
    ".php": "PHP",
    ".html": "HTML",
    ".css": "CSS",
    ".sh": "Shell",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON",
    ".md": "Markdown"
}

# Directories to exclude during parsing
EXCLUDED_DIRS = {
    "node_modules", "venv", ".venv", "env", ".env", "target", "dist", "build",
    "bin", "obj", ".git", ".github", ".gemini", ".idea", ".vscode", "__pycache__",
    ".metadata", ".gradle", ".settings", "out"
}

# File extensions to ignore (binaries, caches, locks)
EXCLUDED_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf", ".zip", ".tar",
    ".gz", ".rar", ".exe", ".dll", ".so", ".dylib", ".class", ".jar", ".war",
    ".pyc", ".pyo", ".db", ".sqlite", ".woff", ".woff2", ".ttf", ".eot",
    ".package-lock.json", ".yarn.lock", ".pnpm-lock.yaml", "cargo.lock"
}

def is_text_file(filepath: Path) -> bool:
    """Quick check to see if a file contains text rather than binary data."""
    if filepath.suffix.lower() in EXCLUDED_EXTS:
        return False
    
    # Check first 1024 bytes for null bytes
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                return False
        return True
    except Exception:
        return False

def count_lines(filepath: Path) -> int:
    """Counts the total lines of code in a file, skipping encoding issues."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def safe_extract_zip(zip_path: str, extract_dir: str) -> str:
    """
    Safely extracts a ZIP file to the target extraction directory.
    Includes validation to prevent path traversal (Zip Slip vulnerability).
    """
    os.makedirs(extract_dir, exist_ok=True)
    resolved_extract_dir = Path(extract_dir).resolve()

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            # Resolve the path to ensure it doesn't escape the target folder
            target_path = Path(resolved_extract_dir / member.filename).resolve()
            if not str(target_path).startswith(str(resolved_extract_dir)):
                # Path traversal attempt detected!
                continue
            
            if member.is_dir():
                os.makedirs(target_path, exist_ok=True)
            else:
                os.makedirs(target_path.parent, exist_ok=True)
                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                    
    return str(resolved_extract_dir)

def parse_codebase(root_path: str) -> Dict[str, Any]:
    """
    Scans the extracted codebase folder.
    Filters files, calculates statistics, and aggregates file contents.
    
    Returns:
        Dict detailing statistics and file details.
    """
    root = Path(root_path).resolve()
    files_data = []
    language_stats = {}
    total_loc = 0
    total_size = 0
    
    for dirpath, dirnames, filenames in os.walk(root):
        # Exclude directories in-place to prevent os.walk from entering them
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS and not d.startswith(".")]
        
        for fname in filenames:
            fpath = Path(dirpath) / fname
            
            # Skip hidden files
            if fname.startswith("."):
                continue
                
            suffix = fpath.suffix.lower()
            
            # Filter binaries and unsupported media
            if suffix in EXCLUDED_EXTS or not is_text_file(fpath):
                continue
                
            # Filter files larger than 1MB
            try:
                fsize = fpath.stat().st_size
                if fsize > 1024 * 1024:  # 1MB
                    continue
            except Exception:
                continue
                
            lang = EXTENSION_MAP.get(suffix, "Other")
            loc = count_lines(fpath)
            
            relative_path = str(fpath.relative_to(root)).replace("\\", "/")
            
            file_record = {
                "filepath": relative_path,
                "absolute_path": str(fpath),
                "language": lang,
                "lines": loc,
                "size_bytes": fsize
            }
            
            files_data.append(file_record)
            
            # Update stats
            language_stats[lang] = language_stats.get(lang, 0) + loc
            total_loc += loc
            total_size += fsize

    # Calculate percentages for languages
    language_percentages = {}
    if total_loc > 0:
        for lang, loc_count in language_stats.items():
            language_percentages[lang] = round((loc_count / total_loc) * 100, 2)
            
    # Sort language percentages descending
    sorted_languages = dict(sorted(language_percentages.items(), key=lambda item: item[1], reverse=True))

    return {
        "files": files_data,
        "languages": sorted_languages,
        "total_files": len(files_data),
        "total_loc": total_loc,
        "total_size_bytes": total_size
    }

def clean_extracted_directory(dir_path: str):
    """Safely cleans up extracted folder when indexing completes."""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    except Exception as e:
        print(f"Error cleaning up repository path {dir_path}: {e}")
