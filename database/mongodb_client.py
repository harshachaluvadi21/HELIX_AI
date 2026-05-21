import os
import json
import time
from bson import ObjectId
from typing import Dict, List, Any, Optional
import pymongo
import config

class MongoDBClient:
    """
    Robust MongoDB Client for the Repository Intelligence Platform.
    If MONGODB_URI is not set or the connection fails, it falls back to a 
    fully functional local JSON/Memory storage mechanism (ideal for demo/offline).
    """
    def __init__(self):
        self.uri = config.MONGODB_URI
        self.db_name = config.MONGODB_DATABASE
        self.use_fallback = True
        self.client = None
        self.db = None
        
        # Local mock storage structure
        self.fallback_db_path = os.path.join(config.BASE_DIR, "sandbox", "fallback_db.json")
        self.fallback_data = {
            "users": {},
            "repositories": {},
            "analyses": {},
            "chats": {}
        }
        
        # Load local database if it exists
        if os.path.exists(self.fallback_db_path):
            try:
                with open(self.fallback_db_path, "r") as f:
                    self.fallback_data = json.load(f)
            except Exception:
                pass

        if self.uri:
            try:
                # 3-second connection timeout to prevent UI freezes
                self.client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=3000)
                # Quick server ping check
                self.client.admin.command('ping')
                self.db = self.client[self.db_name]
                self.use_fallback = False
                print("MongoDB Atlas/Local connected successfully.")
            except Exception as e:
                print(f"MongoDB connection failed: {e}. Activating robust local JSON fallback database.")
                self.use_fallback = True
        else:
            print("No MONGODB_URI configured. Activating robust local JSON fallback database.")

    def _save_fallback_db(self):
        """Saves memory database changes to local JSON file."""
        try:
            with open(self.fallback_db_path, "w") as f:
                json.dump(self.fallback_data, f, indent=2)
        except Exception as e:
            print(f"Failed to write fallback DB to disk: {e}")

    def _generate_id(self) -> str:
        """Generates a standard hexadecimal ID matching BSON ObjectId structure."""
        return str(ObjectId())

    # --- REPOSITORY METADATA SECTION ---

    def save_repository(self, repo_data: Dict[str, Any]) -> str:
        """Saves a repository index details, updating if already exists."""
        repo_data = repo_data.copy()
        
        if self.use_fallback:
            repo_id = repo_data.get("_id") or self._generate_id()
            repo_data["_id"] = repo_id
            repo_data["last_indexed_at"] = repo_data.get("last_indexed_at", time.time())
            
            # Upsert by name
            existing_id = None
            for rid, rdata in self.fallback_data["repositories"].items():
                if rdata.get("name") == repo_data.get("name"):
                    existing_id = rid
                    break
            
            if existing_id:
                repo_data["_id"] = existing_id
                self.fallback_data["repositories"][existing_id] = repo_data
                repo_id = existing_id
            else:
                self.fallback_data["repositories"][repo_id] = repo_data
                
            self._save_fallback_db()
            return repo_id
        else:
            # Query by name first
            coll = self.db["repositories"]
            existing = coll.find_one({"name": repo_data.get("name")})
            
            repo_data["last_indexed_at"] = time.time()
            if existing:
                coll.update_one({"_id": existing["_id"]}, {"$set": repo_data})
                return str(existing["_id"])
            else:
                res = coll.insert_one(repo_data)
                return str(res.inserted_id)

    def get_repository(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a repository metadata by ID."""
        if self.use_fallback:
            return self.fallback_data["repositories"].get(repo_id)
        else:
            coll = self.db["repositories"]
            res = coll.find_one({"_id": ObjectId(repo_id)})
            if res:
                res["_id"] = str(res["_id"])
            return res

    def get_all_repositories(self) -> List[Dict[str, Any]]:
        """Retrieves all indexed repositories."""
        if self.use_fallback:
            # Return as list
            return list(self.fallback_data["repositories"].values())
        else:
            coll = self.db["repositories"]
            results = list(coll.find().sort("last_indexed_at", pymongo.DESCENDING))
            for res in results:
                res["_id"] = str(res["_id"])
            return results

    def delete_repository(self, repo_id: str) -> bool:
        """Deletes repository metadata records."""
        if self.use_fallback:
            if repo_id in self.fallback_data["repositories"]:
                del self.fallback_data["repositories"][repo_id]
                
                # Clean up linked analyses & chats
                self.fallback_data["analyses"] = {
                    aid: adata for aid, adata in self.fallback_data["analyses"].items()
                    if adata.get("repo_id") != repo_id
                }
                self.fallback_data["chats"] = {
                    cid: cdata for cid, cdata in self.fallback_data["chats"].items()
                    if cdata.get("repo_id") != repo_id
                }
                
                self._save_fallback_db()
                return True
            return False
        else:
            try:
                oid = ObjectId(repo_id)
                self.db["repositories"].delete_one({"_id": oid})
                self.db["analyses"].delete_many({"repo_id": repo_id})
                self.db["chats"].delete_many({"repo_id": repo_id})
                return True
            except Exception:
                return False

    # --- AI ANALYSIS SECTION ---

    def save_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Saves a code quality, security scan, or architecture analysis run."""
        analysis_data = analysis_data.copy()
        analysis_data["created_at"] = time.time()
        
        if self.use_fallback:
            aid = self._generate_id()
            analysis_data["_id"] = aid
            self.fallback_data["analyses"][aid] = analysis_data
            self._save_fallback_db()
            return aid
        else:
            coll = self.db["analyses"]
            res = coll.insert_one(analysis_data)
            return str(res.inserted_id)

    def get_analyses_for_repo(self, repo_id: str) -> List[Dict[str, Any]]:
        """Retrieves historical analysis reports computed for a repo."""
        if self.use_fallback:
            return [
                adata for adata in self.fallback_data["analyses"].values()
                if adata.get("repo_id") == repo_id
            ]
        else:
            coll = self.db["analyses"]
            results = list(coll.find({"repo_id": repo_id}).sort("created_at", pymongo.DESCENDING))
            for res in results:
                res["_id"] = str(res["_id"])
            return results

    # --- CHAT WORKFLOW SECTION ---

    def create_chat(self, repo_id: str, title: str) -> str:
        """Creates a new conversational chat session."""
        chat_session = {
            "repo_id": repo_id,
            "title": title,
            "messages": [],
            "created_at": time.time(),
            "updated_at": time.time()
        }
        
        if self.use_fallback:
            cid = self._generate_id()
            chat_session["_id"] = cid
            self.fallback_data["chats"][cid] = chat_session
            self._save_fallback_db()
            return cid
        else:
            coll = self.db["chats"]
            res = coll.insert_one(chat_session)
            return str(res.inserted_id)

    def add_chat_message(self, chat_id: str, role: str, content: str) -> bool:
        """Appends a user/assistant message to a chat thread."""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        if self.use_fallback:
            if chat_id in self.fallback_data["chats"]:
                self.fallback_data["chats"][chat_id]["messages"].append(message)
                self.fallback_data["chats"][chat_id]["updated_at"] = time.time()
                self._save_fallback_db()
                return True
            return False
        else:
            try:
                coll = self.db["chats"]
                coll.update_one(
                    {"_id": ObjectId(chat_id)},
                    {
                        "$push": {"messages": message},
                        "$set": {"updated_at": time.time()}
                    }
                )
                return True
            except Exception:
                return False

    def get_chats_for_repo(self, repo_id: str) -> List[Dict[str, Any]]:
        """Retrieves all chat histories matching a repository context."""
        if self.use_fallback:
            return sorted(
                [cdata for cdata in self.fallback_data["chats"].values() if cdata.get("repo_id") == repo_id],
                key=lambda x: x.get("updated_at", 0),
                reverse=True
            )
        else:
            coll = self.db["chats"]
            results = list(coll.find({"repo_id": repo_id}).sort("updated_at", pymongo.DESCENDING))
            for res in results:
                res["_id"] = str(res["_id"])
            return results

    def get_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single chat session."""
        if self.use_fallback:
            return self.fallback_data["chats"].get(chat_id)
        else:
            try:
                coll = self.db["chats"]
                res = coll.find_one({"_id": ObjectId(chat_id)})
                if res:
                    res["_id"] = str(res["_id"])
                return res
            except Exception:
                return None
