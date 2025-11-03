import os
from pathlib import Path
from typing import List, Optional, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

# ---------- RAG deps (ไม่พึ่ง torch) ----------
import chromadb
from fastembed import TextEmbedding

# --- Load .env from project root ---
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# --- ENV ---
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
LLM_MODEL      = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
BASE_URL       = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# --- FastAPI app (create ONCE) ---
app = FastAPI(title="Dr.Magica Chatbot (Groq + RAG)")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],  # ระบุโดเมนจริงในโปรดักชัน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Groq API Client ----------
client: Optional[OpenAI] = None

# ---------- RAG setup (fastembed + Chroma) ----------
ROOT = Path(__file__).resolve().parents[1]
DB_DIR = Path(os.getenv("CHROMA_DIR", str(ROOT / "data" / "chroma")))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

DB_DIR.mkdir(parents=True, exist_ok=True)

# Initialize embedding model
try:
    embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)
except Exception as e:
    print(f"!! [error] cannot load embedding model: {e}", flush=True)
    embedder = None

# Initialize Chroma database (ต้องมีข้อมูลจาก rag/ingest.py ก่อน)
chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
collection = chroma_client.get_or_create_collection(name="drmagica_knowledge")


# ---------- Helper Functions ----------
def _embed_query(q: str) -> List[float]:
    """ฝังเวกเตอร์ query ด้วย fastembed"""
    if embedder is None:
        raise RuntimeError("Embedding model not initialized")
    return list(embedder.embed([q]))[0]


def retrieve_context(query: str, k: int = 5) -> List[dict]:
    """ดึง top-k context จากฐาน Chroma"""
    if collection.count() == 0:
        return []
    q_emb = _embed_query(query)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = []
    for i in range(len(res["documents"][0])):
        docs.append({
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
            "score": float(res["distances"][0][i]),
        })
    docs.sort(key=lambda d: d["score"])
    return docs


def build_rag_system_prompt(contexts: List[dict]) -> str:
    """สร้าง system prompt ที่ใช้ข้อมูลจาก Chroma"""
    if not contexts:
        return (
            "You are Dr.Magica's assistant. "
            "If no internal knowledge is found, answer briefly from general knowledge."
        )
    ctx_text = "\n\n".join(
        [f"[{i+1}] (src: {c['meta'].get('source','?')}#{c['meta'].get('row', c['meta'].get('idx',''))})\n{c['text']}"
         for i, c in enumerate(contexts)]
    )
    return (
        "You are Dr.Magica's RAG assistant. Use ONLY the CONTEXT below to answer. "
        "Cite sources as [n] where n is the context index. "
        "If unsure, say you don't know.\n\n"
        f"CONTEXT:\n{ctx_text}\n\n"
        "Answer in Thai if the user writes Thai; otherwise answer in English."
    )


# ---------- Lifecycle ----------
@app.on_event("startup")
def init_client() -> None:
    global client
    if not GROQ_API_KEY:
        raise RuntimeError("Missing GROQ_API_KEY env")
    client = OpenAI(base_url=BASE_URL, api_key=GROQ_API_KEY)
    print("✅ Groq client initialized.")


# ---------- Schemas ----------
Role = Literal["system", "user", "assistant"]

class Message(BaseModel):
    role: Role
    content: str

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Chat history in OpenAI format")
    model: Optional[str] = None
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 512
    use_rag: Optional[bool] = True
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    reply: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    sources: Optional[List[dict]] = None


# ---------- Routes ----------
@app.get("/health")
def health():
    """ตรวจสุขภาพระบบ"""
    return {
        "ok": True,
        "kb_chunks": collection.count() if collection else 0,
        "embed_model": EMBED_MODEL_NAME,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    global client
    if client is None:
        raise HTTPException(status_code=500, detail="Groq client not initialized")

    try:
        # ดึงข้อความล่าสุดจากผู้ใช้
        user_query = next((m.content for m in reversed(req.messages) if m.role == "user"), "")

        # ดึงข้อมูลจากฐานความรู้
        contexts: List[dict] = retrieve_context(user_query, k=(req.top_k or 5)) if req.use_rag else []
        sys_prompt = build_rag_system_prompt(contexts)

        # รวมข้อความทั้งหมดเข้ากับ system prompt
        messages = [{"role": "system", "content": sys_prompt}] + [m.model_dump() for m in req.messages]

        model = req.model or LLM_MODEL
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        choice = resp.choices[0].message
        usage = getattr(resp, "usage", None)

        return ChatResponse(
            reply=choice.content,
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            sources=[
                {
                    "source": c["meta"].get("source"),
                    "row": c["meta"].get("row", c["meta"].get("idx")),
                    "score": c["score"]
                } for c in contexts
            ] or None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
