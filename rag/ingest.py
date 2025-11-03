# rag/ingest.py  (RAG ingest แบบ fastembed/ONNX, ใช้ได้ทั้งเครื่องจริงและ Render)
import os, sys, re
from pathlib import Path
from typing import List, Dict, Any

print(">>> [ingest] starting...", flush=True)

# ===== Path Config (รองรับ Render) =====
ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.getenv("SOURCES_DIR", str(ROOT / "data" / "sources")))
DB_DIR = Path(os.getenv("CHROMA_DIR", str(ROOT / "data" / "chroma")))

SRC.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

print(f">>> [ingest] SRC={SRC}", flush=True)
print(f">>> [ingest] DB_DIR={DB_DIR}", flush=True)

# ===== Imports =====
import pandas as pd
import chromadb
from pypdf import PdfReader
from fastembed import TextEmbedding

# ===== ENV Config =====
CSV_COLUMNS = [c.strip() for c in os.getenv("CSV_COLUMNS", "name,description,benefits,ingredients,usage").split(",") if c.strip()]
CSV_SEP = os.getenv("CSV_SEP", "")
MIN_TEXT_LEN = int(os.getenv("MIN_TEXT_LEN", "12"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

print(f">>> [ingest] CSV_COLUMNS={CSV_COLUMNS}", flush=True)
print(f">>> [ingest] CSV_SEP={repr(CSV_SEP) or 'auto'}", flush=True)
print(f">>> [ingest] MIN_TEXT_LEN={MIN_TEXT_LEN}", flush=True)
print(f">>> [ingest] EMBED_MODEL={EMBED_MODEL_NAME}", flush=True)

# ===== Init Embedding + Chroma =====
try:
    embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)
except Exception as e:
    print(f"!! [error] cannot load embedding model: {e}", flush=True)
    sys.exit(1)

client = chromadb.PersistentClient(path=str(DB_DIR))
col = client.get_or_create_collection("drmagica_knowledge")

# ===== Utilities =====
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def chunk_text(text: str, chunk_size=800, overlap=120) -> List[str]:
    text = _clean(text)
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        out.append(text[i:j])
        i = max(0, j - overlap)
        if i == j:
            break
    return out or []

def load_pdf(p: Path) -> str:
    try:
        r = PdfReader(str(p))
        return "\n".join([(pg.extract_text() or "") for pg in r.pages])
    except Exception as e:
        print(f"!! [pdf-error] {p.name}: {e}", flush=True)
        return ""

def load_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def read_csv_smart(p: Path) -> pd.DataFrame:
    seps = [CSV_SEP] if CSV_SEP else [",", ";", "\t", "|"]
    encs = ["utf-8-sig", "utf-8", "cp874", "latin1"]
    for sep in seps:
        for enc in encs:
            try:
                df = pd.read_csv(p, encoding=enc, dtype=str, on_bad_lines="skip", sep=(sep or None))
                df = df.fillna("")
                print(f">>> [csv] encoding={enc}, sep={repr(sep) or 'auto'}, shape={df.shape}", flush=True)
                print(f">>> [csv] columns={list(df.columns)}", flush=True)
                return df
            except Exception:
                continue
    print(f">>> [csv] fallback read without hints", flush=True)
    return pd.read_csv(p, dtype=str).fillna("")

def to_doc_from_row(row: Dict[str, Any], selected_cols: List[str]) -> str:
    if len(selected_cols) == 1 and selected_cols[0] == "*":
        parts = [f"{k}: {v}" for k, v in row.items() if str(v).strip()]
        return "\n".join(parts)
    parts = []
    for c in selected_cols:
        if c in row and str(row[c]).strip():
            parts.append(f"{c}: {row[c]}")
    return "\n".join(parts)

def embed_texts(texts: List[str]) -> List[List[float]]:
    return list(embedder.embed(texts))

def upsert_docs(documents: List[str], metadatas: List[dict], base_id: str):
    if not documents:
        return
    embs = embed_texts(documents)
    ids = [f"{base_id}::{i}" for i in range(len(documents))]
    col.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embs)

# ===== Ingestors =====
def ingest_csv(p: Path):
    print(f">>> [csv] reading {p.name} ...", flush=True)
    df = read_csv_smart(p)
    selected = CSV_COLUMNS if CSV_COLUMNS else ["*"]
    print(f">>> [csv] using columns = {selected}", flush=True)
    added = 0
    for idx, r in df.iterrows():
        doc = to_doc_from_row(r.to_dict(), selected)
        doc = _clean(doc)
        if len(doc) < MIN_TEXT_LEN:
            continue
        chunks = chunk_text(doc)
        metas = [{"source": p.name, "row": int(idx), "chunk": i, "type": "csv"} for i, _ in enumerate(chunks or [""])]
        upsert_docs(chunks or [doc], metas, f"csv::{p.name}::{idx}")
        added += len(chunks) if chunks else 1
    print(f"✓ Ingested CSV {p.name}: {added} records/chunks", flush=True)

def ingest_file(p: Path):
    ext = p.suffix.lower()
    print(f">>> [file] ingest {p.name} (ext={ext})", flush=True)
    if ext == ".csv":
        ingest_csv(p); return
    content = load_pdf(p) if ext == ".pdf" else load_txt(p)
    content = _clean(content)
    if len(content) < MIN_TEXT_LEN:
        print(f"!! skip {p.name}: too short/empty", flush=True)
        return
    chunks = chunk_text(content)
    metas = [{"source": p.name, "idx": i, "type": ext.lstrip(".")} for i, _ in enumerate(chunks)]
    upsert_docs(chunks, metas, f"file::{p.name}")
    print(f"✓ Ingested {p.name}: {len(chunks)} chunks", flush=True)

# ===== Entry =====
def main():
    files = []
    for pat in ("*.csv", "*.pdf", "*.txt", "*.md"):
        files += list(SRC.glob(pat))
    print(f">>> [ingest] found {len(files)} files in {SRC}", flush=True)
    if not files:
        print(f"!! No files in {SRC}. Put CSV/PDF/TXT/MD there.", flush=True)
        return
    for f in files:
        ingest_file(f)
    print(f">>> [ingest] done. total chunks = {col.count()}", flush=True)

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
