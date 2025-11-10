import json, os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMB_MODEL_ID = "intfloat/multilingual-e5-base"
EMB = HuggingFaceEmbeddings(model_name=EMB_MODEL_ID,
                            encode_kwargs={"normalize_embeddings": True})

FAQ_DIR    = "./vectorstore/chroma_faq"
CLAIMS_DIR = "./vectorstore/chroma_claims"

FAQ_JSON    = "./data/faq_metodologia.json"
CLAIMS_JSON = "./data/claims_pib.json"

def load_jsonl_or_json(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        try:
            data = json.loads(txt)
            if isinstance(data, dict):
                data = [data]
        except json.JSONDecodeError:
            data = [json.loads(l) for l in txt.splitlines() if l.strip()]
    return data

def clean_metadata(metadata):
    """Convierte listas y objetos complejos a strings para Chroma"""
    clean = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            # Convertir lista a string separado por comas
            clean[key] = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            # Convertir dict a JSON string
            clean[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        else:
            # Cualquier otro tipo, convertir a string
            clean[key] = str(value)
    return clean

def build_faq_index():
    data = load_jsonl_or_json(FAQ_JSON)
    texts, ids, metas = [], [], []
    for item in data:
        texts.append(item["document"].strip())
        ids.append(item["id"])
        metas.append(clean_metadata(item["metadata"]))
    # crear o upsert
    try:
        vs = Chroma(persist_directory=FAQ_DIR, embedding_function=EMB)
        if vs._collection.count() == 0:
            vs = Chroma.from_texts(texts=texts, metadatas=metas, ids=ids,
                                   embedding=EMB, persist_directory=FAQ_DIR)
        else:
            vs.add_texts(texts=texts, metadatas=metas, ids=ids)
    except Exception:
        vs = Chroma.from_texts(texts=texts, metadatas=metas, ids=ids,
                               embedding=EMB, persist_directory=FAQ_DIR)
        
    print(f"[OK] FAQ index en {FAQ_DIR} — {len(texts)} items")

def build_claims_index():
    data = load_jsonl_or_json(CLAIMS_JSON)
    texts, ids, metas = [], [], []
    for item in data:
        texts.append(item["search_text"].strip())  # lo que se indexa
        ids.append(item["id"])
        # guardamos el texto limpio para mostrarlo en respuesta
        md = item["metadata"].copy()
        md["display_text"] = item["display_text"]
        metas.append(clean_metadata(md))
    try:
        vs = Chroma(persist_directory=CLAIMS_DIR, embedding_function=EMB)
        if vs._collection.count() == 0:
            vs = Chroma.from_texts(texts=texts, metadatas=metas, ids=ids,
                                   embedding=EMB, persist_directory=CLAIMS_DIR)
        else:
            vs.add_texts(texts=texts, metadatas=metas, ids=ids)
    except Exception:
        vs = Chroma.from_texts(texts=texts, metadatas=metas, ids=ids,
                               embedding=EMB, persist_directory=CLAIMS_DIR)
        
    print(f"[OK] Claims index en {CLAIMS_DIR} — {len(texts)} items")

if __name__ == "__main__":
    os.makedirs(FAQ_DIR, exist_ok=True)
    os.makedirs(CLAIMS_DIR, exist_ok=True)
    build_faq_index()
    build_claims_index()
