"""
Carga de FAQs en Chroma (vector store local)
"""
import json
import os
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# CONFIGURACIÓN
# ============================================================

EMB_MODEL_ID = "intfloat/multilingual-e5-base"
EMB = HuggingFaceEmbeddings(
    model_name=EMB_MODEL_ID,
    encode_kwargs={"normalize_embeddings": True}
)

FAQ_DIR = "./vectorstore/chroma_faq"
FAQ_JSON = "./data/faq_metodologia.json"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def load_jsonl_or_json(path):
    """Carga JSON o JSONL (líneas separadas)"""
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
    """
    Convierte listas y objetos complejos a strings para Chroma.
    Chroma solo soporta tipos simples: str, int, float, bool, None
    """
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


# ============================================================
# CONSTRUCCIÓN DEL ÍNDICE
# ============================================================

def build_faq_index():
    """Construye índice de FAQs en Chroma (limpia y reconstruye desde cero)"""
    start = time.time()
    
    # Cargar datos
    data = load_jsonl_or_json(FAQ_JSON)
    
    # Preparar textos, IDs y metadata
    texts, ids, metas = [], [], []
    for item in data:
        texts.append(item["document"].strip())
        ids.append(item["id"])
        
        # Limpiar metadata y agregar el ID
        md = clean_metadata(item["metadata"])
        md["id"] = item["id"]
        metas.append(md)
    
    # Eliminar store anterior si existe
    try:
        vs = Chroma(persist_directory=FAQ_DIR, embedding_function=EMB)
        existing_count = vs._collection.count()
        
        if existing_count > 0:
            print(f"[INFO] Vaciando vectorstore existente ({existing_count} documentos)...")
            # Eliminar la colección completa
            vs.delete_collection()
            print(f"[OK] Vectorstore vaciado")
    except Exception:
        print(f"[INFO] No existe vectorstore previo")
    
    # Crear store desde cero
    vs = Chroma.from_texts(
        texts=texts,
        metadatas=metas,
        ids=ids,
        embedding=EMB,
        persist_directory=FAQ_DIR
    )
    
    elapsed = time.time() - start
    count = vs._collection.count()
    
    print(f"[OK] FAQ index reconstruido desde cero")
    print(f"     Directorio: {FAQ_DIR}")
    print(f"     Documentos procesados: {len(texts)} items")
    print(f"     Total en store: {count} documentos")
    print(f"     Tiempo: {elapsed:.2f}s")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Creando vector store con Chroma (local)")
    
    # Crear directorio
    os.makedirs(FAQ_DIR, exist_ok=True)
    
    # Construir índice
    build_faq_index()
    
    print("Proceso completado")
