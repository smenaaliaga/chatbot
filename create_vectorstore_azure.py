"""
Carga de FAQs en Azure AI Search (vector store en la nube)
"""
import json
import os
import time
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# CONFIGURACIÓN
# ============================================================

EMB_MODEL_ID = "intfloat/multilingual-e5-base"
EMB = HuggingFaceEmbeddings(
    model_name=EMB_MODEL_ID,
    encode_kwargs={"normalize_embeddings": True}
)

# Configuración de Azure AI Search
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "pibot-faq-index"

FAQ_JSON = "./data/faq_metodologia.json"


# ============================================================
# VALIDACIÓN
# ============================================================

if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY:
    raise ValueError(
        "❌ Configuración requerida:\n"
        "   - AZURE_SEARCH_ENDPOINT (variable de entorno)\n"
        "   - AZURE_SEARCH_KEY (variable de entorno)\n\n"
        "   Configúralas en tu archivo .env o como variables de sistema."
    )


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


# ============================================================
# CONSTRUCCIÓN DEL ÍNDICE
# ============================================================

def build_faq_index():
    """Construye índice de FAQs en Azure AI Search"""
    start = time.time()
    
    try:
        from langchain_community.vectorstores.azuresearch import AzureSearch
    except ImportError:
        raise ImportError(
            "❌ Azure AI Search requiere:\n"
            "   pip install azure-search-documents langchain-community"
        )
    
    # Cargar datos
    data = load_jsonl_or_json(FAQ_JSON)
    
    # Preparar textos, IDs y metadata
    texts, ids, metas = [], [], []
    for item in data:
        texts.append(item["document"].strip())
        ids.append(item["id"])
        
        # Metadata (Azure soporta tipos complejos mejor que Chroma)
        md = item["metadata"].copy()
        md["id"] = item["id"]
        
        # Convertir listas a strings si es necesario para compatibilidad
        for key, value in md.items():
            if isinstance(value, list):
                md[key] = ", ".join(str(v) for v in value)
        
        metas.append(md)
    
    # Crear vector store en Azure
    print(f"📤 Conectando a Azure AI Search...")
    print(f"   Endpoint: {AZURE_SEARCH_ENDPOINT}")
    print(f"   Index: {INDEX_NAME}")
    
    vector_store = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=INDEX_NAME,
        embedding_function=EMB,
    )
    
    # Agregar documentos
    print(f"📝 Agregando {len(texts)} documentos...")
    vector_store.add_texts(texts=texts, metadatas=metas, ids=ids)
    
    elapsed = time.time() - start
    
    print(f"[OK] FAQ index en Azure AI Search")
    print(f"     Index: {INDEX_NAME}")
    print(f"     Documentos: {len(texts)} items")
    print(f"     Tiempo: {elapsed:.2f}s")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("☁️  Creando vector store con Azure AI Search")
    print("=" * 60)
    
    try:
        build_faq_index()
        print("=" * 60)
        print("✅ Proceso completado")
        print(f"\n💡 Para consultar desde Azure Portal:")
        print(f"   {AZURE_SEARCH_ENDPOINT}")
        print(f"   Index: {INDEX_NAME}")
    except Exception as e:
        print("=" * 60)
        print(f"❌ Error: {e}")
        print("\n💡 Sugerencias:")
        print("   - Verifica las credenciales en .env")
        print("   - Instala dependencias: pip install azure-search-documents")
        print("   - Verifica acceso de red al endpoint Azure")
