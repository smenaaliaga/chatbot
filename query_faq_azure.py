"""
Consultas a vector store de FAQs en Azure AI Search
"""
import os
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# CONFIGURACIÓN
# ============================================================

EMB_MODEL_ID = "intfloat/multilingual-e5-base"
EMB = HuggingFaceEmbeddings(
    model_name=EMB_MODEL_ID,
    encode_kwargs={"normalize_embeddings": True}
)

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = "pibot-faq-index"

if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY:
    raise ValueError(
        "❌ Configuración requerida:\n"
        "   - AZURE_SEARCH_ENDPOINT\n"
        "   - AZURE_SEARCH_KEY"
    )


# ============================================================
# BÚSQUEDA
# ============================================================

def search_faq(query: str, k: int = 3, filters: dict = None):
    """
    Busca en el vector store de FAQs en Azure.
    
    Args:
        query: Pregunta o búsqueda
        k: Número de resultados
        filters: Filtros de metadata, ej: {"topic": "definicion_general"}
    
    Returns:
        Lista de tuplas (documento, score)
    """
    try:
        from langchain_community.vectorstores.azuresearch import AzureSearch
    except ImportError:
        raise ImportError(
            "Azure AI Search requiere: pip install azure-search-documents langchain-community"
        )
    
    # Cargar vector store
    vs = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=INDEX_NAME,
        embedding_function=EMB,
    )
    
    # Construir filtro OData si se especifica
    kwargs = {"k": k}
    if filters:
        # Convertir dict a OData filter
        conditions = []
        for key, value in filters.items():
            if isinstance(value, str):
                conditions.append(f"{key} eq '{value}'")
            else:
                conditions.append(f"{key} eq {value}")
        kwargs["filters"] = " and ".join(conditions)
    
    # Búsqueda con score
    docs_with_scores = vs.similarity_search_with_score(query, **kwargs)
    return docs_with_scores


def print_results(docs_with_scores):
    """Imprime resultados de manera formateada"""
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        print(f"\n{'='*60}")
        print(f"[{i}] {doc.page_content} (Score: {score:.4f})")
        print(f"\n📌 Respuesta:")
        answer = doc.metadata.get("answer", "N/A")
        print(f"   {answer[:200]}..." if len(answer) > 200 else f"   {answer}")
        print(f"\n🏷️  Metadata:")
        print(f"   Topic: {doc.metadata.get('topic', 'N/A')}")
        print(f"   Metric: {doc.metadata.get('metric', 'N/A')}")
        keywords = doc.metadata.get("keywords", "")
        if keywords:
            print(f"   Keywords: {keywords}")


# ============================================================
# EJEMPLOS
# ============================================================

if __name__ == "__main__":
    print("☁️  Consultando FAQs en Azure AI Search")
    print("=" * 60)
    
    try:
        # Ejemplo 1: Búsqueda simple
        query = "¿Qué es el PIB?"
        print(f"\n📝 Query: {query}")
        docs = search_faq(query, k=2)
        print_results(docs)
        
        # Ejemplo 2: Búsqueda con filtros
        print("\n\n" + "=" * 60)
        query = "metodología"
        filters = {"topic": "metodologia_real_encadenada"}
        print(f"\n📝 Query: {query}")
        print(f"🔎 Filtros: {filters}")
        docs = search_faq(query, k=2, filters=filters)
        print_results(docs)
        
        # Ejemplo 3: Búsqueda por keywords
        print("\n\n" + "=" * 60)
        query = "enfoque de producción"
        print(f"\n📝 Query: {query}")
        docs = search_faq(query, k=3)
        print_results(docs)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Verifica la configuración y credenciales de Azure")
