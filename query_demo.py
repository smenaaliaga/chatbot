# query_pipeline_demo.py
# pip install langchain-chroma langchain-huggingface chromadb sentence-transformers

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMB = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base",
                            encode_kwargs={"normalize_embeddings": True})

FAQ_DIR    = "./vectorstore/chroma_faq"
CLAIMS_DIR = "./vectorstore/chroma_claims"

def load_store(path):
    return Chroma(persist_directory=path, embedding_function=EMB)

def search_faq(query, metric="PIB", k=20):
    vs = load_store(FAQ_DIR)
    retriever = vs.as_retriever(search_kwargs={"k": k, "filter": {"metric": metric}})
    docs = retriever.invoke(query)
    return docs

def search_claims(query, metric="PIB", k=8):
    vs = load_store(CLAIMS_DIR)
    retriever = vs.as_retriever(search_kwargs={"k": k, "filter": {"metric": metric}})
    docs = retriever.invoke(query)
    return docs

def pick_best_faq(docs, tau_len=200):
    """
    Heurística simple: si el mejor candidato trae una respuesta canónica razonable (no vacía),
    lo devolvemos. (En producción usarías score del re-ranker o similitud).
    """
    if not docs:
        return None
    best = docs[0]
    ans = best.metadata.get("answer","").strip()
    if len(ans) >= tau_len:
        return best
    return None

def pipeline_answer(query):
    # 1) FAQ primero
    faq_docs = search_faq(query)
    best_faq = pick_best_faq(faq_docs)
    if best_faq:
        return {
            "mode": "FAQ",
            "answer": best_faq.metadata["answer"],
            "source": best_faq.metadata.get("source_url"),
            "debug": {"matched_variant": best_faq.page_content, "group_id": best_faq.metadata.get("group_id")}
        }

    # 2) Fallback: Claims (devuelve 1–2 trozos concretos)
    claim_docs = search_claims(query)
    if claim_docs:
        top = claim_docs[:2]
        claims_out = [
            {
                "text": d.metadata.get("display_text", d.page_content),
                "source": d.metadata.get("source_url"),
                "page": d.metadata.get("page")
            } for d in top
        ]
        return {"mode": "CLAIMS", "claims": claims_out}

    # 3) No-answer
    return {
        "mode": "NO_ANSWER",
        "message": "No encontré suficiente evidencia. ¿Te refieres a definición general o al cálculo real encadenado?"
    }

if __name__ == "__main__":
    tests = [
        "¿Cómo se calcula el PIB?",
        "Explícame la metodología real encadenada del PIB",
        "Qué es el PIB y para qué sirve"
    ]
    for q in tests:
        out = pipeline_answer(q)
        print(f"\nQ: {q}\nR: {out}")
