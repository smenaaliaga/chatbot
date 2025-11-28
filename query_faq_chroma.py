"""
Consultas a vector store de FAQs en Chroma con re-ranking
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# ============================================================
# CONFIGURACIÓN
# ============================================================

EMB_MODEL_ID = "intfloat/multilingual-e5-base"
EMB = HuggingFaceEmbeddings(
    model_name=EMB_MODEL_ID,
    encode_kwargs={"normalize_embeddings": True}
)

# Cross-encoder para re-ranking (multilingüe optimizado)
# Alternativas: 
#   - 'cross-encoder/ms-marco-MiniLM-L-6-v2' (inglés, más rápido)
#   - 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1' (multilingüe MS MARCO)
#   - 'cross-encoder/multilingual-miniLM-L12-v2' (multilingüe general)
RERANKER = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

FAQ_DIR = "./vectorstore/chroma_faq"
USE_RERANK = True  # Activar/desactivar re-ranking


# ============================================================
# BÚSQUEDA
# ============================================================

def search_faq(query: str, k: int = 3, filters: dict = None, use_rerank: bool = USE_RERANK):
    """
    Busca en el vector store de FAQs con re-ranking opcional.
    
    Args:
        query: Pregunta o búsqueda
        k: Número de resultados finales
        filters: Filtros de metadata, ej: {"topic": "definicion_general"}
        use_rerank: Si True, aplica re-ranking con cross-encoder
    
    Returns:
        Lista de tuplas (documento, score de similitud)
    """
    # Cargar vector store
    vs = Chroma(persist_directory=FAQ_DIR, embedding_function=EMB)
    
    # Búsqueda inicial más amplia si se usa re-ranking
    search_k = k * 3 if use_rerank else k
    
    kwargs = {"k": search_k}
    if filters:
        kwargs["filter"] = filters
    
    docs_with_scores = vs.similarity_search_with_score(query, **kwargs)
    
    # Re-ranking con cross-encoder si está activado
    if use_rerank and docs_with_scores:
        docs_with_scores = rerank_results(query, docs_with_scores, top_k=k)
    
    return docs_with_scores


def rerank_results(query: str, docs_with_scores, top_k: int = 8):
    """
    Re-rankea resultados usando cross-encoder.
    
    El cross-encoder evalúa la relevancia de cada par (query, documento)
    de forma más precisa que la búsqueda vectorial inicial.
    
    Args:
        query: Pregunta del usuario
        docs_with_scores: Lista de (doc, score) del vector store
        top_k: Cuántos resultados finales retornar
    
    Returns:
        Lista re-rankeada de (doc, rerank_score)
    """
    if not docs_with_scores:
        return []
    
    # Preparar pares query-documento para el cross-encoder
    pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
    
    # Calcular scores de re-ranking (valores más altos = mejor)
    # El cross-encoder devuelve logits crudos (típicamente entre -10 y +10)
    # Para modelos multilingües en español, los logits tienden a ser más bajos
    # que en inglés, así que ajustamos con un offset
    import numpy as np
    rerank_scores = RERANKER.predict(pairs, convert_to_numpy=True, activation_fn=None)
    
    # Aplicar sigmoid con ajuste para modelos multilingües
    # Offset de +2 para compensar bias hacia scores bajos en español
    # sigmoid(x + offset) para recalibrar
    LOGIT_OFFSET = 2.0  # Ajustable según el idioma y modelo
    normalized_scores = 1 / (1 + np.exp(-(rerank_scores + LOGIT_OFFSET)))
    
    # Combinar documentos con nuevos scores y agregar metadata
    reranked = []
    for i, (doc, old_score) in enumerate(docs_with_scores):
        raw_logit = float(rerank_scores[i])
        normalized_score = float(normalized_scores[i])
        
        # Guardar scores originales en metadata para análisis
        doc.metadata['_vector_score'] = old_score
        doc.metadata['_rerank_score_logit'] = raw_logit  # Logit original
        doc.metadata['_rerank_score_raw'] = normalized_score  # Probabilidad [0, 1]
        
        # Invertir para mantener convención (menor es mejor)
        inverted_score = 1.0 - normalized_score
        doc.metadata['_rerank_score'] = inverted_score  # Score invertido para ordenamiento
        
        reranked.append((doc, inverted_score))
    
    # Ordenar por score descendente del cross-encoder (mayor es mejor)
    # pero invertido para mantener consistencia (menor es mejor)
    reranked.sort(key=lambda x: x[1])
    
    return reranked[:top_k]


def deduplicate_by_group(docs_with_scores):
    """
    Elimina duplicados basándose en group_id y pondera por repetición.
    Combina: mejor score + frecuencia de aparición del grupo.
    
    Returns:
        Lista de tuplas (doc, score_ponderado, metadata_extra)
        donde metadata_extra incluye: match_count, avg_score, best_score
    """
    group_stats = {}
    
    for doc, score in docs_with_scores:
        group_id = doc.metadata.get("group_id")
        
        if group_id:
            if group_id not in group_stats:
                group_stats[group_id] = {
                    'doc': doc,
                    'scores': [],
                    'count': 0,
                    'best_score': score
                }
            
            group_stats[group_id]['scores'].append(score)
            group_stats[group_id]['count'] += 1
            
            # Mantener el documento con mejor score individual
            if score < group_stats[group_id]['best_score']:
                group_stats[group_id]['doc'] = doc
                group_stats[group_id]['best_score'] = score
        else:
            # Sin group_id, tratarlo como único
            doc_id = doc.metadata.get("id", str(hash(doc.page_content)))
            group_stats[doc_id] = {
                'doc': doc,
                'scores': [score],
                'count': 1
            }
    
    # Calcular scores ponderados
    results = []
    for group_id, stats in group_stats.items():
        best_score = min(stats['scores'])
        avg_score = sum(stats['scores']) / len(stats['scores'])
        match_count = stats['count']
        
        # Ponderación: mejor score con boost por frecuencia
        # Score más bajo es mejor, así que reducimos según apariciones
        frequency_boost = 0.95 ** (match_count - 1)  # 0.95, 0.90, 0.86...
        weighted_score = best_score * frequency_boost
        
        # Agregar metadata de análisis
        doc = stats['doc']
        doc.metadata['_match_count'] = match_count
        doc.metadata['_avg_score'] = avg_score
        doc.metadata['_best_score'] = best_score
        doc.metadata['_weighted_score'] = weighted_score
        
        results.append((doc, weighted_score))
    
    # Ordenar por score ponderado (menor es mejor)
    results.sort(key=lambda x: x[1])
    return results


def print_results_compact(docs_with_scores):
    """Imprime resultados en formato compacto de una línea"""
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        confidence_pct = max(0, min(100, (1 - score) * 100))
        group_id = doc.metadata.get('group_id', 'N/A')
        print(f"[{i}] {doc.page_content} | Confianza: {confidence_pct:.1f}% (Score ponderado: {score:.4f}) | GROUP ID: {group_id}")


def print_results(docs_with_scores, show_metadata=False):
    """Imprime resultados en formato detallado con estadísticas de matching"""
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        # Obtener scores originales si están disponibles
        vector_score = doc.metadata.get('_vector_score')
        rerank_score_raw = doc.metadata.get('_rerank_score_raw')  # Score original del cross-encoder [-1, 1]
        rerank_score_inv = doc.metadata.get('_rerank_score')  # Score invertido para ordenamiento [0, 1]
        
        # Calcular confianza usando el score del cross-encoder si está disponible
        # Cross-encoder devuelve scores normalizados en [0, 1] donde 1 es mejor match
        if rerank_score_raw is not None:
            # Convertir de [0, 1] a [0, 100]%
            confidence_pct = max(0, min(100, rerank_score_raw * 100))
            confidence_label = "Cross-encoder"
        else:
            # Usar score vectorial (distancia L2, menor es mejor)
            confidence_pct = max(0, min(100, (1 - score) * 100))
            confidence_label = "Vector"
        
        print(f"\n[{i}] {doc.page_content}")
        print(f"    ID: {doc.metadata.get('id', 'N/A')}")
        print(f"    Confianza: {confidence_pct:.1f}% ({confidence_label})")
        
        # Mostrar scores de re-ranking si están disponibles
        rerank_logit = doc.metadata.get('_rerank_score_logit')
        if vector_score is not None and rerank_score_raw is not None:
            logit_str = f" (logit: {rerank_logit:.2f})" if rerank_logit is not None else ""
            print(f"    [Re-rank] Vector: {vector_score:.4f} -> Cross-encoder: {rerank_score_raw:.4f}{logit_str} -> Score final: {score:.4f}")
        
        # Mostrar estadísticas de matching si están disponibles
        match_count = doc.metadata.get('_match_count')
        if match_count and match_count > 1:
            best_score = doc.metadata.get('_best_score', 0)
            best_confidence = max(0, min(100, (1 - best_score) * 100))
            print(f"    Variantes encontradas: {match_count} (mejor individual: {best_confidence:.1f}%)")
        
        # Mostrar si es una pregunta canónica o variante
        canonical = doc.metadata.get("canonical_question")
        if canonical and canonical != doc.page_content:
            print(f"    Pregunta canónica: {canonical}")
        
        answer = doc.metadata.get("answer", "N/A")
        print(f"\n -> Respuesta: {answer}..." if len(answer) > 200 else f"\n -> Respuesta: {answer}")
        
        # Siempre mostrar keywords si existen
        keywords = doc.metadata.get("keywords", "")
        if keywords:
            print(f"   Keywords: {keywords}")
        
        if show_metadata:
            print(f"\n   Metadata adicional:")
            print(f"     Topic: {doc.metadata.get('topic', 'N/A')}")
            print(f"     Metric: {doc.metadata.get('metric', 'N/A')}")
            group_id = doc.metadata.get("group_id")
            if group_id:
                print(f"     Group ID: {group_id}")


# ============================================================
# INTERACTIVO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BASE DE DATOS VECTORIAL PIBot - BANCO CENTRAL DE CHILE")
    print("=" * 60)
    print(f"\nRe-ranking: {'ACTIVADO' if USE_RERANK else 'DESACTIVADO'}")
    print("\nEscribe una pregunta y obtén respuestas de la base.")
    print("Comandos: 'salir' para terminar\n")
    
    while True:
        try:
            query = input("\n>> Pregunta: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['salir', 'exit', 'quit']:
                print("\nHasta luego!")
                break
            
            print("\nBuscando...")
            docs = search_faq(query, k=8)
            
            if not docs:
                print("No se encontraron resultados.")
                continue
            
            # Mostrar resultados sin deduplicar (formato compacto)
            print(f"\n--- Resultados totales sin filtrar: {len(docs)} ---")
            print_results_compact(docs)
            
            # Deduplicar resultados
            docs_unique = deduplicate_by_group(docs)
            
            # Verificar si hay resultados con confianza suficiente
            CONFIDENCE_THRESHOLD = 2.0  # Umbral mínimo de confianza (1%)
            
            valid_results = []
            for doc, score in docs_unique[:3]:
                # Calcular confianza
                rerank_score_raw = doc.metadata.get('_rerank_score_raw')
                if rerank_score_raw is not None:
                    confidence_pct = max(0, min(100, rerank_score_raw * 100))
                else:
                    confidence_pct = max(0, min(100, (1 - score) * 100))
                
                if confidence_pct >= CONFIDENCE_THRESHOLD:
                    valid_results.append((doc, score))
            
            # Mostrar resultados deduplicados (formato detallado)
            print(f"\n--- Respuestas filtradas: {len(docs_unique)} ---")
            
            if not valid_results:
                print("\n  No se encontraron respuestas con suficiente confianza.")
                print(f"    (Ningún resultado supera el umbral de {CONFIDENCE_THRESHOLD}% de confianza)")
            
            print_results(valid_results, show_metadata=False)

        except KeyboardInterrupt:
            print("\n\nHasta luego!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue