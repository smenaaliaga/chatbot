# Estudio para AI PIBot 

## Vector Store o Base de Datos Vectorial

Una __Vector Store__ (o base de datos vectorial) es un sistema especializado para almacenar y buscar representaciones numéricas (embeddings) de texto, imagenes o cualquier tipo de dato semántico. Cada documento o fragmento de texto se transforma en un vector.

Estos embeddings se almacenan en la vector store junto con sus metadatos (ej. _id\_doc_, _fecha_, _fuente_).

### ¿Para qué sirve?

Permite realizar búsquedas semánticas, es decir, encuentra los textos que significan lo mismo que esta preguntando, no solo los que contienen la misma palabra.

### Cómo funcionan internamente

1. Indexación:
    - Se calculan los embeddings con un modelo (como _sentence-transformers_ o _e5_).
    - Se almacenan en una estructura que permita búsquedas rápidas: KD-trees, HNSW (Hierarchical Navigable Small World graphs), IVF, etc.
2. Consulta:
    - La pregunta del usuario se convierte en embedding.
    - Se calculan las distancias con los embeddings almacenados.
    - Devuelve los _k_ más similares (por ejemplo, los 3 fragmentos de texto más relevantes).
3. Resultados:
    - Estos fragmentos son enviados como _contexto_ al LLM (en RAG).

### Metadatos

Cada embedding se guarda con información extra (metadatos), como:

```
{
  "id": "manual_ipc_2023_p14",
  "text": "El IPC se calcula sobre una canasta de bienes...",
  "metadata": {
    "fuente": "Manual IPC 2023",
    "página": 14,
    "fecha": "2023-06-01"
  }
}
```

### Chroma

[__Chroma__](https://docs.trychroma.com/docs/overview/introduction) es una base de datos vectorial open source muy usado con LangChain. Sirve para guardar embeddings y consultar por similitud.

Características:
- 100% local (persistencia en disco con SQLite y Parquet).
- Compatible nativamente con LangChain.
- Permite guardar y cargar índices fácilmente.
- Soporta filtrado por metadatos.
- Tiene una API simple en Python.

### Integración con LangChain 

LangChain usa el concepto llamado __Retriever__, que es un envoltorio alrededor de la _vector store_ para conectarla fácilmente a un LLM.

```
retriever = vectorstore.as_retriever(search_kwarg={"k": 4})
query = "¿Cómo se calcula el IPC?"
docs  = retriever.get_relevant_documents(query)
```

Cada `doc` trae:
- `page_content` (el text)
- `metadata` (document, fecha, etc.)

Y luego esos `docs` son pasados al __RAG prompt__ como contexto:

```
context = "\n".join([d.page_content for d in docs])
prompt = f"Usa el siguiente contexto para responder:\n{context}\n\nPregunta: {query}"
```

## RAG (Retrieval-Augmented Generation) 

RAG es un patrón para que un LLM responda usando evidencias reales recuperadas en el momento, en lugar de "recordar" de memoria. Combina búsqueda (retrieval) + redacción (generation) para dar respuestas trazables, actualizadas y con menos alucinaciones.

### ¿Qué problema resuelve?

- Los LLMs "saben" muchas cosas, pero:
    - se desactualizan,
    - no tienen acceso a documentos especificos,
    - pueden inventar detalles.
- RAG hace que el modelo lea las fuentes (PDFs, wikis, APIs) y se limite a responder sobre ellas.

### Flujo base

1. Ingesta & preparación
    - Ingestas documentos (PDF, HTML, CSV, JSON).
    - Limpias y partes en chunks (p. ej., 500-1200 tokens con solape 10-20\%).
    - Calculas embeddings (E5/BGE/SBERT) para cada chunk.
    - Guardas en una __vector store__ (Chroma/FAISS/PGVector/Milvus) junto con metadatos (doc_id, página, fecha, permisos).
2. Consulta
    - El usuario pregunta: se crea el embedding de la pregunta.
    - La vector store hace kNN semántico (coseno/inner-product) y trae k chunks relevantes.
    - (Opcional) Re-ranqueo con un cross-encoder (ej, bge-reranker) para mejorar precisión.
3. Montaje de prompt
    - Se arma un prompt con: instrucciones, la pregunta, y los chunks citables.
    - (Opcional) Inyectas resultados de Tools (API/SQL) si la intención es "dato numérico".
4. Generación
    - El LLM redacta la respuesta basada en el contexto (RAG).
    - Se pide citar (IDs/páginas/enlaces) y no inventar si algo no está en el contexto.
5. Post-proceso & validación
    - Verficas formato (JSON Schema/regex), políticas de citas, y reglas anti-alucinación.
    - Cache de consultas frecuentes.
    - Logs/telemetría para mejorar el índice y los prompts.

### Variantes de RAG (del más simple al más potente)

- __Clásico "Retrive-then-Read"__: 1 consulta → l documentos → respuesta.
- __Fusion-in-Decoder__: mezcla de múltiples consultas y fuentes.
- __HyDE / Query expansion__: el LLM inventa una "respuesta hipotética" para mejorar el embedding de la consulta y recuperar mejor.
- __Re-rankers__: tras el vector search, un cross-encoder reordena por relevancia verdadera.
- __Self-RAG / Critic__: el LLM evalúa su propia respuesta/fuente y puede recuperar de nuevo si la confianza es baja.
- __Graph-RAG__: construye un grafo de entidades/relaciones para recuperar estructuras (útil en corpus complejos).
- __Agentic/Router RAG__: primero decide qué herramienta usar (RAG vs API/SQL) y encadena pasos.

### Decisiones claves de diseño

1) Chunking
    - __Tamaño__: 500-1200 tokens suele funcionar bien; añade solape (10-20\%)
    - __Estrategias__: no cortes párrafos/viñetas a la mitad; respeta secciones/títulos.
    - __Metadatos__: doc_id, páginas, sección, tags. Sirven para filtrar.
2) Embeddings
    - __Multilingue__ (español): __E5-multilingual__, __BGE-m3__, __MiniLM Multilingie__.
    - Normalizar (L2) y mantener misma familia para indexado y consulta.
    - Actualiza el índice cuando cambien documentos.
3) Recuperación
    - __k__: 3-8 suele ser correcto (evitar "context stuffing").
    - __Filtros__: por fecha/versión/tag; evitar mezclar bases incompatibles.
    - __Híbrido (BM5 + vector)__: útil si el dominio es muy terminológico.
    - __Re-rank__: mejora mucho la precisión del top-k, a costa de latencia.
5) Guardia y calidad
    - __No answer__ si la evidencia es insuficiente.
    - __Citas__ obligatorias (IDs o enlaces + páginas).
    - Evaluación continua del dataset de preguntas con "gold context".

#### * Híbrido BM25 + Vector

1. Los documentos se indexan en dos índices:
    - Uno clásico tipo BM25 (ej. Elasticsearch, Whoosh, Lucene, Vespa)
    - Uno vectorial (ej. Chroma, FAISS, Weavite, PGVector)
2. Cuando llega una pregunta:
    - Se ejecuta la búsqueda BM25 → devuelve los documentos con palabras clave exactas más parecidas.
    - Se ejecuta la búsqueda vectorial → devuelve los documentos más similares semánticamente.
    - Se combinan los resultados (ej. sumando o ponderandos los scores)
3. El resultado final mezcla ambos resultados:
```
score_total = α * score_vector + (1 - α) * score_BM25
```
donde `α` puede ser 0.6–0.8 (se da más peso al embedding).

### Métricas de evaluación 

- __Answer relevancy__: ¿responde a la pregunta?
- __Context precision/recall__ ¿los chunks traídos contienen la evidencia correcta?
- __Faithfulness (fidelidad)__: ¿cada afirmación se respalda en el contexto? (menor alucinación)
- __Groundedness__: porcentaje de frases con cita válida.
- __Latency & throughput__: tiempo de recuperación y de generación.
- Herramientas: __RAGAS__, __DeepEval__, human eval con rúbricas.
