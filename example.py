from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Verificar GPU
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA versión: {torch.version.cuda}")

# 1. Cargar documento
loader = PyPDFLoader("data/Imacec septiembre 2025.pdf")
docs = loader.load()

# 2. Crear embeddings locales con GPU
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cuda'}  # Usar tu GPU local
)
vectorstore = Chroma.from_documents(docs, embeddings)

# 3. LLM local con GPU (se descarga la primera vez ~4GB)
print("Cargando modelo en GPU...")
model_id = "microsoft/Phi-3-mini-4k-instruct"  # Modelo pequeño y eficiente
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"  # Usa tu GPU automáticamente
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500, 
    temperature=0.1,
    do_sample=True,
    top_p=0.95,
    return_full_text=False 
)

llm = HuggingFacePipeline(pipeline=pipe)

# 5. Consultar
query = "¿Cuál fue la inflación anual según el informe?"

# Buscar documentos relevantes
docs_relevantes = vectorstore.similarity_search(query, k=3)
contexto = "\n\n".join([doc.page_content[:800] for doc in docs_relevantes])

# Crear prompt optimizado para respuesta detallada
prompt = f"""Basándote únicamente en el siguiente contexto, responde la pregunta de forma clara y detallada. Explica qué información se encuentra disponible en el documento.

Contexto:
{contexto}

Pregunta: {query}
Respuesta detallada:"""

# Generar respuesta
respuesta_completa = llm.invoke(prompt)

# Limpiar respuesta (remover repeticiones de "Power BI Desktop")
respuesta = respuesta_completa.strip()
respuesta = respuesta.split("Power BI Desktop")[0].strip()
respuesta = respuesta.split("\n\n")[0] if len(respuesta.split("\n\n")) > 0 else respuesta

# Mostrar resultado
print(f"Question: {query}")
print(f"\Helpful Answer:\n{respuesta}")
