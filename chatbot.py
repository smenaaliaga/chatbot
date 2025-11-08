from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Verificar GPU
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA version: {torch.version.cuda}\n")

# 1. Cargar documento
print("Cargando documento PDF...")
loader = PyPDFLoader("data/Imacec septiembre 2025.pdf")
docs = loader.load()

# 2. Crear embeddings locales con GPU
print("Creando embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cuda'}
)
vectorstore = Chroma.from_documents(docs, embeddings)

# 3. LLM local con GPU
print("Cargando modelo en GPU...")
model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.2,
    do_sample=True,
    top_p=0.95,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

print("\nSistema de Preguntas y Respuestas")
print("Escribe 'salir' para terminar\n")

# Función para procesar preguntas
def responder_pregunta(query):
    docs_relevantes = vectorstore.similarity_search(query, k=3)
    contexto = "\n\n".join([doc.page_content[:800] for doc in docs_relevantes])
    
    prompt = f"""Basándote únicamente en el siguiente contexto, responde la pregunta de forma clara y detallada.

Contexto:
{contexto}

Pregunta: {query}
Respuesta:"""
    
    respuesta_completa = llm.invoke(prompt)
    respuesta = respuesta_completa.strip()
    respuesta = respuesta.split("Power BI Desktop")[0].strip()
    respuesta = respuesta.split("\n\n")[0] if len(respuesta.split("\n\n")) > 0 else respuesta
    
    return respuesta

# Loop interactivo
while True:
    try:
        query = input("\nPregunta: ").strip()
        
        if not query:
            continue
            
        if query.lower() in ['salir', 'exit', 'quit']:
            print("\nHasta luego!")
            break
        
        print("\nProcesando...")
        respuesta = responder_pregunta(query)
        
        print(f"\nRespuesta:\n{respuesta}\n")
        
    except KeyboardInterrupt:
        print("\n\nHasta luego!")
        break
    except Exception as e:
        print(f"\nError: {e}")
        continue
