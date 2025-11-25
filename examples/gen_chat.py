import os
import json
import time
import sys
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
from threading import Thread
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Deshabilitar warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Configuración
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
EMB_MODEL_ID = "intfloat/multilingual-e5-base"
CHROMA_DIR = "./chroma_store"
METHODO_JSON = "./data/metodologia.json"

# Datos numéricos simulados
NUMERIC_DATA = {
    ("IPC", "2024-12"): (4.2, "% a/a", "INE"),
    ("PIB", "2024"): (2.1, "% crecimiento real", "Banco Central"),
    ("IMACEC", "2025-01"): (1.3, "% a/a", "Banco Central"),
}

# Verificar GPU
if torch.cuda.is_available():
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")

# 1. Cargar LLM
print("Cargando modelo en GPU...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    device_map="auto"
)

gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=gen_pipe)

# 2. Cargar embeddings y vectorstore
print("Cargando embeddings...")
emb = HuggingFaceEmbeddings(
    model_name=EMB_MODEL_ID,
    encode_kwargs={"normalize_embeddings": True},
)

def load_or_build_vectorstore():
    try:
        vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
        if vs._collection.count() > 0:
            return vs
    except:
        pass
    
    with open(METHODO_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = [d["text"] for d in data["docs"]]
    metadatas = [{"id": d.get("id", "doc")} for d in data["docs"]]
    
    return Chroma.from_texts(texts=texts, embedding=emb, metadatas=metadatas, persist_directory=CHROMA_DIR)

vectorstore = load_or_build_vectorstore()

print("\nSistema de Preguntas sobre Metodología e Indicadores")
print("Escribe 'salir' para terminar\n")

# Función para normalizar periodos
def normalize_period(periodo_raw: Optional[str]) -> Optional[str]:
    if not periodo_raw:
        return None
    s = periodo_raw.strip().lower()
    meses = {
        "enero":"01","febrero":"02","marzo":"03","abril":"04","mayo":"05","junio":"06",
        "julio":"07","agosto":"08","septiembre":"09","octubre":"10","noviembre":"11","diciembre":"12",
        "ene":"01","feb":"02","mar":"03","abr":"04","may":"05","jun":"06",
        "jul":"07","ago":"08","sep":"09","oct":"10","nov":"11","dic":"12",
    }
    
    if s.isdigit() and len(s) == 4:
        return s
    
    for m_name, m_num in meses.items():
        if m_name in s:
            for tok in s.split():
                if tok.isdigit() and len(tok) == 4:
                    return f"{tok}-{m_num}"
    return s

# Función para buscar datos numéricos
def buscar_dato_numerico(question: str) -> Optional[Dict[str, Any]]:
    q_lower = question.lower()
    
    # Detectar métrica
    metric = None
    if "ipc" in q_lower:
        metric = "IPC"
    elif "pib" in q_lower:
        metric = "PIB"
    elif "imacec" in q_lower:
        metric = "IMACEC"
    
    if not metric:
        return None
    
    # Detectar periodo
    periodo = None
    import re
    years = re.findall(r'\b(20\d{2})\b', question)
    for mes in ["enero","febrero","marzo","abril","mayo","junio","julio","agosto",
                "septiembre","octubre","noviembre","diciembre",
                "ene","feb","mar","abr","may","jun","jul","ago","sep","oct","nov","dic"]:
        if mes in q_lower and years:
            periodo = f"{mes} {years[0]}"
            break
    
    # Buscar en datos
    per = normalize_period(periodo)
    key = (metric.upper(), per) if per else None
    
    if key and key in NUMERIC_DATA:
        val, unid, fuente = NUMERIC_DATA[key]
        return {
            "metric": metric,
            "periodo": per,
            "valor": val,
            "unidad": unid,
            "fuente": fuente
        }
    return None

# Función para buscar metodología con streaming
def buscar_metodologia(question: str) -> str:
    # Limpiar pregunta para mejorar búsqueda semántica (eliminar fechas/periodos específicos)
    import re
    question_clean = re.sub(r'\b(de\s+)?(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+\d{4}\b', '', question, flags=re.IGNORECASE)
    question_clean = re.sub(r'\b\d{4}[-/]\d{2}\b', '', question_clean)
    question_clean = re.sub(r'\bcuál\s+fue\b', 'qué es', question_clean, flags=re.IGNORECASE)
    question_clean = question_clean.strip()
    
    docs = vectorstore.similarity_search(question_clean, k=2)
    ctx = "\n\n".join([f"[{d.metadata.get('id')}] {d.page_content[:400]}" for d in docs])
    
    # Simplificar la pregunta para el prompt, enfocándose solo en metodología
    question_simple = re.sub(r'¿?Cuál fue .+ (y|,) ', '¿', question, flags=re.IGNORECASE)
    question_simple = question_simple.strip()
    
    prompt = f"""Contexto:
{ctx}

Pregunta: {question_simple}

Respuesta breve usando el contexto [cita el ID]:"""
    
    # Streaming con TextIteratorStreamer
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=None,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    respuesta_completa = ""
    for text in streamer:
        # Detener si encuentra el inicio de una nueva pregunta
        if "\nPregunta:" in respuesta_completa or "\n\nPregunta:" in respuesta_completa:
            break
        print(text, end="", flush=True)
        respuesta_completa += text
        # Detener si parece que está generando otra pregunta
        if "Pregunta:" in text:
            break
    
    thread.join()
    # Limpiar cualquier texto después de "Pregunta:"
    if "Pregunta:" in respuesta_completa:
        respuesta_completa = respuesta_completa.split("Pregunta:")[0]
    return respuesta_completa.strip()

# Función para responder preguntas con streaming
def responder_pregunta(question: str) -> None:
    q_lower = question.lower()
    
    # Buscar y mostrar dato numérico primero
    dato_keywords = ["cuál fue", "cuánto", "valor", "número", "dato", "cifra"]
    dato_encontrado = False
    if any(k in q_lower for k in dato_keywords):
        dato = buscar_dato_numerico(question)
        if dato:
            print(f"**{dato['metric']} ({dato['periodo']})**: {dato['valor']}{dato['unidad']} — Fuente: {dato['fuente']}")
            dato_encontrado = True
    
    # Buscar y mostrar metodología solo si se pregunta explícitamente
    metodo_keywords = ["cómo se calcula", "metodología", "qué es", "base", "canasta", "definición", "explica", "significa"]
    if any(k in q_lower for k in metodo_keywords):
        if dato_encontrado:
            print()
        print("**Metodología:**")
        buscar_metodologia(question)
        return
    
    # Si ya encontró un dato numérico, no buscar metodología
    if dato_encontrado:
        return
    
    # Si no detectó intención específica, buscar en metodología
    buscar_metodologia(question)

# Loop interactivo
while True:
    try:
        query = input("\nPregunta: ").strip()
        
        if not query:
            continue
        
        if query.lower() in ['salir', 'exit', 'quit']:
            print("\nHasta luego!")
            break
        
        print("\nRespuesta:")
        inicio = time.time()
        responder_pregunta(query)
        tiempo_total = time.time() - inicio
        
        print(f"\n\nTiempo de procesamiento: {tiempo_total:.2f} segundos\n")
        
    except KeyboardInterrupt:
        print("\n\nHasta luego!")
        break
    except Exception as e:
        print(f"\nError: {e}")
        continue
