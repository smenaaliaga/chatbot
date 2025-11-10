from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Modelo de embeddings
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# Textos de ejemplo
texts = [
    "El IPC mide la variación de precios de bienes y servicios.",
    "El PIB representa la producción total de la economía."
]
metadatas = [
    {"fuente": "Manual IPC 2023", "página": 14},
    {"fuente": "Cuentas Nacionales", "año": 2024}
]

# Crear vector store
vectorstore = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="./chroma_example"
)

# Buscar
query = "¿Cómo se mide la inflación?"
docs = vectorstore.similarity_search(query, k=2)

for d in docs:
    print("→", d.page_content, d.metadata)
