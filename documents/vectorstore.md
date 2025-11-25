# Opciones de almacenamiento de VectorStore

## ¿Qué es un VectorStore?

### ¿Qué bases de datos vectoriales hay?

## ¿Cómo funciona Chroma?

La base de datos vectorial

1. Convierte la pregunta a embedding: [0.12, -0.34, ...]
2. Busca los 3 vectores más cercanos en la base de datos
3. Devuelve los documentos originales

### ¿Qué puede almacenar Chroma?"

- `id`: string único
- `embedding`: la representación númerica (matriz de float)
- `document`: texto crudo
- `metadata`: diccionario clave valor con cualquier dato de interés

### ¿Qué información le puedo pedir?

- `distances` / `similarities`: scores de cercanía
- `embeddings`

## ¿Qué información metodologica puedo almacenar que entregue valor a la busqueda?

- Las respuestas o definición
- Las preguntas 
- Conceptos claves
- Fuente

## ¿Qué opciones de almacenamiento hay para este caso?

### 1) Indexar respuesta 

### 2) Indexar pregunta

- Se almacena cada variante de pregunta como documento en Chroma.
- En la `metadata` se guarda el texto de la respuesta canónica y un `grupo_id`.
- En consulta: haces top k → (opcional) re-rank → colapsas por `grupo_id` y devuelve la respuesta canónica del grupo ganador.
- Pros: latencia bajísima, no necesita LLM para re-redactar.
- Se puede almacenar nuevas formas de preguntas de otros usuarios para enriquecer el vectorstore.

### 3) Indexación múltiple

### 4) Índice lexical (BM25) en paralelo

## Sugerencias para almacenar documento en Vectorstore

1. Enriquecer los documentos con contexto mínimo (prefijos/titulos)

```
"PIB · tres enfoques — En Chile se calcula desde tres enfoques: producción, gasto e ingreso."
```

## Crear un Chatbot tipo FAQ Semántico

- Ventaja: Es muy preciso y elimina la posibilidad de alucinaciones.
- Desventaja: Limitado para preguntas compuestas o complejas.