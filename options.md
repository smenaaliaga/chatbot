# Opciones de arquitectura

## Opciones de router

### 1) Reglas + erxtracción de slots (cero-LLM, ultra-rápido)

- __Qué es__: Conjunto de reglas/regex + diccionario de alias que detectan `métricas`, `periodo` y "palabras-señal" (p. ej., valor, variación, desestacionalizado).
- __Cuándo conviene__: cuando se busca la __mínima latencia__ y los dominios/formatos están bien acotados.
- __Pros__: deterministas, barato y rápido.
- __Contras__: menos flexible con lenguaje libre; hay que mantener reglas.

__Recomendado__: combinar reglas con un vector-search de intents (ver #2) para robustez.

### 2) Embeddings + nearests-centroid/knn para intents

- __Qué es__: define 2-5 intents (p. ej., `dato_concreto`, `metodología`, `mixto`, `otro`), crea embeddings de ejemplos y en línea asignas la consulta al intent más cercano (coseno/knn).
- __Pros__: ligero, funciona bien con paráfrasis.
- __Contras__: sin entrenamiento supervisado, puede confundirse con clases próximas.

__Base teórica__: _sentence-transformers_ y E5 son estándares para esto (multilangue y robusto).

### 3) Clasificador supervisado

- __Qué es__: entrenas un clasificador de intención con pocos ejemplos, fine-tune de _sentence-transformers_ y luego una cabeza lineal (LogReg).
- __Por qué destaca__: requiere pocos datos, rendimiento alto y despliegue simple. SetFit está bien documentado y probado en intent classification.
- __Pros__: preciso/estable vs. pruo embeddings; barato en inferencia (sin LLM).
- __Contra__: necesitar curar ejemplos y reentrenar si cambian intents.

### 4) Router con LLM (structured output / function calling)

- __Qué es__: el LLM devuelve un JSON del tipo `{"route": "API"|"RAG"|"Mixto", "metric":..., "period":...}`, tu backend ejecuta la(s) función(es) declaradas.
- __Pros__ muy flexible; puede planificar (p. ej., detectar "mixto").
- __Contras__: +latencia y costo (aun que con un 7B local se mantiene estable); hay que validar salidas.

### Recomendación

__Requisito__: baja latencia y exactitud en valores numéricos.

1. Router primario (cero-LLM):
    - Reglas + extracción de slots para `metric/period/operacion` + palabra-señal.
    - Si los slots son completos y válidos => rita API directa (y corta).
    - Si detecta patrones metodológicos o `slots` insuficientes => marcar metodología o mixto.
2. Router secundario (aprendizaje):
    - __Setfit__ para `datos_concreto` vs `metodolgía` vs `mixto` (añade confianza).
3. (Opcional) Router LLM solo cuando hay ambiguedad
    - LLM local (7B) con _structured output_ para desempatar casos grises (p. ej., inteciones mezcladas o faltan parámetros).

Se me ocurre que se puedan utilizar las tres opciones, y solo para la ambiguedad el caso Router LLM, y marcar en el chat que el Chatbot está pensando para aclarar la latencia.

### Detalles finos

- Confianza + fallbacks: si el clasificador devuelve `dato_concreto` con confianza $≥ \tau$ y los slots están completos → API; si no, repregunta mínima o pasa por LLM para completar slots.
- Multi-intento ("mixto"): ejecuta en paralelo API (dato) + RAG (metodolgía).
- BM25 + Vector en RAG: en dominios con jerga exacta, usa híbrido o re-rank para mejorar el _top-k_ antes del LLM.
- Evaluación continua: mide el _routing accuracy_, latencia por ruta y _drop rate_ de re-preguntas.


