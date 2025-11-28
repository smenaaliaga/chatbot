# Preguntas compuestas

## ¿Qué es una "pregunta atómica"?

Una pregunta atómica es una parte de la consulta del usuario que se puede responder con un solo flujo de procesamiento (una llamda a la API o una búsqueda en la base de conocimiento), sin tiener que mezclar resultados de flujos distintos.

Ejemplos:

✅ Atómica — solo dato, una métrica, un período coherente

“¿Cuál fue el IMACEC de septiembre de 2024?”
→ Un solo llamado a tu API IMACEC, 2024-09.

✅ Atómica — dato con varios periodos de la misma serie

“¿Cuál fue el IMACEC en 2023 y 2024?”
→ Podrías resolverlo en una sola llamada a la API (rango de años) → una atómica (aunque la respuesta tenga 2 números).

✅ Atómica — solo metodología

“¿Cómo se calcula el PIB real encadenado?”
→ Un solo flujo RAG contra tu base de conocimiento.

❌ NO atómica — mezcla dato + metodología

“¿Cuál fue el IMACEC de septiembre de 2024 y cómo se calcula el PIB encadenado?”
→ Son dos flujos distintos: API datos + RAG.
Debes dividir en 2 preguntas atómicas.

❌ NO atómica — varias métricas

“Dame el IMACEC y la inflación anual de 2023.”
→ Si tu pipeline actual asume “una métrica por pregunta”, aquí conviene dividir en 2 atómicas.
(Más adelante podrías soportar varias métricas en una pregunta, pero complica el orquestador).

## Patrones típico de consultas compuestas

Se pueden ver 4 grandes patrones:

### Multi-pregunta clásica (varios "¿...?")

Ejemplo:

`¿Cuál fue la inflación anual en 2023? ¿Y cómo se calcula el PIB encadenado?`

Patrón trivial:

- Separar por `?`,
- Cada trozo que tenga texto → una pregunta atómica candidata.
- Luego el NLU hace intención/métrica/período.

## NLU (Natural Language Understanding)

Es la parte de la IA que se encarga de entender qué está diciendo el usuario en lenguaje humano y traducirlo a algo que una máquina pueda usar.

El NLU es la primera capa: toma el texto crudo y lo transforma en información estructurada. Luego, viene el orquestador (decide qué servicios llamar: API/RAG) y, finalmente, un LLM o plantilla que hace la respuesta en lenguaje natural al usuario.

`Usuario → NLU → Plan/Orquestador → API / KB → Respuesta.`

El NLU debe:

### 1. Descomponer en preguntas atómicas

Si el usuario mezcla cosas: `¿Cuál fue el IMACEC de septiembre 2024 y cómo se calcula el PIB real encadenado?`, entonces el NLU debe producir:

```
[
    "¿Cuál fue el IMACEC de septiembre 2024?",
    "¿Cómo se calcula el PIB real encadenado?"
]
```

### 2. Para cada pregunta atómica, debe detectar:

- Intención (intent):
    - `data`: quiere el número de una serie.
    - `method`: quiere una explicación metodológica.
    - `other`: resto (small talk, etc).
- Slots/entidades:
    - `metric`: IMACEC, PIB, PIB_real, etc.
    - `period`: año/mes/rango normalizado.
    - `tipo`: agregación/desestacionalizado.
- Resultado:
```
{
  "id": 1,
  "text": "¿Cuál fue el IMACEC de septiembre de 2024?",
  "intent": "data",
  "slots": {
    "metric_id": "IMACEC",
    "period": { "type": "monthly", "year": 2024, "month": 9 },
    "tipo": "desestacionalizado"
  }
}
```

Este JSON pasa al orquestador.

## Investigación al respecto

Lo que está súper estudiado en academia/insdustria:

- Intent + Slots conjuntos → “Joint intent classification & slot filling”.
- Multi-intent → cuando una misma frase trae varias intenciones.
- Question decomposition → cuando una pregunta es compleja y conviene partirla.

Resumen literario:

### Intent + Slots

__Idea__: Un solo modelo (tipo BERT) que, a la vez: 
- clasifica la intención (intent) de la pregunta, y
- etiqueta cada token con un solo slot (NER) por palabra (métrica, periodo, tipo, etc.).

Esto se conoce como _joint intent classification and slot filling_:

- Chen et al. (2019) proponen BERT para intent + slots conjuntos y muestran mejoras claras sobre modelos RNN/attention antiguos ([arXiv](https://arxiv.org/abs/1902.10909?utm_source=chatgpt.com)).

Cómo lo usaría:

- Base: `bert-base-multilingual-cased` como encoder.
- Encima:
    - capa de clasificación para intent (`data`, `method`).
    - capa de etiquetado tipo CRF o softmax por token para slots.

#### Como funciona (idea intuitiva):

Se toma la frase:

```
"¿Cuál fue el IMACEC de septiembre de 2024?"
```

Se tokeniza y pasa por BERT:

```
[CLS] ¿ cuál fue el IMACEC de septiembre de 2024 ? [SEP]
```

BERT te da vectores:

- `h_cls` → el embedding del token [CLS] (resumen de toda la frase).
- `h_1...h_n` → embeddings de cada token.

Encima de eso pasa por dos cabezas: 

1. Cabeza de intent (frase completa):

```
intent = softmax(W_intent · h_cls)
→ "data"
```

2. Cabeza de slots (token a token, con softmax o CRF):

```
slot_t = softmax(W_slot · h_t)
→ [O, O, O, B-METRIC, O, B-DATE, I-DATE, O, ...]
```

Todo se entrena junto con una __pérdida combinada__:

```
Loss_total = Loss_intent + Loss_slots
```

Eso es lo que en muchos papers y repos llaman “Joint BERT for Intent Classification and Slot Filling”.

### Segmentación Multi-intent, conultas con varias preguntas

En la literatura reciente de SLU lo trata como "multi-intent detection + slots por intent":

- Modelos que hacen detection + slots conjuntos para multi-intent (ej SPM: split-parsing method: [ACL](https://aclanthology.org/2023.acl-industry.64/?utm_source=chatgpt.com)).
- La idea común: dividir la oración en fragmentos por intent, y para cada fragmento hacer intent+slots.

Entonces, el usuario mete varias preguntas o intenciones en la misma oración:

`"¿Cuál fue el IMACEC de septiembre de 2024 y cómo se calcula el PIB real?"`

Aquí hay dos intenciones:

- Intent 1: `data` (IMACEC septiembre 2024).
- Intent 2: `method` (cómo se calcula el PIB real).

A esto se le llama:

- __Multi-intent__ (la frase tiene variasd intenciones), y
- La tarea es __segmentar__ o __decomponer__ la frase en sub-preguntas atómicas.

La __segmentación multi-intent__ consiste en:

1. Detectar que la oración tienen varias intenciones.
2. Decidir __dóndne cortar__ (por conectores, estructura, etc.).
3. Asignar a cada __segmento__ su propio `intent + slots`.

Entonces, el __multi-intent__ realiza la siguiente anotación:

`[Q1, Q1, Q1, Q1, Q1, Q1, Q1, Q1, Q1, Q2, Q2, Q2, Q2, Q2...]`

Entonces, en terminos de flujo:

- Joint BERT clásico asume:

    _una oración → un intent + una secuencia de slots._

- Segmentación multi-intent dice:

    _una oración → varias sub-oraciones → cada sub-oración tiene su intent + slots._


### Como se debe conjugar ambas

```
Texto usuario
   ↓
Segmentación multi-intent
   ↓ (sub-pregunta 1, 2, 3...)
Joint BERT (intent+slots) por sub-pregunta
   ↓
JSON estructurado por pregunta atómica
```

Imagina que:

- La segmentación multi-intent es como cortar una torta en porciones:
    
    _“esto es trozo 1, trozo 2…”_

- El Joint BERT es el catador que prueba cada porción y dice:

    _“esta es torta de chocolate con nueces” (intent+slots), “esta es de frutilla con crema”, etc._

Primero cortas la torta en porciones lógicas (segmentación).
Luego categorizas bien cada porción (Joint BERT).

### Descomposición de preguntas complejas (questions decomposition)

Otra línea de investigación que cuadra mucho con lo que se busca es la de _question decomposition_ en QA compleja:

- Método tipo __Question Decomposition Tree (QDT)__ que representen preguntas complejas como un árbol de subpreguntas simples, y que muestran mejoras fuertes en QA sobre base de conocimieto ([arXiv](https://arxiv.org/abs/2306.07597?utm_source=chatgpt.com)).
- Trabajo de Perez et al (020) sobre descomposición no supervisada de preguntas complejas, mostrando que dividir la pregunta y responder sub-preguntas mejora el F1 de modelos como BERT y RoBERTa ([Scott Wen-tau](https://scottyih.org/files/emnlp2020-decompQ.pdf?utm_source=chatgpt.com)).
- Estudios más recientes en KBQA destacan que la descomposición reduce la complejidad de razonamiento y mejora la precisión en escenarios multi-hop ([MDPI](https://www.mdpi.com/2073-8994/17/7/1022?utm_source=chatgpt.com)).


```
┌──────────────────────────┐
│        Usuario           │
│  "¿PIB 2024 y cómo se    │
│   calcula el IMACEC?"    │
└────────────┬─────────────┘
             │ texto libre
             ▼
┌──────────────────────────┐
│           NLU            │
│ ┌──────────────────────┐ │
│ │ 1) Segmentador       │ │  → divide en preguntas atómicas
│ │    (decomposición)   │ │
│ ├──────────────────────┤ │
│ │ 2) Modelo conjunto   │ │  → para cada pregunta:
│ │    Intención + Slots │ │     - intención (dato / método / otra)
│ │    (métrica, periodo │ │     - slots: métrica, periodo, país, etc.
│ │     país, etc.)      │ │
│ └──────────────────────┘ │
└────────────┬─────────────┘
             │ lista de preguntas estructuradas (JSON)
             ▼
┌──────────────────────────┐
│       Orquestador        │
│  - Si intent = dato   → API series       │
│  - Si intent = método → RAG / KB         │
└────────────┬─────────────┘
             │ respuestas parciales
             ▼
┌──────────────────────────┐
│   Generador de respuesta │
│  - arma texto final      │
│  - explica datos + met.  │
└──────────────────────────┘
```

El NLU es el bloque que convierte texto en estructura.