import json

# Contar objetos en el JSON
with open("data/PIBOT_InformacionV2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total de objetos: {len(data)}")

# Contar preguntas totales
total_preguntas = sum(len(item["preguntas"]) for item in data)
print(f"Total de preguntas: {total_preguntas}")
print(f"Promedio de preguntas por objeto: {total_preguntas/len(data):.1f}")
