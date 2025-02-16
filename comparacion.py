import os
import re
import matplotlib.pyplot as plt

# Ruta base donde están los modelos
ruta_base = "modelos"

# Diccionario para almacenar los resultados
resultados_modelos = {}

# Recorre todas las carpetas dentro de "modelos"
for modelo in os.listdir(ruta_base):
    ruta_resultados = os.path.join(ruta_base, modelo, "resultados.txt")
    
    if os.path.isfile(ruta_resultados):
        try:
            with open(ruta_resultados, "r", encoding="utf-8") as file:
                contenido = file.read()
        except UnicodeDecodeError:
            with open(ruta_resultados, "r", encoding="latin-1") as file:
                contenido = file.read()

        # Expresiones regulares para extraer los datos
        patrones = {
            "Error Promedio Global": r"Error promedio global:\s*([-.\d]+)",
            "Desviación Estándar Global": r"Desviación estándar global:\s*([-.\d]+)",
            "Error Relativo Porcentual Promedio": r"Error relativo porcentual promedio global:\s*([-.\d]+)%",
            "MAPE Promedio": r"MAPE promedio:\s*([-.\d]+)",
            "Error Más Grande": r"El error más grande cometido:\s*([-.\d]+)"
        }
        
        # Extraer valores
        datos = {}
        for key, patron in patrones.items():
            match = re.search(patron, contenido)
            if match:
                datos[key] = float(match.group(1))
            else:
                datos[key] = None  # Si no se encuentra, se deja como None
        
        # Guardar en el diccionario general
        resultados_modelos[modelo] = datos

# Mostrar los modelos encontrados
print("Modelos encontrados:", list(resultados_modelos.keys()))

# Pedir al usuario qué modelos comparar
modelos_seleccionados = input("Ingrese los modelos a comparar separados por coma (o ENTER para todos): ").strip()
if modelos_seleccionados:
    modelos_seleccionados = [m.strip() for m in modelos_seleccionados.split(",")]
else:
    modelos_seleccionados = list(resultados_modelos.keys())

# Pedir la métrica a graficar
metricas_disponibles = list(patrones.keys())
print("Métricas disponibles:", metricas_disponibles)
metrica = input("Ingrese la métrica a graficar: ").strip()

if metrica not in metricas_disponibles:
    print("Métrica no válida. Saliendo...")
    exit()

# Preparar los datos para graficar
modelos = []
valores = []

for modelo in modelos_seleccionados:
    if modelo in resultados_modelos and resultados_modelos[modelo][metrica] is not None:
        modelos.append(modelo)
        valores.append(resultados_modelos[modelo][metrica])

# Graficar los resultados
plt.figure(figsize=(10, 5))
plt.bar(modelos, valores, color="royalblue")
plt.xlabel("Modelos")
plt.ylabel(metrica)
plt.title(f"Comparación de {metrica} entre modelos")
plt.xticks(rotation=90)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
