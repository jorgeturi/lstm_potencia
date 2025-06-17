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
            "Error Más Grande": r"El error más grande cometido:\s*([-.\d]+)",
            "MAPE por columna": r"MAPE por columna: \[([\d.,' %]+)\]",
            "Desviación estándar por columna": r"Desviación estándar por columna: \[([\d.,' %]+)\]",
            "Errores <= 3": r"Cantidad de errores menores o iguales a 3: \d+ \(([-.\d]+)%\)",
            "Errores <= 5": r"Cantidad de errores menores o iguales a 5: \d+ \(([-.\d]+)%\)"
        }
        
        # Extraer valores
        datos = {}
        for key, patron in patrones.items():
            match = re.search(patron, contenido)
            if match:
                if key in ["MAPE por columna", "Desviación estándar por columna"]:
                    valores = [float(x.replace("'", "").strip('%')) for x in match.group(1).split(',')]
                    datos[key] = valores[-1] if valores else None
                else:
                    datos[key] = float(match.group(1))
            else:
                datos[key] = None
        
        resultados_modelos[modelo] = datos

# Mostrar los modelos encontrados
print("Modelos encontrados:", list(resultados_modelos.keys()))

# Selección de modelos
modelos_seleccionados = input("Ingrese los modelos a comparar separados por coma (o ENTER para todos): ").strip()
if modelos_seleccionados:
    modelos_seleccionados = [m.strip() for m in modelos_seleccionados.split(",")]
else:
    modelos_seleccionados = list(resultados_modelos.keys())

# Selección de métrica
metricas_disponibles = list(patrones.keys())
print("Métricas disponibles:", metricas_disponibles)
metrica = input("Ingrese la métrica a graficar (o 'varias' para graficar múltiples): ").strip()

# Función para detectar si es modelo "h"
def es_modelo_h(nombre):
    return nombre.lower().strip().endswith("h")

# ---------------------------------------------
# OPCIÓN 1: Graficar una sola métrica
# ---------------------------------------------
if metrica != "varias":
    if metrica not in metricas_disponibles:
        print("Métrica no válida. Saliendo...")
        exit()

    modelos = []
    valores = []
    colores = []

    for modelo in modelos_seleccionados:
        if modelo in resultados_modelos and resultados_modelos[modelo][metrica] is not None:
            modelos.append(modelo)
            valores.append(resultados_modelos[modelo][metrica])
            colores.append("steelblue" if es_modelo_h(modelo) else "darkorange")

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.bar(modelos, valores, color=colores)
    plt.xlabel("Modelos")
    plt.ylabel(metrica)
    plt.title(f"Comparación de {metrica} entre modelos")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    #plt.legend(["Con 'h'", "Sin 'h'"], loc="best")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# OPCIÓN 2: Graficar varias métricas juntas
# ---------------------------------------------
else:
    metricas_a_graficar = [
        "MAPE por columna",
        "Desviación estándar por columna",
        "Errores <= 3",
        "Errores <= 5",
        "Error Más Grande"
    ]

    fig, axs = plt.subplots(len(metricas_a_graficar), 1, figsize=(12, 5 * len(metricas_a_graficar)))

    for idx, met in enumerate(metricas_a_graficar):
        modelos = []
        valores = []
        colores = []
        for modelo in modelos_seleccionados:
            if modelo in resultados_modelos and resultados_modelos[modelo][met] is not None:
                modelos.append(modelo)
                valores.append(resultados_modelos[modelo][met])
                colores.append("steelblue" if es_modelo_h(modelo) else "darkorange")

        axs[idx].bar(modelos, valores, color=colores)
        axs[idx].set_title(met, fontsize=12)
        axs[idx].tick_params(axis='x', rotation=90)
        axs[idx].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

# ---------------------------------------------
# Ranking de modelos
# ---------------------------------------------
mejores_mape = sorted(resultados_modelos.items(), key=lambda x: x[1]["MAPE por columna"] or float('inf'))[:5]
mejores_std = sorted(resultados_modelos.items(), key=lambda x: x[1]["Desviación estándar por columna"] or float('inf'))[:5]
mejores_err3 = sorted(resultados_modelos.items(), key=lambda x: x[1]["Errores <= 3"] or -float('inf'), reverse=True)[:5]
mejores_err5 = sorted(resultados_modelos.items(), key=lambda x: x[1]["Errores <= 5"] or -float('inf'), reverse=True)[:5]
mejores_error_max = sorted(resultados_modelos.items(), key=lambda x: x[1]["Error Más Grande"] or float('inf'))[:5]

# Mostrar rankings
print("\nTop 5 modelos con menor Último valor de MAPE por columna:")
for modelo, datos in mejores_mape:
    print(f"{modelo}: {datos['MAPE por columna']}")

print("\nTop 5 modelos con menor Último valor de Desviación estándar por columna:")
for modelo, datos in mejores_std:
    print(f"{modelo}: {datos['Desviación estándar por columna']}")

print("\nTop 5 modelos con mayor Porcentaje de errores <= 3:")
for modelo, datos in mejores_err3:
    print(f"{modelo}: {datos['Errores <= 3']}%")

print("\nTop 5 modelos con mayor Porcentaje de errores <= 5:")
for modelo, datos in mejores_err5:
    print(f"{modelo}: {datos['Errores <= 5']}%")

print("\nTop 5 modelos con menor Error Más Grande:")
for modelo, datos in mejores_error_max:
    print(f"{modelo}: {datos['Error Más Grande']}")
