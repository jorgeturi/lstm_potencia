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

# Selección de los 5 mejores modelos en diferentes métricas
mejores_mape = sorted(resultados_modelos.items(), key=lambda x: x[1]["MAPE por columna"] or float('inf'))[:5]
mejores_std = sorted(resultados_modelos.items(), key=lambda x: x[1]["Desviación estándar por columna"] or float('inf'))[:5]
mejores_err3 = sorted(resultados_modelos.items(), key=lambda x: x[1]["Errores <= 3"] or -float('inf'), reverse=True)[:5]
mejores_err5 = sorted(resultados_modelos.items(), key=lambda x: x[1]["Errores <= 5"] or -float('inf'), reverse=True)[:5]
mejores_error_max = sorted(resultados_modelos.items(), key=lambda x: x[1]["Error Más Grande"] or float('inf'))[:5]

# Imprimir los resultados
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