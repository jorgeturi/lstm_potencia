####
### OJO CUANDO IMPRIMO TENGO QUE PONER COL 3 SI ES DE LOS 15 MIN, SINO 0 PARA EL DE LA HORA
###
###


import os
import re
import sys
import numpy as np

# Añadir carpeta hermana 'prediccion_potencia' al path
ruta_actual = os.path.dirname(__file__)
ruta_hermana = os.path.abspath(os.path.join(ruta_actual, '..', 'prediccion_potencia'))
sys.path.append(ruta_hermana)

# Importar módulos
from tools import red_principal as red_local
from tools import red_principal_h as red_h

# Mapeos para salida (A) e entrada (B)
A_h = {0: 1, 1: 6, 2: 12, 9:1}
B_h = {
    0: 1, 1: 6, 2: 12, 3: 24,
    4: 24*2, 5: 24*5, 6: 24*7
}

A_no_h = {0: 4, 1: 24, 2: 48}
B_no_h = {
    0: 4, 1: 24, 2: 48, 3: 96,
    4: 192, 5: 480, 6: 672
}

# Rangos de datos (dividido por 4 si es h)
train_end = 71500
val_len = 18100
test_len = 11615


def plot_rango_predicciones(resultados, modelos, inicio, fin, paso=1):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14,6))
    for modelo in modelos:
        y_true = resultados[modelo]['y_real'][inicio:fin]
        y_pred = resultados[modelo]['y_pred'][inicio:fin]

        plt.plot(range(inicio, fin, paso), y_pred[::paso, 3], label=f'Predicción {modelo}')
        plt.plot(range(inicio, fin, paso), y_true[::paso, 3], linestyle='dashed', label=f'Real {modelo}')

    plt.xlabel('Índice temporal')
    plt.ylabel('Potencia')
    plt.title(f'Comparación de predicciones y reales de modelos del índice {inicio} al {fin}')
    plt.legend()
    plt.grid(True)
    plt.show()






    # Ahora gráfico con predicción desplazada 1 y real alineada al final
    plt.figure(figsize=(14,6))
    for modelo in modelos:
        y_true = resultados[modelo]['y_real'][inicio:fin-1]  # real desde inicio+1 a fin (desplazado)
        y_pred = resultados[modelo]['y_pred'][inicio+1:fin]  # predicción desde inicio a fin-1 (desplazado)
        
        indices = range(inicio+1, fin)  # para alinear ambos
        
        plt.plot(indices, y_pred[::paso, 3], label=f'Predicción desplazada {modelo}')
        plt.plot(indices, y_true[::paso, 3], linestyle='dashed', label=f'Real desplazada {modelo}')
    
    plt.xlabel('Índice temporal')
    plt.ylabel('Potencia')
    plt.title(f'Predicción desplazada vs Real desplazada del índice {inicio+1} al {fin}')
    plt.legend()
    plt.grid(True)
    plt.show()


def interpolar_por_bloques(valores, pasos_por_bloque):
    interpolado = []
    for i in range(len(valores) - 1):
        # Interpola entre el valor actual y el siguiente
        bloque = np.linspace(valores[i], valores[i+1], pasos_por_bloque, endpoint=False)
        interpolado.extend(bloque)
    # Agregar último valor extendido
    interpolado.extend([valores[-1]] * pasos_por_bloque)
    return np.array(interpolado)

def es_modelo_h(nombre):
    return nombre.strip().lower().endswith("h")

def extraer_parametros(nombre_modelo):
    nombre_limpio = nombre_modelo[:-1] if es_modelo_h(nombre_modelo) else nombre_modelo
    partes = re.findall(r'\d+', nombre_limpio)
    print(f"modelo={modelo}, partes extraídas: {partes}")
    if len(partes) < 4:
        raise ValueError(f"No se pudieron extraer A-B-C-D del nombre: {nombre_modelo} => partes encontradas: {partes}")
    return list(map(int, partes[:4]))

# Carpeta con los modelos
ruta_modelos = "modelos"
modelos_disponibles = sorted(os.listdir(ruta_modelos))

print("Modelos encontrados:")
for i, m in enumerate(modelos_disponibles):
    print(f"{i}. {m}")

# Selección
entrada = input("\nIngrese los modelos a ejecutar separados por coma o ENTER para todos: ").strip()
modelos_a_ejecutar = [modelos_disponibles[int(i.strip())] for i in entrada.split(',')] if entrada else modelos_disponibles
resultados = {} ###PARA GUARDAR RESULTADOS
# Bucle principal
for modelo in modelos_a_ejecutar:
    print(f"\n🔧 Procesando modelo: {modelo}")
    try:
        es_h = es_modelo_h(modelo)
        print("es modelo con h", es_h)
        A_id, B_id, C_id, D_id = extraer_parametros(modelo)

        ventana_salida = A_h[A_id] if es_h else A_no_h[A_id]
        ventana_entrada = B_h[B_id] if es_h else B_no_h[B_id]
        modulo = red_h if es_h else red_local

        # Cargar datos y ventana
        dias = [0, 1, 2, 3, 4, 5, 6]  # 0 domingo
        horas = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        datos = modulo.cargar_datos_especificos('potencias.csv', dias_semanales=dias, horas=horas)
        X, y = modulo.crear_ventana(datos, ventana_entrada, ventana_salida)

        # Calcular índices
        factor = 1 if not es_h else 0.25
        t_end = int(train_end * factor)
        v_start = t_end
        v_end = v_start + int(val_len * factor)
        tst_start = v_end
        tst_end = tst_start + int(test_len * factor)

        # Particiones
        Xtrain, ytrain = X[0:t_end], y[0:t_end]
        Xval, yval = X[v_start:v_end], y[v_start:v_end]
        Xtest, ytest = X[tst_start:tst_end], y[tst_start:tst_end]

        print(f"➡️  Entrada: {ventana_entrada}, Salida: {ventana_salida}, Batch: {C_id}, Variación: {D_id}")
        print(f"📊 Train: {Xtrain.shape}, Val: {Xval.shape}, Test: {Xtest.shape}")

        # Cargar y evaluar modelo
        print("cargando " + modelo)

        modelo_keras = modulo.cargar_modelo("modelos/"+modelo+"/modelo")
        scal = modulo.cargar_escaladores(os.path.join(ruta_modelos, modelo, "scalers.pkl"))
        Xtest_n = Xtest.copy()
        Xtest_n[:, :, 0] = scal['scaleractiva'].transform(Xtest[:, :, 0])
        

        print("inicio prediccion de " + modelo)

        prediccionestest_n = modelo_keras.predict(Xtest_n)

        # Invertir escala de salida
        prediccionestest = scal['salidas'].inverse_transform(prediccionestest_n)

        mejor_que_persistente = 0
        total_comparaciones = 0

        for i in range(1, ytest.shape[0]):
            valor_real = ytest[i, -1]            # último paso real
            valor_anterior = ytest[i - 1, -1]   # último paso persistente (valor anterior mismo paso)
            pred_modelo = prediccionestest[i, -1]  # último paso predicho

            diferencia_real = abs(valor_real - valor_anterior)

            # Solo comparar si diferencia real es >= 4 kW
            if diferencia_real >= 5:
                error_modelo = abs(valor_real - pred_modelo)
                error_persistente = abs(pred_modelo - valor_anterior)  # acá corregí para comparar con valor real, no con predicción

                if error_modelo < error_persistente:
                    mejor_que_persistente += 1

                total_comparaciones += 1

        if total_comparaciones > 0:
            porcentaje_mejora = (mejor_que_persistente / total_comparaciones) * 100
        else:
            porcentaje_mejora = 0

        #print(f"Total de comparaciones (última columna, dif ≥ 5kW): {total_comparaciones}")
        #print(f"Cantidad mejores que persistente: {mejor_que_persistente}")
        #print(f"Porcentaje de mejora sobre persistencia: {porcentaje_mejora:.2f}%")









        resultados[modelo] = {
            'y_real': ytest,
            'y_pred': prediccionestest,
            'porcentaje mejora': porcentaje_mejora,
            'Total de comparacione': total_comparaciones,
            'Cantidad mejores que persistente' : mejor_que_persistente
        }
        #pred = modulo.evaluar_modelo(modelo_keras, Xtest, ytest, modelo)

        print(f"✅ Modelo {modelo} ejecutado con éxito.")

    except Exception as e:
        print(f"❌ Error al procesar {modelo}: {e}")


# Ahora para plotear

import numpy as np
import matplotlib.pyplot as plt

modelo_h = [m for m in resultados if m.endswith('h')][0]
modelo_no_h = [m for m in resultados if not m.endswith('h')][0]

y_true_h = resultados[modelo_h]['y_real']
y_pred_h = resultados[modelo_h]['y_pred']

y_true_no_h = resultados[modelo_no_h]['y_real']
y_pred_no_h = resultados[modelo_no_h]['y_pred']

h_expanded = np.repeat(y_pred_h[:, 0], 4)
h_real = np.repeat(y_true_h[:, 0], 4)

# Alinear quitando 14 pasos iniciales (por algún motivo que ya manejás)
h_expanded_aligned = h_expanded[14:]
h_real_expanded_aligned = h_real[14:]

# Índices donde empieza cada nuevo valor (0, 4, 8, ...)
indices_marcados = np.arange(0, len(h_expanded_aligned), 4)

plt.figure(figsize=(12, 5))
#plt.plot(h_expanded_aligned, label='Predicción h expandida')
#plt.plot(h_real_expanded_aligned, label='Real h expandido')
#plt.plot(y_pred_no_h[:, 3], label='Predicción no h')
#plt.plot(y_true_no_h[:, 0], label='Real no h 1 paso')

#plt.plot(y_true_no_h[:, 3], label='Real no h')
plt.plot(y_pred_h, label='Predicción h ')
plt.plot(y_true_h, label='Real h ')

# 🔴 Marcar los puntos iniciales de cada valor expandido
#plt.scatter(indices_marcados, h_expanded_aligned[indices_marcados], color='red', marker='o', label='Valor pred h mantenido')
#plt.scatter(indices_marcados, h_real_expanded_aligned[indices_marcados], color='green', marker='x', label='Valor real h mantenido')

plt.title("Comparación: Predicción modelo h vs modelo no h (valores mantenidos marcados)")
plt.legend(loc="best")
plt.grid(True)
plt.show()



# Aplica interpolación a las predicciones y reales
h_interp = interpolar_por_bloques(y_pred_h[:, 0], 4)
h_real_interp = interpolar_por_bloques(y_true_h[:, 0], 4)

# Alineación (si sigue siendo necesario)
h_interp_aligned = h_interp[14:]
h_real_interp_aligned = h_real_interp[14:]
print("las predicciones", len(y_pred_h))
print("saltendo las primeas 86", len(h_interp[14:]))
print("el original tenia",  len(y_true_no_h))

# Graficar resultados interpolados
plt.figure(figsize=(12, 5))
plt.plot(h_interp_aligned, label='Predicción h interpolada')
plt.plot(h_real_interp_aligned, label='Real h interpolado')
#plt.plot(y_pred_no_h[:, 3], label='Predicción no h')
plt.plot(y_true_no_h[:, 3], label='Real no h')
plt.title("Interpolación lineal entre predicciones de modelo h vs modelo no h")
plt.legend()
plt.grid(True)
plt.show()


offset = 14 // 4

pred_no_h_cada_hora = y_pred_no_h[offset::4, 3]   # columna 3 = 1h adelante
real_no_h_cada_hora = y_true_no_h[offset::4, 3]

# Emparejar longitudes
min_len = min(len(y_pred_h), len(pred_no_h_cada_hora))

pred_no_h_cada_hora = pred_no_h_cada_hora[:min_len]
real_no_h_cada_hora = real_no_h_cada_hora[:min_len]
pred_h = y_pred_h[:min_len, 0]
real_h = y_true_h[:min_len, 0]

plt.figure(figsize=(12,5))
plt.plot(pred_h, label='Predicción h')
plt.plot(pred_no_h_cada_hora, label='Predicción no h (cada 1h)')
plt.plot(real_h, label='Real h')
plt.plot(real_no_h_cada_hora, label='Real no h (cada 1h)', linestyle='dotted')
plt.title("Comparación: modelo h vs modelo no h (cada 1h)")
plt.legend()
plt.show()





# Extraer nombres de modelos
modelo_h = [m for m in resultados if m.endswith('h')][0]
modelo_no_h = [m for m in resultados if not m.endswith('h')][0]

# Cargar resultados
y_true_h = resultados[modelo_h]['y_real']
y_pred_h = resultados[modelo_h]['y_pred']
y_true_no_h = resultados[modelo_no_h]['y_real']
y_pred_no_h = resultados[modelo_no_h]['y_pred']

# Valores por hora (predicción y real)
valores_h = y_pred_h[:, 0]
valores_real_h = y_true_h[:, 0]

# Crear ejes para interpolar
x_h = np.arange(len(valores_h))          # [0, 1, 2, ..., 2884]
x_interp = np.linspace(0, len(valores_h) - 1, len(valores_h) * 4)  # 4 puntos por cada hora

# Interpolación lineal
interp_pred_h = np.interp(x_interp, x_h, valores_h)
interp_real_h = np.interp(x_interp, x_h, valores_real_h)

# Alinear con el modelo no_h (recortar desfase inicial)
offset = 14  # el mismo desfase que ya calculaste antes
interp_pred_h_aligned = interp_pred_h[offset:]
interp_real_h_aligned = interp_real_h[offset:]

# Recortar para que todas las series tengan el mismo largo
min_len = min(len(interp_pred_h_aligned), y_pred_no_h.shape[0])
interp_pred_h_aligned = interp_pred_h_aligned[:min_len]
interp_real_h_aligned = interp_real_h_aligned[:min_len]
y_pred_no_h_trimmed = y_pred_no_h[:min_len, 3]
y_true_no_h_trimmed = y_true_no_h[:min_len, 3]

# Plot final
plt.figure(figsize=(12, 5))
plt.plot(interp_pred_h_aligned, label='Predicción h interpolada')
plt.plot(interp_real_h_aligned, label='Real h interpolada')
#plt.plot(y_pred_no_h_trimmed, label='Predicción no h (1h adelante)')
#plt.plot(y_true_no_h_trimmed, label='Real no h (1h adelante)')
plt.title("Interpolación modelo h vs predicción no h")
plt.legend()
plt.show()




import matplotlib.pyplot as plt

# Extraer nombres de modelos y sus mejoras
nombres_modelos = list(resultados.keys())
mejoras = [resultados[m]['porcentaje mejora'] for m in nombres_modelos]

# Crear gráfico de barras
plt.figure(figsize=(10, 5))
bars = plt.bar(nombres_modelos, mejoras, color='skyblue')
plt.xticks(rotation=90, ha='right')
plt.ylabel('% de mejora sobre persistente')
plt.title('Comparación de mejora saltos 5kw')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar valor encima de cada barra
for bar, mejora in zip(bars, mejoras):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{mejora:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

print("Modelos disponibles:")
for i, m in enumerate(resultados.keys()):
    print(f"{i}: {m}")

entrada_modelos = input("\nIngrese índices de modelos a comparar separados por coma (ej: 0,2) o ENTER para todos: ").strip()
if entrada_modelos:
    indices = [int(i) for i in entrada_modelos.split(',')]
    modelos_seleccionados = [list(resultados.keys())[i] for i in indices]
else:
    modelos_seleccionados = list(resultados.keys())

while True:
    try:
        inicio = int(input("Ingrese índice inicial del rango (>=0): "))
        fin = int(input("Ingrese índice final del rango (> inicio): "))
        if 0 <= inicio < fin:
            break
        else:
            print("Rango inválido, asegúrese que 0 <= inicio < fin.")
    except ValueError:
        print("Por favor ingrese números enteros válidos.")

plot_rango_predicciones(resultados, modelos_seleccionados, inicio, fin)