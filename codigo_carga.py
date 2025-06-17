import logging
from tools.logger_config import setup_logger
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os



if __name__ == "__main__":
    # codigo principal
    setup_logger()    
    from tools.red_principal import *
    
    path = "modelos/modelo 0.0.3.5/"
    ventana_entrada = 4
    
    ventana_prediccion = 4


    modelo = cargar_modelo(path + "modelo")

    if modelo is not None:  #si consegui el modelo
        modelo.summary()  
        scal = cargar_escaladores(path + "scalers.pkl")
        if scal is not None:
            #print("scalers:", scal)
            #dias = [0, 1, 2, 3, 4, 5, 6]  # 0 domingo
            dias = [0, 1, 2, 3, 4, 5, 6]  # 0 domingo
            horas = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]  # Ejemplo de horas
            df = cargar_datos_especificos('potencias.csv', dias_semanales=dias, horas=horas)
            print("tengo estos datos ",df.shape)
            X, y = crear_ventana(df[000:200000], 4, 4)
            """df = df.reset_index(drop=True)
            #df.iloc[:, 0] = df.iloc[:, 0].rolling(window=6, min_periods=1).mean()
            df2 = df.copy()
            #df2.loc[:, 'media_movil'] = df.iloc[:, 0].rolling(window=2, min_periods=1).mean()
            df.loc[:, 'media_movil'] = df.iloc[:, 0].rolling(window=4, min_periods=1).mean()
            # Crear una nueva columna basada en la media móvil
            df.loc[:, "media_ajustada"] = df["media_movil"]
            df["modificado"] = 0

            # Ajustar la media móvil en función de la diferencia relativa con el valor anterior
            for i in range(1, len(df)):  # Evitar i=-1 en la primera iteración
                diferencia = df.iloc[i, 0] - df.iloc[i - 1, 0]  # Diferencia con el valor anterior
                diferencia_relativa = (diferencia / df.iloc[i - 1, 0]) * 100  # Diferencia relativa en porcentaje
                
                #diferenciam = df.loc[i,"media_movil"] - df.loc[i-1,"media_movil"]

                # Si la diferencia relativa es mayor al 10%, ajustar con +15 o -15
                if diferencia_relativa > 13:
                    df.iloc[i, df.columns.get_loc("media_ajustada")] += 0.20*df.iloc[i, 0]
                    df.loc[i, "modificado"] = 1  # Marcar que fue modificado
                    
                elif diferencia_relativa < -13:
                    df.iloc[i, df.columns.get_loc("media_ajustada")] -= 0.20*df.iloc[i, 0]
                    df.loc[i, "modificado"] = 2  # Marcar que fue modificado
                

                if df.loc[i-1, "modificado"] == 1:  # Si el anterior fue modificado
                    df.iloc[i, df.columns.get_loc("media_ajustada")] += 0.11*df.iloc[i, 0]
                if df.loc[i-1, "modificado"] == 2:  # Si el anterior fue modificado
                    df.iloc[i, df.columns.get_loc("media_ajustada")] -= 0.11*df.iloc[i, 0]
            

                

            # Asignar la media ajustada a "activa"
            df["activa"] = df["media_ajustada"]
"""

            #df.iloc[:, 0] = df.iloc[:, 0].rolling(window=4, min_periods=1).mean()
            #X1, y1 = crear_ventana(df2[0:200000], ventana_entrada, ventana_prediccion)

            #X, y = crear_ventana(df[0:200000], ventana_entrada, ventana_prediccion)
            print("tengo datos: ",len(X))
            print("el shape es:", X.shape)
            ####### SEPARACION DE DATOS
            inicio_train = 0
            fin_train = 71500#17875#71500
            inicio_val = fin_train+1
            fin_val = fin_train+1+18100#4525#18100
            inicio_test = fin_val+1
            fin_test = inicio_test+1+11615#2903#11615
            # conjunto de validación
            Xval = X[inicio_val:fin_val]
            yval = y[inicio_val:fin_val]
            #conjunto de entrenamiento
            Xtrain = X[inicio_train:fin_train]
            ytrain = y[inicio_train:fin_train]
            # conjunto de validación
            Xtest = X[inicio_test:fin_test]
            ytest = y[inicio_test:fin_test]
            

            
            #X_n = escalar_entrada(X,scal)
            print("el scaler levantado es ")

            #print("mmin ", scal['scaleractiva'].data_min_)
            #print("max ", scal['scaleractiva'].data_max_)


            Xtest_n = Xtest.copy()
            Xtest_n[:, :, 0] = scal['scaleractiva'].transform(Xtest[:, :, 0])

            #Xtest_n[:, :, 1] = scal['l1'].transform(Xtest[:, :, 1])
            #Xtest_n[:, :, 2] = scal['l2'].transform(Xtest[:, :, 2])
            #Xtest_n[:, :, 3] = scal['l3'].transform(Xtest[:, :, 3])

            logging.info("inicio prediccion")

            prediccionestest_n = modelo.predict(Xtest_n)
            prediccionestest = prediccionestest_n.copy()
            prediccionestest = scal['salidas'].inverse_transform(prediccionestest_n)

            #prediccionestest = modelo.predict(Xtrain, batch_size=1)
            
            print("fin prediccion")
            """import pandas as pd
            prediccionesval = prediccionestest
                        # Crear listas para almacenar los resultados
            valores_reales = []
            predicciones = []
            errores = []
            resultados = []
            errores_totales = []
            r2_por_columna = []
            mae_por_columna = []
            desviacion_estandar_por_columna = []
            import numpy as np

            # Supongamos que df tiene los datos originales con las columnas codificadas
            for valor in range(len(ytest)):
                y_real = ytest[valor]
                prediccion = prediccionesval[valor]
                
                # Calcular errores
                errores = [prediccion[i] - y_real[i] for i in range(len(y_real))]
                errores_totales.extend(errores)  # Guardamos todos los errores para análisis global

                errores_acumulativos = []
                # Error máximo
            

                # Calcular MAPE por columna
                from sklearn.metrics import mean_absolute_percentage_error
                mape_por_columna = [mean_absolute_percentage_error(ytest[:, i], prediccionesval[:, i]) for i in range(ytest.shape[1])]

                # Calcular R² por columna
                r2_por_columna = [r2_score(ytest[:, i], prediccionesval[:, i]) for i in range(ytest.shape[1])]

                # Calcular MAE por columna
                mae_por_columna = [mean_absolute_error(ytest[:, i], prediccionesval[:, i]) for i in range(ytest.shape[1])]
                mae_por_columna = [f"{mae:.2f}" for mae in mae_por_columna]
                desviacion_estandar_por_columna = [np.std( prediccionesval[:, i] - ytest[:, i]) for i in range(ytest.shape[1])]
                desviacion_estandar_por_columna = [f"{desviacion:.2f}" for desviacion in desviacion_estandar_por_columna]


                # Calcular error promedio y desviación estándar para este valor
                error_promedio = np.mean(np.abs(errores))
                desviacion_estandar = np.std(errores)

                # Contar cuántas diferencias son mayores a 3 kilowatts
                diferencias_mayores_a_3 = sum(abs(error) > 3 for error in errores)
                
                # Obtener los valores de codificación para cada predicción
                for i in range(len(y_real)):
                    errores_acumulativos.append(np.abs(errores[i]))
                    
                    # Calcular el error promedio acumulativo hasta el índice actual
                    error_promedio_acumulativo = np.mean(errores_acumulativos)
                    desviacion_estandar_acumulativa = np.std(errores_acumulativos)
                    
                    # Calcular el error relativo porcentual
                    if y_real[i] != 0:  # Evitar división por cero
                        error_relativo_porcentual = (abs(errores[i]) / abs(y_real[i])) * 100
                    else:
                        error_relativo_porcentual = 0  # Si y_real es cero, podemos definir el error relativo como 0

                    # Almacenar resultados
                    resultados.append({
                        'valor': valor,
                        'prediccion': prediccion[i],
                        'valor_real': y_real[i],
                        'error': errores[i],
                        'error_promedio': error_promedio,
                        'desviacion_estandar': desviacion_estandar,
                        'diferencias_mayores_a_3': diferencias_mayores_a_3,
                        'error_promedio_acumulativo': error_promedio_acumulativo,
                        'desviacion_estandar_acumulativa': desviacion_estandar_acumulativa,
                        'error_relativo_porcentual': error_relativo_porcentual  # Nueva columna
                    })

            # Convertir a DataFrame
            df_resultados = pd.DataFrame(resultados)

            # Calcular error relativo porcentual promedio global
            error_relativo_porcentual_promedio = np.mean(df_resultados['error_relativo_porcentual'])
            error_maximo = max(abs(error) for error in errores_totales)

            # Calcular error promedio global y desviación estándar promedio
            error_promedio_global = np.mean(df_resultados['error'])
            desviacion_estandar_global = np.std(df_resultados['error'])

            errores_menores_a_1 = sum(abs(error) <= 1 for error in errores_totales)
            errores_menores_a_2 = sum(abs(error) <= 2 for error in errores_totales)
            errores_menores_a_3 = sum(abs(error) <= 3 for error in errores_totales)
            errores_menores_a_4 = sum(abs(error) <= 4 for error in errores_totales)
            errores_menores_a_5 = sum(abs(error) <= 5 for error in errores_totales)
            
            total_datos = len(errores_totales)

            # Calcular el porcentaje de errores menores a 1, 2, 3, 4, 5
            porcentaje_menores_a_1 = (errores_menores_a_1 / total_datos) * 100
            porcentaje_menores_a_2 = (errores_menores_a_2 / total_datos) * 100
            porcentaje_menores_a_3 = (errores_menores_a_3 / total_datos) * 100
            porcentaje_menores_a_4 = (errores_menores_a_4 / total_datos) * 100
            porcentaje_menores_a_5 = (errores_menores_a_5 / total_datos) * 100

            # Guardar a un archivo CSV
            #df_resultados.to_csv('resultados_predicciones_con_datos.csv', index=False)

            # Imprimir resultados globales
            print(f"Error promedio global: {error_promedio_global:.2f}")
            print(f"Desviación estándar global: {desviacion_estandar_global:.2f}")
            print(f"Error relativo porcentual promedio global: {error_relativo_porcentual_promedio:.2f}%")

            print(f"Error maximo: {error_maximo}")
            print(f"cantidad de datos: {total_datos}")
            #print(f"mae por columan: {mae_por_columna:.2f}%")
            print(f"p < 1 : {porcentaje_menores_a_1:.2f}")
            print(f"p < 2 : {porcentaje_menores_a_2:.2f}")
            print(f"p < 3 : {porcentaje_menores_a_3:.2f}")
            print(f"p < 4 : {porcentaje_menores_a_4:.2f}")
            print(f"p < 5 : {porcentaje_menores_a_5:.2f}")

            print(f"MAPE por columna: {mape_por_columna}\n")
            print(f"R² por columna: {r2_por_columna}\n")
            print(f"MAE por columna: {mae_por_columna}\n")
            print(f"Desviación estándar por columna: {desviacion_estandar_por_columna}\n")
        
"""
        import matplotlib.pyplot as plt

   # Definir índices de rango
        start = 0
        end = fin_test - inicio_test 
        x = np.arange(start, end)

        # Calcular errores
        error = ytest - prediccionestest

        # Calcular errores con predicción desplazada una unidad hacia atrás
        pred_shift = prediccionestest[:-1]
        ytest_shift = ytest[1:]  # para que coincida en tamaño
        error_shift = ytest_shift - pred_shift
        x_shift = np.arange(len(error_shift))

        # Crear figura con subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 12))

        # Subplot 1: Error normal
        axs[0].plot(x, error[:, 0], label='Error', color='blue')
        axs[0].set_title('Error entre predicciones y valores reales')
        axs[0].set_xlabel('Índice')
        axs[0].set_ylabel('Error')
        axs[0].legend()
        axs[0].grid(True)

        # Subplot 2: Error con predicción desplazada
        axs[1].plot(x_shift, error_shift[:, 0], label='Error (predicción desplazada)', color='orange')
        axs[1].set_title('Error con predicción desplazada una unidad hacia atrás')
        axs[1].set_xlabel('Índice')
        axs[1].set_ylabel('Error desplazado')
        axs[1].legend()
        axs[1].grid(True)

        # Subplot 3: Comparación entre predicción normal y desplazada
        axs[2].plot(x, prediccionestest[:, 0], label='Predicción original', color='red')
        axs[2].plot(x_shift, pred_shift[:, 0], label='Predicción desplazada', color='green', linestyle='--')
        axs[2].set_title('Comparación entre predicción original y desplazada')
        axs[2].set_xlabel('Índice')
        axs[2].set_ylabel('Valor predicho')
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()



persistentes = 0
total_predicciones = len(ytest)

for i in range(1, total_predicciones):
    valor_anterior = ytest[i - 1][-1]  # último timestep del anterior real
    prediccion = prediccionestest[i][-1]  # último timestep del actual predicho

    if valor_anterior == 0:
        continue  # evitamos división por cero

    margen_inferior = valor_anterior * 0.99
    margen_superior = valor_anterior * 1.01

    if margen_inferior <= prediccion <= margen_superior:
        persistentes += 1

porcentaje_persistencia = (persistentes / (total_predicciones - 1)) * 100  # -1 porque empezamos en i=1

print(f"Total de predicciones: {total_predicciones}")
print(f"Cantidad de predicciones persistentes: {persistentes}")
print(f"Porcentaje de persistencia: {porcentaje_persistencia:.2f}%")



fig, axs = plt.subplots(3, 1, figsize=(12, 12))
start = 0
end = fin_test - inicio_test 
x = np.arange(start, end)

        # Calcular errores
error = ytest - prediccionestest

        # Calcular errores con predicción desplazada una unidad hacia atrás
pred_shift = prediccionestest[:-1]
ytest_shift = ytest[1:]  # para que coincida en tamaño
error_shift = ytest_shift - pred_shift
x_shift = np.arange(len(error_shift))

        # Crear figura con subplots
x= np.arange(len(ytest))
axs[0].plot(x, ytest[:, 0], label='Real', color='blue')
axs[0].plot(x, prediccionestest[:, 0], label='Predicción', color='orange')
axs[0].set_title('Predicciones vs Valores reales')
axs[0].set_xlabel('Índice')
axs[0].set_ylabel('Valor')
axs[0].legend()
axs[0].grid(True)

# Segundo subplot: ytest desplazado 1 hacia atrás vs predicción desplazada 1 hacia adelante
y_real_shift = ytest[:-1, 0]               # valores reales sin el último
y_pred_shift = prediccionestest[1:, 0]     # predicciones desde el índice 1
x_shift = np.arange(len(y_real_shift))

axs[1].plot(x_shift, y_real_shift, label='Real (t)', color='blue')
axs[1].plot(x_shift, y_pred_shift, label='Predicción (t+1)', color='orange')
axs[1].set_title('Predicción en t+1 vs valor real en t')
axs[1].set_xlabel('Índice')
axs[1].set_ylabel('Valor')
axs[1].legend()
axs[1].grid(True)

# Tercer subplot opcional: error absoluto desplazado
error_shift = y_real_shift - y_pred_shift
axs[2].plot(x_shift, error_shift, label='Error (t+1 - t)', color='green')
axs[2].set_title('Error desplazado entre predicción y valor real anterior')
axs[2].set_xlabel('Índice')
axs[2].set_ylabel('Error')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()



persistentes = 0
total_predicciones = len(ytest)

for i in range(1, total_predicciones):
    valor_anterior = ytest[i - 1,0]
    prediccion = prediccionestest[i,0]

    if valor_anterior == 0:
        continue  # evitamos división por cero

    margen_inferior = valor_anterior * 0.995
    margen_superior = valor_anterior * 1.005

    if margen_inferior <= prediccion <= margen_superior:
        persistentes += 1

porcentaje_persistencia = (persistentes / (total_predicciones - 1)) * 100  # -1 porque empezamos desde i=1

print(f"Total de predicciones: {total_predicciones}")
print(f"Cantidad de predicciones persistentes: {persistentes}")
print(f"Porcentaje de persistencia al +-0.5%: {porcentaje_persistencia:.2f}%")



persistentes = 0
total_predicciones = len(ytest)

for i in range(1, total_predicciones):
    valor_anterior = ytest[i - 1,0]
    prediccion = prediccionestest[i,0]

    if valor_anterior == 0:
        continue  # evitamos división por cero

    margen_inferior = valor_anterior * 0.99
    margen_superior = valor_anterior * 1.01

    if margen_inferior <= prediccion <= margen_superior:
        persistentes += 1

porcentaje_persistencia = (persistentes / (total_predicciones - 1)) * 100  # -1 porque empezamos desde i=1

print(f"Total de predicciones: {total_predicciones}")
print(f"Cantidad de predicciones persistentes: {persistentes}")
print(f"Porcentaje de persistencia al +-1%: {porcentaje_persistencia:.2f}%")



persistentes = 0
total_predicciones = len(ytest)

for i in range(1, total_predicciones):
    valor_anterior = ytest[i - 1,0]
    prediccion = prediccionestest[i,0]

    if valor_anterior == 0:
        continue  # evitamos división por cero

    margen_inferior = valor_anterior * 0.98
    margen_superior = valor_anterior * 1.02

    if margen_inferior <= prediccion <= margen_superior:
        persistentes += 1

porcentaje_persistencia = (persistentes / (total_predicciones - 1)) * 100  # -1 porque empezamos desde i=1

print(f"Total de predicciones: {total_predicciones}")
print(f"Cantidad de predicciones persistentes: {persistentes}")
print(f"Porcentaje de persistencia al +-2%: {porcentaje_persistencia:.2f}%")


print("aAAAAAAAAAAAAAAAAA")



persistentes = 0
total_predicciones = len(ytest)
total_filtrados = 0  # para contar solo los casos con diferencia ≤ 4 kW

for i in range(1, total_predicciones):
    valor_anterior = ytest[i - 1, 0]
    valor_real = ytest[i, 0]
    prediccion = prediccionestest[i, 0]

    if valor_anterior == 0:
        continue  # evitamos división por cero

    diferencia_real = abs(valor_real - valor_anterior)

    # Solo consideramos casos con diferencia ≤ 4 kW entre valor real y valor anterior
    if diferencia_real >= 4:
        total_filtrados += 1

        margen_inferior = valor_anterior * 0.99
        margen_superior = valor_anterior * 1.01

        if margen_inferior <= prediccion <= margen_superior:
            persistentes += 1

if total_filtrados > 0:
    porcentaje_persistencia = (persistentes / total_filtrados) * 100
else:
    porcentaje_persistencia = 0

print(f"Total de predicciones: {total_predicciones}")
print(f"Total de casos con diferencia real ≤ 4 kW: {total_filtrados}")
print(f"Cantidad de predicciones persistentes en esos casos: {persistentes}")
print(f"Porcentaje de persistencia al ±1% en esos casos: {porcentaje_persistencia:.2f}%")

mejor_que_persistente = 0
total_comparaciones = 0

# solo iteramos filas, desde la 1 porque necesitamos valor anterior para comparar
for i in range(1, ytest.shape[0]):
    valor_real = ytest[i, -1]         # último paso real
    valor_anterior = ytest[i - 1, -1] # último paso persistente (valor anterior mismo paso)
    pred_modelo = prediccionestest[i, -1]  # último paso predicho

    error_modelo = abs(valor_real - pred_modelo)
    error_persistente = abs(valor_anterior - pred_modelo)

    if error_modelo < error_persistente:
        mejor_que_persistente += 1

    total_comparaciones += 1

porcentaje_mejora = (mejor_que_persistente / total_comparaciones) * 100

print(f"Total de comparaciones (última columna): {total_comparaciones}")
print(f"Cantidad mejores que persistente: {mejor_que_persistente}")
print(f"Porcentaje de mejora sobre persistencia: {porcentaje_mejora:.2f}%")



mejor_que_persistente = 0
total_comparaciones = 0

for i in range(1, ytest.shape[0]):
    valor_real = ytest[i, -1]            # último paso real
    valor_anterior = ytest[i - 1, -1]   # último paso persistente (valor anterior mismo paso)
    pred_modelo = prediccionestest[i, -1]  # último paso predicho

    diferencia_real = abs(valor_real - valor_anterior)

    # Solo comparar si diferencia real es >= 4 kW
    if diferencia_real >= 9:
        error_modelo = abs(valor_real - pred_modelo)
        error_persistente = abs(pred_modelo - valor_anterior)  # acá corregí para comparar con valor real, no con predicción

        if error_modelo < error_persistente:
            mejor_que_persistente += 1

        total_comparaciones += 1

if total_comparaciones > 0:
    porcentaje_mejora = (mejor_que_persistente / total_comparaciones) * 100
else:
    porcentaje_mejora = 0

print(f"Total de comparaciones (última columna, dif ≥ 5kW): {total_comparaciones}")
print(f"Cantidad mejores que persistente: {mejor_que_persistente}")
print(f"Porcentaje de mejora sobre persistencia: {porcentaje_mejora:.2f}%")