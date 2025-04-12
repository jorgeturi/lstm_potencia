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
    
    path = "modelos/modelo 0.0.3.4/"
    ventana_entrada = 4
    
    ventana_prediccion = 14


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
            ####### SEPARACION DE DATOS
            inicio_train = 0
            fin_train = 71500
            inicio_val = fin_train+1
            fin_val = fin_train+1+18100
            inicio_test = fin_val+1
            fin_test = inicio_test+1+11615
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

    # Definir el rango que deseas graficar
    start = 0  # Índice inicial
    end = 11616  # Índice final
    

    # Calcular errores
    error = ytest - prediccionestest

    # Graficar el error
    plt.figure(figsize=(12, 5))
    x = np.arange(start, end)
    plt.plot(x, error, label='Errores', color='blue')
    plt.title(f"Error (Índices {start} a {end})")
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.show()

   
    # Graficar valores reales vs predicciones
    plt.figure(figsize=(12, 5))
    plt.plot(x, ytest[:,0], label='Original', color='blue')
    start = 0  # Índice inicial
    end = 11615
    x = np.arange(start, end)
    plt.plot(x, prediccionestest[1:,0], label='Predicciones', color='red')

    plt.title(f"Comparación Valores Reales y Predicciones (Índices {start} a {end})")
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.show()


