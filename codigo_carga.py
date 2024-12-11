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
    path = "modelos/modelo 0.0.5.3/"
    ventana_entrada = 4
    ventana_prediccion = 4


    modelo = cargar_modelo(path + "modelo")

    if modelo is not None:  #si consegui el modelo
        modelo.summary()  
        scal = cargar_escaladores(path + "scalers.pkl")
        if scal is not None:
            print("scalers:", scal)
            dias = [0,1,2,3,4,5,6]  # Lunes, Miércoles, Viernes
            horas = [0,1,2,3,4,5,6,7,8,9, 10, 11, 12,13,14,15,16,17,18,19,20,21,22,23]  # De 9 a 12 horas
            df = cargar_datos_especificos('potencias.csv', dias_semanales=dias, horas=horas)

            X, y = crear_ventana(df[000:200000], ventana_entrada, ventana_prediccion)
            
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

            print("mmin ", scal['scaleractiva'].data_min_)
            print("max ", scal['scaleractiva'].data_max_)


            Xtest_n = Xtest.copy()
            Xtest_n[:, :, 0] = scal['scaleractiva'].transform(Xtest[:, :, 0])

            logging.info("inicio prediccion")

            prediccionestest_n = modelo.predict(Xtest_n)
            prediccionestest = prediccionestest_n.copy()
            prediccionestest = scal['salidas'].inverse_transform(prediccionestest_n)

            #prediccionestest = modelo.predict(Xtrain, batch_size=1)

            print("fin prediccion")
            import pandas as pd
            prediccionesval = prediccionestest
                        # Crear listas para almacenar los resultados
            valores_reales = []
            predicciones = []
            errores = []
            resultados = []
            errores_totales = []

            import numpy as np

            # Supongamos que df tiene los datos originales con las columnas codificadas
            for valor in range(len(ytest)):
                y_real = ytest[valor]
                prediccion = prediccionesval[valor]
                
                # Calcular errores
                errores = [prediccion[i] - y_real[i] for i in range(len(y_real))]
                errores_acumulativos = []
                
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

            # Calcular error promedio global y desviación estándar promedio
            error_promedio_global = np.mean(df_resultados['error'])
            desviacion_estandar_global = np.std(df_resultados['error'])

            # Guardar a un archivo CSV
            df_resultados.to_csv('resultados_predicciones_con_datos.csv', index=False)

            # Imprimir resultados globales
            print(f"Error promedio global: {error_promedio_global:.2f}")
            print(f"Desviación estándar global: {desviacion_estandar_global:.2f}")
            print(f"Error relativo porcentual promedio global: {error_relativo_porcentual_promedio:.2f}%")
