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
    path = "modelos/modelo 01.2.3.1fbgru/"
    ventana_entrada = 48
    ventana_prediccion = 1


    modelo = cargar_modelo(path + "modelo")

    if modelo is not None:  #si consegui el modelo
        modelo.summary()  
        scal = cargar_escaladores(path + "scalers.pkl")
        if scal is not None:
            #print("scalers:", scal)
            dias = [1,2,3,4,5]  # Lunes, Miércoles, Viernes
            horas = [8,9,10,11,12,13,14,15,16,17,18,19,20]  # Ejemplo de horas
            df = cargar_datos_especificos('potencias.csv', dias_semanales=dias, horas=horas)

            X, y = crear_ventana(df[000:200000], ventana_entrada, ventana_prediccion)
            print("tengo datos: ",len(X))
            ####### SEPARACION DE DATOS
            inicio_train = 0
            fin_train = 25000
            inicio_val = fin_train+1
            fin_val = fin_train+1+7000
            inicio_test = fin_val+1
            fin_test = inicio_test+1+7000
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

            logging.info("inicio prediccion")

            prediccionestest_n = modelo.predict(Xtest_n)
            prediccionestest = prediccionestest_n.copy()
            prediccionestest = scal['salidas'].inverse_transform(prediccionestest_n)

            #prediccionestest = modelo.predict(Xtrain, batch_size=1)
            ytest = ytest
            print("fin prediccion")
            import pandas as pd
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
        

        import matplotlib.pyplot as plt

    # Definir el rango que deseas graficar
    start = 50  # Índice inicial
    end = 100   # Índice final

    # Crear una figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Graficar yval y prediccionestest en el primer subplot
    ax1.plot(range(start, end), ytest[start:end], label='Valores Reales (yval)', color='blue')
    ax1.plot(range(start, end), prediccionestest[start:end], label='Predicciones', color='red')
    ax1.set_title(f"Comparación de Valores Reales y Predicciones (Índices {start} a {end})")
    ax1.set_xlabel('Índice')
    ax1.set_ylabel('Valor')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(start, end), ytest[start:end], label='Valores Reales (ytest) +1', color='blue')
    ax2.plot(range(start, end-1), prediccionestest[start+1:end], label='Predicciones', color='red')
    ax2.set_title(f"Comparación de Valores Reales y Predicciones (+1) (Índices {start} a {end})")
    ax2.set_xlabel('Índice')
    ax2.set_ylabel('Valor')
    ax2.legend()
    ax2.grid(True)

    # Ajustar el espacio entre subplots
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()