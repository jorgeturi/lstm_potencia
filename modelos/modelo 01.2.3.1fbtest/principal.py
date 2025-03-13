import os
import logging
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from tools.logger_config import setup_logger
from tools.red_principal import *  # para usar y guardar
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
# Parámetros
n_muestras = 30000  # Número de muestras
n_pasos = 48       # Número de pasos temporales por muestra

# Generar características cíclicas
def generar_caracteristicas_ciclicas(n_muestras, n_pasos):
    # Momento del día (horas)
    horas = np.linspace(0, 24, n_pasos, endpoint=False)
    seno_horas = np.sin(2 * np.pi * horas / 24)
    coseno_horas = np.cos(2 * np.pi * horas / 24)

    # Día del año (días)
    dias = np.linspace(0, 365, n_pasos, endpoint=False)
    seno_dias = np.sin(2 * np.pi * dias / 365)
    coseno_dias = np.cos(2 * np.pi * dias / 365)

    # Repetir para todas las muestras
    seno_horas = np.tile(seno_horas, (n_muestras, 1))
    coseno_horas = np.tile(coseno_horas, (n_muestras, 1))
    seno_dias = np.tile(seno_dias, (n_muestras, 1))
    coseno_dias = np.tile(coseno_dias, (n_muestras, 1))

    return seno_horas, coseno_horas, seno_dias, coseno_dias

# Generar medición (target)
def generar_medicion(seno_horas, coseno_horas, seno_dias, coseno_dias):
    # La medición depende de las características cíclicas
    medicion = 10 * seno_horas + 5 * coseno_horas + 20 * seno_dias + 10 * coseno_dias
    # Añadir ruido para mayor realismo
    ruido = np.random.normal(0, 1, medicion.shape)
    medicion += ruido
    return medicion

if __name__ == "__main__":
    setup_logger()

    # nombre del modelo, con esto se crea la carpeta y archivos salida
    ## salida.entrada.bz.variacion
    #f fast 30 epocs
    nombre_modelo = "modelo 01.2.3.1fbtest" 
    #modificado cargar_datos_especificos, crear_ventana, M solo mediciones
    carpeta, resultados_path = crear_carpeta_y_guardar(nombre_modelo)

    # Procesamiento de los datos
    df = cargar_datos()
    dias = [1, 2, 3, 4, 5]  # 0 domingo
    horas = [8,9,10,11,12,13,14,15,16,17,18,19,20]  # Ejemplo de horas
    df = cargar_datos_especificos('potencias.csv', dias_semanales=dias, horas=horas)
    print("tengo estos datos ",df.shape)
    #X, y = crear_ventana(df[000:200000], 48, 1)
    seno_horas, coseno_horas, seno_dias, coseno_dias = generar_caracteristicas_ciclicas(n_muestras, n_pasos)
    medicion = generar_medicion(seno_horas, coseno_horas, seno_dias, coseno_dias)

    # Crear X e y
    X = np.stack([medicion, seno_horas, coseno_horas, seno_dias, coseno_dias], axis=-1)  # (n_muestras, n_pasos, 5)
    y = np.array([medicion[i, -4:] for i in range(n_muestras)])  # (n_muestras, 4)


    # Verificar formas
    print("Forma de X:", X.shape)  # Debería ser (n_muestras, n_pasos, 5)
    print("Forma de y:", y.shape)  # Debería ser (n_muestras, 4)

    # Visualizar una muestra
    plt.figure(figsize=(12, 6))
    plt.plot(X[0, :, 0], label='Medición')
    plt.plot(X[0, :, 1], label='Seno horas')
    plt.plot(X[0, :, 2], label='Coseno horas')
    plt.plot(X[0, :, 3], label='Seno días')
    plt.plot(X[0, :, 4], label='Coseno días')
    plt.legend()
    plt.title("Datos Sintéticos - Una Muestra")
    plt.show()
            
    ####### SEPARACION DE DATOS
    inicio_train = 0
    fin_train = 20000
    inicio_val = fin_train+1
    fin_val = fin_train+1+4000
    inicio_test = fin_val+1
    fin_test = inicio_test+1+4000
    # conjunto de validación
    Xval = X[inicio_val:fin_val]
    yval = y[inicio_val:fin_val]
    #conjunto de entrenamiento
    Xtrain = X[inicio_train:fin_train]
    ytrain = y[inicio_train:fin_train]
    # conjunto de validación
    Xtest = X[inicio_test:fin_test]
    ytest = y[inicio_test:fin_test]




    #SCALERS
    scaleractiva = MinMaxScaler(feature_range=(0, 1))
    Xtrain_n = Xtrain.copy()
    Xtrain_n[:, :, 0] = scaleractiva.fit_transform(Xtrain[:, :, 0])
    Xval_n = Xval.copy()
    Xval_n[:, :, 0] = scaleractiva.transform(Xval[:, :, 0])
    Xtest_n = Xtest.copy()
    Xtest_n[:, :, 0] = scaleractiva.transform(Xtest[:, :, 0])

    salidas = MinMaxScaler(feature_range=(0, 1))
    ytrain_n = ytrain.copy()
    ytrain_n = salidas.fit_transform(ytrain)
    yval_n = yval.copy()
    yval_n = salidas.transform(yval)
    ytest_n = ytest.copy()
    ytest_n = salidas.transform(ytest)

    scalers = {'scaleractiva': scaleractiva, 'salidas': salidas}


    # Entrenar el modelo
    modell = entrenar_modelo_gru(Xtrain_n, ytrain_n, Xval_n, yval_n)
    
    prediccionesTest = modell.predict(Xtest_n)
    prediccionesTest = salidas.inverse_transform(prediccionesTest)

   # Calcular los resultados
    calcular_resultados(ytest, prediccionesTest,carpeta)

    # Guardar el modelo, los escaladores y los resultados
    guardar_modelo_y_resultados(carpeta, modell, scalers)

    print(f"Modelo, escaladores y resultados guardados en {carpeta}")
