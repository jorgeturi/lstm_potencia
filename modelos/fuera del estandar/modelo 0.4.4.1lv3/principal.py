import os
import logging
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from tools.logger_config import setup_logger
from tools.red_principal import *  # para usar y guardar
from sklearn.preprocessing import MinMaxScaler



if __name__ == "__main__":
    setup_logger()

    # nombre del modelo, con esto se crea la carpeta y archivos salida
    ## salida.entrada.bz.variacion
    #f fast 30 epocs
    #o buscando overffiting
    nombre_modelo = "modelo 0.4.4.1lv3" 
    #modificado cargar_datos_especificos, crear_ventana, M solo mediciones
    carpeta, resultados_path = crear_carpeta_y_guardar(nombre_modelo)

    # Procesamiento de los datos
    df = cargar_datos()
    dias = [1]  # 0 domingo
    horas = [8,9,10,11,12,13,14,15,16,17,18,19,20]  # Ejemplo de horas
    df = cargar_datos_especificos('potencias.csv', dias_semanales=dias, horas=horas)
    print("tengo estos datos ",df.shape)
    df.loc[:, "activa"] = df.iloc[:, 0].rolling(window=16, min_periods=1).mean()

    X, y = crear_ventana(df[0:200000], 96, 16)

    print(X)
    ####### SEPARACION DE DATOS
    inicio_train = 0
    fin_train = 5000
    inicio_val = fin_train+1
    fin_val = fin_train+1+1000
    inicio_test = fin_val+1
    fin_test = inicio_test+1+1150
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
    modell = entrenar_modelo(Xtrain_n, ytrain_n, Xval_n, yval_n)
    
    prediccionesTest = modell.predict(Xtest_n)
    prediccionesTest = salidas.inverse_transform(prediccionesTest)

   # Calcular los resultados
    calcular_resultados(ytest, prediccionesTest,carpeta)

    # Guardar el modelo, los escaladores y los resultados
    guardar_modelo_y_resultados(carpeta, modell, scalers)

    print(f"Modelo, escaladores y resultados guardados en {carpeta}")
