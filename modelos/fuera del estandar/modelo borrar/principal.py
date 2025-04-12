import os
import logging
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from tools.logger_config import setup_logger
from tools.red_principal import *  # para usar y guardar
from sklearn.preprocessing import MinMaxScaler,StandardScaler

import matplotlib.pyplot as plt


if __name__ == "__main__":
    setup_logger()

    # nombre del modelo, con esto se crea la carpeta y archivos salida
    ## salida.entrada.bz.variacion
    #f fast 30 epocs
    #o buscando overffiting
    nombre_modelo = "modelo borrar" 
    #modificado cargar_datos_especificos, crear_ventana, M solo mediciones
    carpeta, resultados_path = crear_carpeta_y_guardar(nombre_modelo)

    # Procesamiento de los datos
    df = cargar_datos()
    #dias = [0, 1, 2, 3, 4, 5, 6]  # 0 domingo
    dias = [1]  # 0 domingo
    #horas = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]  # Ejemplo de horas
    horas = [8,9,10,11,12,13,14,15,16,17,18,19,20]  # Ejemplo de horas
    
    #df = cargar_datos_especificos('potencias.csv', dias_semanales=dias, horas=horas)
    df = cargar_datos()
    #df['activa'] = df['activa'].interpolate(method='spline', order=3)

    """ df = df.reset_index(drop=True)

    
    
    df.loc[:, 'media_movil'] = df.iloc[:, 0].rolling(window=4, min_periods=1).mean()
    # Crear una nueva columna basada en la media móvil
    df.loc[:, "media_ajustada"] = df["media_movil"]
    df["modificado"] = 0

    # Ajustar la media móvil en función de la diferencia relativa con el valor anterior
    for i in range(1, len(df)):  # Evitar i=-1 en la primera iteración
        diferencia = df.iloc[i, 0] - df.iloc[i - 1, 0]  # Diferencia con el valor anterior
        diferencia_relativa = (diferencia / df.iloc[i - 1, 0]) * 100  # Diferencia relativa en porcentaje
        
        diferenciam = df.loc[i,"media_movil"] - df.loc[i-1,"media_movil"]

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
   
    
    print("tengo estos datos ",df.shape)
    #df.loc[:, "activa"] = df.iloc[:, 0].rolling(window=4, min_periods=1).mean()

    X, y = crear_ventana(df[0:200000], 4, 1)

    print(X)
    ####### SEPARACION DE DATOS
    inicio_train = 0
    #fin_train = 5000
    fin_train = 55000

    inicio_val = fin_train+1
    fin_val = fin_train+1+15000
    #fin_val = fin_train+1+1000

    inicio_test = fin_val+1
    fin_test = inicio_test+1+12150
    #fin_test = inicio_test+1+1000

    # conjunto de validación
    Xval = X[inicio_val:fin_val]
    yval = y[inicio_val:fin_val]
    #conjunto de entrenamiento
    Xtrain = X[inicio_train:fin_train]
    ytrain = y[inicio_train:fin_train]
    # conjunto de validación
    Xtest = X[inicio_test:fin_test]
    ytest = y[inicio_test:fin_test]

    print(Xtrain[0])
    print(ytrain[0])

    print(Xtrain[1])
    print(ytrain[1])
    print("Shape de Xtra:", Xtrain.shape)
    print("Shape de Xtest:", Xtest.shape)
    print("Shape de Xtva:", Xval.shape)


    #SCALERS
    #scaleractiva = MinMaxScaler(feature_range=(0, 1))
    scaleractiva = StandardScaler()
    
    Xtrain_n = Xtrain.copy()
    Xtrain_n[:, :, 0] = scaleractiva.fit_transform(Xtrain[:, :, 0])
    Xval_n = Xval.copy()
    Xval_n[:, :, 0] = scaleractiva.transform(Xval[:, :, 0])
    Xtest_n = Xtest.copy()
    Xtest_n[:, :, 0] = scaleractiva.transform(Xtest[:, :, 0])

    #scalerl1 = MinMaxScaler(feature_range=(0, 1))
    scalerl1 = StandardScaler()
    Xtrain_n[:, :, 1] = scalerl1.fit_transform(Xtrain[:, :, 1])
    Xval_n[:, :, 1] = scalerl1.transform(Xval[:, :, 1])
    Xtest_n[:, :, 1] = scalerl1.transform(Xtest[:, :, 1])

    #scalerl2 = MinMaxScaler(feature_range=(0, 1))
    scalerl2 = StandardScaler()
    Xtrain_n[:, :, 2] = scalerl2.fit_transform(Xtrain[:, :, 2])
    Xval_n[:, :, 2] = scalerl2.transform(Xval[:, :, 2])
    Xtest_n[:, :, 2] = scalerl2.transform(Xtest[:, :, 2])

    #scalerl3 = MinMaxScaler(feature_range=(0, 1))
    scalerl3 = StandardScaler()
    Xtrain_n[:, :, 3] = scalerl3.fit_transform(Xtrain[:, :, 3])
    Xval_n[:, :, 3] = scalerl3.transform(Xval[:, :, 3])
    Xtest_n[:, :, 3] = scalerl3.transform(Xtest[:, :, 3])

    salidas = StandardScaler()
    #salidas = MinMaxScaler(feature_range=(0, 1))
    ytrain_n = ytrain.copy()
    ytrain_n = salidas.fit_transform(ytrain)
    yval_n = yval.copy()
    yval_n = salidas.transform(yval)
    ytest_n = ytest.copy()
    ytest_n = salidas.transform(ytest)

    scalers = {'scaleractiva': scaleractiva, 'salidas': salidas, 'l1': scalerl1, 'l2':scalerl2, 'l3':scalerl3}


    # Entrenar el modelo
    modell = entrenar_modelo(Xtrain_n, ytrain_n, Xval_n, yval_n)
    
    prediccionesTest = modell.predict(Xtest_n)
    prediccionesTest = salidas.inverse_transform(prediccionesTest)
    print("el shape es",prediccionesTest.shape)

   # Calcular los resultados
    calcular_resultados(ytest, prediccionesTest,carpeta)

    # Guardar el modelo, los escaladores y los resultados
    guardar_modelo_y_resultados(carpeta, modell, scalers)

    print(f"Modelo, escaladores y resultados guardados en {carpeta}")

    prediccionesTest = prediccionesTest.ravel()
    ytest = ytest.ravel()
    print(prediccionesTest.shape)
    print( ytest.shape)
    
    

    start = fin_val+1  # Índice inicial
    end = fin_test  # Índice final
    # Crear una figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Graficar yval y prediccionestest en el primer subplot
    ax1.plot(range(start, end), prediccionesTest, label='pred', color='red')
    ax1.plot(range(start, end), ytest, label='valores reales', color='green')

    ax1.set_title(f"Modelo ideal train (Índices {start} a {end})")
    ax1.set_xlabel('Índice')
    ax1.set_ylabel('Valor')
    ax1.legend()
    ax1.grid(True)
    error = prediccionesTest - ytest
    ax2.plot(range(start, end), error, label='Error reales', color='blue')

    ax2.set_title(f"Modelo (Índices {start} a {end})")
    ax2.set_xlabel('Índice')
    ax2.set_ylabel('Valor')
    ax2.legend()
    ax2.grid(True)


    # Ajustar el espacio entre subplots
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()