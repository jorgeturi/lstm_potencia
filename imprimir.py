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
    #carpeta, resultados_path = crear_carpeta_y_guardar(nombre_modelo)

    # Procesamiento de los datos
    df = cargar_datos()
    #dias = [0, 1, 2, 3, 4, 5, 6]  # 0 domingo
    dias = [1]  # 0 domingo
    #horas = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]  # Ejemplo de horas

    horas = [8,9,10,11,12,13,14,15,16,17,18,19,20]  # Ejemplo de horas
    df = cargar_datos_especificos('potencias.csv', dias_semanales=dias, horas=horas)
    df = df.reset_index(drop=True)

    #df.loc[:, 0] = df.iloc[:, 0].rolling(window=4, min_periods=1).mean()
    #df.loc[:, "activa"] = df.iloc[:, 0].rolling(window=4, min_periods=1).mean()
    print (df)
    df2 = df.copy()
    df3 = df.copy()

    df2.loc[:, 'activa'] = df2.iloc[:, 0].rolling(window=4, min_periods=1).mean()
            
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
        if df.iloc[i, 0] < 35:
            
            if diferencia_relativa > 13:
                df.iloc[i, df.columns.get_loc("media_ajustada")] += 0.30*df.iloc[i, 0]
                df.loc[i, "modificado"] = 1  # Marcar que fue modificado
                
            elif diferencia_relativa < -13:
                df.iloc[i, df.columns.get_loc("media_ajustada")] -= 0.30*df.iloc[i, 0]
                df.loc[i, "modificado"] = 2  # Marcar que fue modificado

            if diferencia_relativa > 33:
                df.iloc[i, df.columns.get_loc("media_ajustada")] += 0.20*df.iloc[i, 0]
                df.loc[i, "modificado"] = 1  # Marcar que fue modificado
            
            elif diferencia_relativa < -33:
                df.iloc[i, df.columns.get_loc("media_ajustada")] -= 0.20*df.iloc[i, 0]
                df.loc[i, "modificado"] = 2  # Marcar que fue modificado
            

            if df.loc[i-1, "modificado"] == 1:  # Si el anterior fue modificado
                df.iloc[i, df.columns.get_loc("media_ajustada")] += 0.11*df.iloc[i, 0]
            if df.loc[i-1, "modificado"] == 2:  # Si el anterior fue modificado
                df.iloc[i, df.columns.get_loc("media_ajustada")] -= 0.11*df.iloc[i, 0]
        elif df.iloc[i, 0] > 35:
            if diferencia_relativa > 5:
                df.iloc[i, df.columns.get_loc("media_ajustada")] += 0.10*df.iloc[i, 0]
                df.loc[i, "modificado"] = 1  # Marcar que fue modificado
                    
            elif diferencia_relativa < -5:
                df.iloc[i, df.columns.get_loc("media_ajustada")] -= 0.10*df.iloc[i, 0]
                df.loc[i, "modificado"] = 2  # Marcar que fue modificado
                

            if df.loc[i-1, "modificado"] == 1:  # Si el anterior fue modificado
                df.iloc[i, df.columns.get_loc("media_ajustada")] += 0.05*df.iloc[i, 0]
            if df.loc[i-1, "modificado"] == 2:  # Si el anterior fue modificado
                df.iloc[i, df.columns.get_loc("media_ajustada")] -= 0.05*df.iloc[i, 0]

        

    # Asignar la media ajustada a "activa"
    df["activa"] = df["media_ajustada"]
        




   # Suavizado con filtro de mediana

    #window_size = 2  # Tamaño de la ventana para el filtro de mediana
    #for i in range(1, len(df)-1):
    #    window = df['activa'][i-1:i+2]
    #    df.iloc[i, df.columns.get_loc("activa")] = window.median()




    #df.iloc[:, 0] = df.iloc[:, 0].rolling(window=4, min_periods=1).mean()
    X1, y1 = crear_ventana(df2[0:200000], 8, 4)
    X, y = crear_ventana(df[0:200000], 8, 4)
    X2, y2 = crear_ventana(df3[0:200000], 8, 4)

            
   ####### SEPARACION DE DATOS
    ####### SEPARACION DE DATOS
    inicio_train = 0
    fin_train = 4000
    #fin_train = 71500

    inicio_val = fin_train+1
    #fin_val = fin_train+1+18100
    fin_val = fin_train+1+1000

    inicio_test = fin_val+1
    #fin_test = inicio_test+1+11615
    fin_test = inicio_test+1+1000
    # conjunto de validación
    Xval = X[inicio_val:fin_val]
    yval = y[inicio_val:fin_val]
    #conjunto de entrenamiento
    Xtrain = X[inicio_train:fin_train]
    ytrain = y[inicio_train:fin_train]
    # conjunto de validación
    Xtest = X[inicio_test:fin_test]
    
    ytest = y[inicio_test:fin_test]
    ytestreales = y2[inicio_test:fin_test]
    ytestsuavizados = y1[inicio_test:fin_test]
    print("ytest " , ytest)
    

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

    error = ytestreales[:,0] - ytest[:,0]
    datos = error
    media = np.mean(datos)
    desviacion = np.std(datos)
    minimo = np.min(datos)
    maximo = np.max(datos)
    mediana = np.median(datos)
    varianza = np.var(datos)

    print(f"Media: {media}")
    print(f"Desviación estándar: {desviacion}")
    print(f"Mínimo: {minimo}")
    print(f"Máximo: {maximo}")
    print(f"Mediana: {mediana}")
    print(f"Varianza: {varianza}")

    print("el shapeo es", ytrain.shape)
    # Entrenar el modelo
    import matplotlib.pyplot as plt

    # Definir el rango que deseas graficar
    start = 0  # Índice inicial
    end = 1000  # Índice final

    # Crear una figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Graficar yval y prediccionestest en el primer subplot
    ax1.plot(range(start, end), ytestreales[start:end,0], label='valores reales', color='red')
    ax1.plot(range(start, end), ytest[start:end,0], label='media ajustada', color='green')
    ax1.plot(range(start, end), ytestsuavizados[start:end,0], label='suavizados', color='orange')

    ax1.set_title(f"Modelo ideal train (Índices {start} a {end})")
    ax1.set_xlabel('Índice')
    ax1.set_ylabel('Valor')
    ax1.legend()
    ax1.grid(True)

    errorsuavizados = ytestreales[:,0] - ytestsuavizados[:,0]
    ax2.plot(range(start, end), error[start:end], label='Error reales', color='blue')
    ax2.plot(range(start, end), errorsuavizados[start:end], label='Error suavizados', color='orange')

    ax2.set_title(f"Modelo (Índices {start} a {end})")
    ax2.set_xlabel('Índice')
    ax2.set_ylabel('Valor')
    ax2.legend()
    ax2.grid(True)


    # Ajustar el espacio entre subplots
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()