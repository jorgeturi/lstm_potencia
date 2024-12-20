import logging
import numpy as np
from tensorflow.keras.models import load_model  # Cargar modelo
import joblib #cargar escaladores
import pandas as pd # cargar datos 
import os
import shutil #para copiar archivo
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import sys #para terminar 



def funcion():
    logging.info("Este es un mensaje desde la función.")
    logging.debug("Este es un mensaje de depuración en la función.")
    logging.warning("Este es un mensaje de advertencia en la función.")
    logging.error("Este es un mensaje de error en la función.")

def cargar_modelo(ruta_al_modelo="default.txt"):
    """
    Carga un modelo desde un archivo y registra el resultado en el log.

    Args:
    ruta_al_modelo (str): La ruta al archivo del modelo.

    Returns:
    model: El modelo cargado si la carga fue exitosa, None sino.
    """
    try:
        # Intentar cargar el modelo
        model = load_model(ruta_al_modelo)
        logging.info(f'Modelo cargado exitosamente desde {ruta_al_modelo}.')
        return model
    except Exception as e:
        logging.error(f'Error al cargar el modelo desde {ruta_al_modelo}: {e}')
        return None

def cargar_escaladores(ruta_a_escaladores="rutafake.txt"):
    """
    Carga escaladores desde un archivo y registra el resultado en el log.

    Args:
    ruta_a_escaladores (str): La ruta a escaladores del modelo.

    Returns:
    escaladores: escaladores cargados si la carga fue exitosa, None sino.
    """
    try:
        # Intentar cargar el modelo
        scalers = joblib.load(ruta_a_escaladores)

        logging.info(f'Escaladores cargados exitosamente desde {ruta_a_escaladores}.')
        return scalers
    except Exception as e:
        logging.error(f'Error al cargar escaladores desde {ruta_a_escaladores}: {e}')
        return None




def crear_ventana(dataset, ventana_entrada, ventana_salida):
    logging.info("Creando ventanas.")

    # Extraer las características necesarias
    #features = dataset[['activa', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos', 'diferencia_activa', 'numero_de_medicion']].values
    features = dataset[['activa','dia_sen', 'dia_cos', 'mes_sen', 'mes_cos']].values

    #features = dataset[['activa', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos', 'diferencia_activa']].values

    #features = dataset[['activa']].values


    # Calcular el número total de ventanas que se pueden crear
    total_muestras = len(dataset) - ventana_entrada - ventana_salida + 1

    # Ventanas de entrada
    X = np.array([features[i:i + ventana_entrada] for i in range(total_muestras)])

    # Ventanas de salida
    y = np.array([dataset['activa'].values[i + ventana_entrada:i + ventana_entrada + ventana_salida] for i in range(total_muestras)])

    logging.info("Ventanas creadas")

    return X, y

import numpy as np
import logging

def crear_ventana_dataset(dataset, ventana):
    X = []
    logging.info("Creando ventanas.")

    # veo si es posible
    if len(dataset) < ventana:
        logging.warning("El tamaño del dataset es menor que la ventana. No se pueden crear ventanas.")
        return np.array(X)  # Returno vacio

    # Crear ventanas
    for i in range(len(dataset) - ventana + 1):  
        # Crear la ventana
        window = dataset.iloc[i:i + ventana].copy()

        # Características de la ventana
        window_features = window[['activa', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos']].values
        
        X.append(window_features)

    # Convertir las listas a arrays de NumPy para facilitar su uso
    X = np.array(X)
    logging.info("Ventanas creadas")

    return X




def codificar_tiempo(dt):
    dt2 = dt.copy()  

    # Ensure the 'timestamp' column is in datetime format
    dt2['timestamp'] = pd.to_datetime(dt2['timestamp'], errors='coerce')

    # Check if any timestamps failed to convert
    if dt2['timestamp'].isnull().any():
        raise ValueError("Some timestamps could not be converted to datetime format.")

    # Separar la columna de timestamp en año, mes, día, hora, minuto
    dt2['año'] = dt2['timestamp'].dt.year
    dt2['mes'] = dt2['timestamp'].dt.month
    dt2['dia'] = dt2['timestamp'].dt.day
    dt2['hora'] = dt2['timestamp'].dt.hour
    dt2['minuto'] = dt2['timestamp'].dt.minute

    # Codificación del tiempo del día
    dt2['tiempo_del_dia'] = dt2['hora'] + dt2['minuto'] / 60.0
    dt2['dia_sen'] = np.sin(2 * np.pi * dt2['tiempo_del_dia'] / 24)
    dt2['dia_cos'] = np.cos(2 * np.pi * dt2['tiempo_del_dia'] / 24)

    dt2['dia_semana'] = dt2['timestamp'].dt.dayofweek
    dt2['sem_sen'] = np.sin(2 * np.pi * dt2['dia_semana'] / 7)
    dt2['sem_cos'] = np.cos(2 * np.pi * dt2['dia_semana'] / 7)


    dt2 = dt2[['activa', 'dia_sen', 'dia_cos', 'sem_sen', 'sem_cos']]
    logging.info("tiempo codificado.")

    return dt2



def cargar_datos(archivo_potencias='potencias.csv', archivo_corrientes='corrientes.csv'):
    """
    Carga los datos desde los archivos y registra el resultado en el log.

    Args:
    archivo_potencias (str): La ruta al archivo de potencias (csv).
    archivo_corrientes (str): La ruta al archivo de corrientes (csv)

    Returns:
    final_df: dataframe con los datos si la carga fue exitosa, None sino.
    """
    try:
        # Leer encabezados para verificar que los archivos se pueden abrir
        encabezados_corrientes = pd.read_csv(archivo_corrientes, nrows=0).columns
        encabezados_potencias = pd.read_csv(archivo_potencias, nrows=0).columns
        
        fila_inicio = 1
        numero_filas = 120000  # Número de filas que deseas cargar
        
        # Leer los archivos CSV
        corrientes = pd.read_csv(archivo_corrientes, skiprows=fila_inicio, nrows=numero_filas, header=None, names=encabezados_corrientes)
        potencias = pd.read_csv(archivo_potencias, skiprows=fila_inicio, nrows=numero_filas, header=None, names=encabezados_potencias)

        # Convertir la columna 'timestamp' a datetime
        corrientes['timestamp'] = pd.to_datetime(corrientes['timestamp'])
        potencias['timestamp'] = pd.to_datetime(potencias['timestamp'])

        # Unir los dataframes en base al ID y timestamp
        df_unido = pd.merge(corrientes, potencias, on=['id', 'timestamp'])

        # Separar la columna de timestamp en año, mes, día, hora, minuto
        df_unido['año'] = df_unido['timestamp'].dt.year
        df_unido['mes'] = df_unido['timestamp'].dt.month
        df_unido['dia'] = df_unido['timestamp'].dt.day
        df_unido['hora'] = df_unido['timestamp'].dt.hour
        df_unido['minuto'] = df_unido['timestamp'].dt.minute

        # Codificación del tiempo del día
        df_unido['tiempo_del_dia'] = df_unido['hora'] + df_unido['minuto'] / 60.0
        df_unido['dia_sen'] = np.sin(2 * np.pi * df_unido['tiempo_del_dia'] / 24)
        df_unido['dia_cos'] = np.cos(2 * np.pi * df_unido['tiempo_del_dia'] / 24)

        # Codificación del día del año
        #df_unido['dia_del_año'] = df_unido['timestamp'].dt.dayofyear
        #df_unido['mes_sen'] = np.sin(2 * np.pi * df_unido['dia_del_año'] / 365)
        #df_unido['mes_cos'] = np.cos(2 * np.pi * df_unido['dia_del_año'] / 365)

        df_unido['dia_semana'] = df_unido['timestamp'].dt.dayofweek
        df_unido['mes_sen'] = np.sin(2 * np.pi * df_unido['dia_semana'] / 7)
        df_unido['mes_cos'] = np.cos(2 * np.pi * df_unido['dia_semana'] / 7)

        # Seleccionar y reorganizar las columnas en el formato deseado
        final_df = df_unido[['activa', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos']]
        logging.info("Datos cargados.")

        if np.any(np.isnan(final_df)):
                print("Hay valores NaN en los datos.")
        if np.any(np.isinf(final_df)):
                print("Hay valores infinitos en los datos.")
        return final_df
    
    except Exception as e:
        logging.error(f'Error al cargar los datos: {e}')
        return None




import pandas as pd
import numpy as np
import logging

import pandas as pd
import numpy as np
import logging

import pandas as pd
import numpy as np
import logging

def cargar_datos_especificos(archivo_potencias='potencias.csv', dias_semanales=None, horas=None):
    """
    Carga los datos desde el archivo CSV de potencias, filtra según días de la semana y horas, y registra el resultado en el log.

    Args:
    archivo_potencias (str): La ruta al archivo de potencias (csv).
    dias_semanales (list): Lista de días de la semana a cargar (0=domingo, 1=lunes, ..., 6=sábado). 
                           Si es None, no filtra por días de la semana.
    horas (list): Lista de horas (0-23) a cargar. Si es None, no filtra por horas.
    
    Returns:
    final_df: dataframe con los datos si la carga fue exitosa, None sino.
    """
    try:
        # Leer encabezados para verificar que los archivos se pueden abrir
        encabezados_potencias = pd.read_csv(archivo_potencias, nrows=0).columns
        
        fila_inicio = 1
        numero_filas = 120000  # Número de filas que deseas cargar
        
        # Leer el archivo CSV de potencias
        potencias = pd.read_csv(archivo_potencias, skiprows=fila_inicio, nrows=numero_filas, header=None, names=encabezados_potencias)

        # Convertir la columna 'timestamp' a datetime
        potencias['timestamp'] = pd.to_datetime(potencias['timestamp'], errors='coerce')

        # Verificar si alguna fecha no fue convertida correctamente
        if potencias['timestamp'].isnull().any():
            raise ValueError("Algunas fechas no pudieron ser convertidas correctamente en el archivo de potencias.")

        # Codificación del tiempo del día
        potencias['tiempo_del_dia'] = potencias['timestamp'].dt.hour + potencias['timestamp'].dt.minute / 60.0
        potencias['dia_sen'] = np.sin(2 * np.pi * potencias['tiempo_del_dia'] / 24)
        potencias['dia_cos'] = np.cos(2 * np.pi * potencias['tiempo_del_dia'] / 24)

        # Codificación del día de la semana
        potencias['dia_semana'] = potencias['timestamp'].dt.dayofweek
        potencias['mes_sen'] = np.sin(2 * np.pi * potencias['dia_semana'] / 7)
        potencias['mes_cos'] = np.cos(2 * np.pi * potencias['dia_semana'] / 7)

        # Normalización de la codificación cíclica a rango [0, 1]
        potencias['dia_sen'] = (potencias['dia_sen'] + 1) / 2
        potencias['dia_cos'] = (potencias['dia_cos'] + 1) / 2
        potencias['mes_sen'] = (potencias['mes_sen'] + 1) / 2
        potencias['mes_cos'] = (potencias['mes_cos'] + 1) / 2

        # Filtrar por días de la semana
        if dias_semanales is not None:
            potencias = potencias[potencias['timestamp'].dt.dayofweek.isin(dias_semanales)]

        # Filtrar por horas
        if horas is not None:
            potencias = potencias[potencias['timestamp'].dt.hour.isin(horas)]

        # Calcular la diferencia con el valor anterior en la columna 'activa'
        #potencias['diferencia_activa'] = potencias['activa'].diff()

        # Agregar la columna "n° de medición"
        #potencias['numero_de_medicion'] = range(1, len(potencias) + 1)

        # Seleccionar y reorganizar las columnas en el formato deseado
        final_df = potencias[['activa', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos']]

        logging.info(f"Datos cargados y filtrados. Total de registros: {len(final_df)}")

        if np.any(np.isnan(final_df)):
            print("Hay valores NaN en los datos.")
        if np.any(np.isinf(final_df)):
            print("Hay valores infinitos en los datos.")
        
        return final_df
    
    except Exception as e:
        logging.error(f'Error al cargar los datos: {e}')
        return None




def cargar_datos_especificos2(archivo_potencias='potencias.csv', dias_semanales=None, horas=None):
    """
    Carga los datos desde el archivo CSV de potencias, filtra según días de la semana y horas, y registra el resultado en el log.

    Args:
    archivo_potencias (str): La ruta al archivo de potencias (csv).
    dias_semanales (list): Lista de días de la semana a cargar (0=domingo, 1=lunes, ..., 6=sábado). 
                           Si es None, no filtra por días de la semana.
    horas (list): Lista de horas (0-23) a cargar. Si es None, no filtra por horas.
    
    Returns:
    final_df: dataframe con los datos si la carga fue exitosa, None sino.
    """
    try:
        # Leer encabezados para verificar que los archivos se pueden abrir
        encabezados_potencias = pd.read_csv(archivo_potencias, nrows=0).columns
        
        fila_inicio = 1
        numero_filas = 120000  # Número de filas que deseas cargar
        
        # Leer el archivo CSV de potencias
        potencias = pd.read_csv(archivo_potencias, skiprows=fila_inicio, nrows=numero_filas, header=None, names=encabezados_potencias)

        # Convertir la columna 'timestamp' a datetime
        potencias['timestamp'] = pd.to_datetime(potencias['timestamp'], errors='coerce')

        # Verificar si alguna fecha no fue convertida correctamente
        if potencias['timestamp'].isnull().any():
            raise ValueError("Algunas fechas no pudieron ser convertidas correctamente en el archivo de potencias.")

        # Codificación del tiempo del día
        potencias['tiempo_del_dia'] = potencias['timestamp'].dt.hour + potencias['timestamp'].dt.minute / 60.0
        potencias['dia_sen'] = np.sin(2 * np.pi * potencias['tiempo_del_dia'] / 24)
        potencias['dia_cos'] = np.cos(2 * np.pi * potencias['tiempo_del_dia'] / 24)

        # Codificación del día de la semana
        potencias['dia_semana'] = potencias['timestamp'].dt.dayofweek
        potencias['mes_sen'] = np.sin(2 * np.pi * potencias['dia_semana'] / 7)
        potencias['mes_cos'] = np.cos(2 * np.pi * potencias['dia_semana'] / 7)

        # Normalización de la codificación cíclica a rango [0, 1]
        potencias['dia_sen'] = (potencias['dia_sen'] + 1) / 2
        potencias['dia_cos'] = (potencias['dia_cos'] + 1) / 2
        potencias['mes_sen'] = (potencias['mes_sen'] + 1) / 2
        potencias['mes_cos'] = (potencias['mes_cos'] + 1) / 2

        # Filtrar por días de la semana
        if dias_semanales is not None:
            potencias = potencias[potencias['timestamp'].dt.dayofweek.isin(dias_semanales)]

        # Filtrar por horas
        if horas is not None:
            potencias = potencias[potencias['timestamp'].dt.hour.isin(horas)]

        # Calcular la diferencia con el valor anterior en la columna 'activa'
        #potencias['diferencia_activa'] = potencias['activa'].diff()

        # Agregar la columna "n° de medición"
        potencias['numero_de_medicion'] = range(1, len(potencias) + 1)

        # Verificar si hay valores NaN en el DataFrame
        if np.any(np.isnan(potencias)):
            # Encontrar las filas que contienen NaN
            filas_nan = potencias[potencias.isnull().any(axis=1)]
            print("Filas con valores NaN:")
            print(filas_nan)
        
        # Seleccionar y reorganizar las columnas en el formato deseado
        final_df = potencias[['activa', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos', 'numero_de_medicion']]

        logging.info(f"Datos cargados y filtrados. Total de registros: {len(final_df)}")

        if np.any(np.isnan(final_df)):
            print("Hay valores NaN en los datos.")
        if np.any(np.isinf(final_df)):
            print("Hay valores infinitos en los datos.")
        
        return final_df
    
    except Exception as e:
        logging.error(f'Error al cargar los datos: {e}')
        return None


def escalar_datos(Xtrain, ytrain, scalers):
    Xtrain_n = Xtrain.copy()

    Xtrain_n[:, :, 0] = scalers['scaleractiva'].transform(Xtrain[:, :, 0])
    Xtrain_n[:, :, 1] = scalers['scalersenhora'].transform(Xtrain[:, :, 1]) 
    Xtrain_n[:, :, 2] = scalers['scalercoshora'].transform(Xtrain[:, :, 2])
    Xtrain_n[:, :, 3] = scalers['scalersendia'].transform(Xtrain[:, :, 3])
    Xtrain_n[:, :, 4] = scalers['scalercosdia'].transform(Xtrain[:, :, 4])

    ytrain_n = ytrain.copy()
    ytrain_n = ytrain_n.reshape(-1, 1)  
    ytrain_n = scalers['salidas'].transform(ytrain_n)  

    logging.info("datos escalados")
    return Xtrain_n, ytrain_n


def escalar_entrada(Xtrain, scalers):
    Xtrain_n = Xtrain.copy()

    Xtrain_n[:, :, 0] = scalers['scaleractiva'].transform(Xtrain[:, :, 0])
    Xtrain_n[:, :, 1] = scalers['scalersenhora'].transform(Xtrain[:, :, 1]) 
    Xtrain_n[:, :, 2] = scalers['scalercoshora'].transform(Xtrain[:, :, 2])
    Xtrain_n[:, :, 3] = scalers['scalersendia'].transform(Xtrain[:, :, 3])
    Xtrain_n[:, :, 4] = scalers['scalercosdia'].transform(Xtrain[:, :, 4])

    logging.info("entrada escalada")
    return Xtrain_n







from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, BatchNormalization, Dense, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import HeNormal








from tensorflow.keras import layers

class Swish(layers.Layer):
    def call(self, inputs):
        return inputs * tf.nn.sigmoid(inputs)

def define_model(Xtrain, ytrain, Xval, yval, path_guardado='modelo_entrenado.h5'):
    # Capa de entrada
    input1 = Input(shape=(Xtrain.shape[1], Xtrain.shape[2]))

    # Primera capa LSTM con return_sequences=True para mantener la secuencia
    x = LSTM(288, return_sequences=True)(input1)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    # Segunda capa LSTM con return_sequences=True
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    # Tercera capa LSTM sin return_sequences, ya que solo necesitamos el último valor
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    
    # Capa densa con activación Swish
    x = Dense(128, activation=Swish())(x)

    # Capa de salida con activación Swish (puedes probar otras activaciones dependiendo de tu problema)
    dnn_output = Dense(4, activation=Swish())(x)

    # Definir el modelo
    model = Model(inputs=input1, outputs=[dnn_output])

    # Compilar el modelo con RMSprop
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.RMSprop())
    model.summary()

    # EarlyStopping para evitar sobreajuste
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # ModelCheckpoint para guardar el modelo durante el entrenamiento
    checkpoint = ModelCheckpoint(path_guardado, monitor='val_loss', save_best_only=True, verbose=1)

    try:
        # Entrenar el modelo con datos de validación, EarlyStopping y ModelCheckpoint
        model.fit(Xtrain, ytrain, epochs=250, verbose=1, batch_size=4,
                  validation_data=(Xval, yval), callbacks=[early_stopping, checkpoint])
    except MemoryError as e:
        print("Error de memoria: ", e)
        print("Guardando el modelo hasta el último punto alcanzado...")
        model.save(path_guardado)  # Guarda el modelo al momento de fallo
    except Exception as e:
        print(f"Se produjo un error: {e}")
        print("Guardando el modelo hasta el último punto alcanzado...")
        model.save(path_guardado)  # Guarda el modelo si ocurre otro error

    return model












from keras.layers import Attention, Concatenate, LSTM, Bidirectional, Dropout, BatchNormalization, Dense, Input
from keras.models import Model
from keras.layers import MultiHeadAttention

def entrenar_modelo_con_atencion(Xtrain, ytrain, Xval, yval, path_guardado='modelo_entrenado.h5'):
    # Asegurar reproducibilidad
    np.random.seed(47)
    tf.random.set_seed(47)
    initializer = GlorotUniform(seed=47)

    # Definir el modelo
    inputs = Input(shape=(Xtrain.shape[1], Xtrain.shape[2]))  # (timesteps, features)
    
    # Capa LSTM Bidireccional
    lstm_out = Bidirectional(LSTM(64, return_sequences=True, kernel_initializer=initializer, kernel_regularizer=l2(0.01)))(inputs)
    lstm_out = Dropout(0.1)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)
    
    # Aplicar el mecanismo de atención sobre la salida LSTM
    attention_out = MultiHeadAttention(num_heads=2, key_dim=10)(lstm_out, lstm_out)
    print(attention_out.shape)  # (None, 24, 128)
    
    # Aplicar GlobalAveragePooling1D para reducir la secuencia
    pooled_out = GlobalAveragePooling1D()(attention_out)  # Esto reducirá la salida a (None, 128)
    
    # Capa de salida
    output = Dense(ytrain.shape[1], activation="relu" , kernel_initializer=initializer)(pooled_out)

    # Crear el modelo
    model = Model(inputs=inputs, outputs=output)

    # Definir el optimizador y la compilación del modelo
    boundaries = [2, 3, 4, 10, 100, 250]
    values = [0.04, 0.005, 0.002, 0.0001, 0.00001, 0.00005, 0.000001]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries,
        values=values
    )
    optimizer = Adam(learning_rate=lr_schedule, clipnorm=1)
    model.compile(optimizer=optimizer, loss='mse')

    # Configuración de callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint(path_guardado, monitor='val_loss', save_best_only=True, verbose=1)

    # Entrenamiento del modelo
    model.fit(Xtrain, ytrain, epochs=1, batch_size=1, validation_data=(Xval, yval), callbacks=[early_stopping, checkpoint])

    return model




def calcular_resultados(ytest, prediccionesTest, carpeta):
    # Crear listas para almacenar los resultados
    resultados = []
    errores_totales = []  # Para almacenar todos los errores

    # Inicializar listas para métricas por columna
    r2_por_columna = []
    mae_por_columna = []
    desviacion_estandar_por_columna = []

    for valor in range(len(ytest)):
        y_real = ytest[valor]
        prediccion = prediccionesTest[valor]
        
        # Calcular errores
        errores = [prediccion[i] - y_real[i] for i in range(len(y_real))]
        errores_totales.extend(errores)  # Guardamos todos los errores para análisis global

        # Calcular error promedio y desviación estándar
        error_promedio = np.mean(np.abs(errores))
        desviacion_estandar = np.std(errores)
        
        # Error relativo porcentual
        error_relativo_porcentual = [(abs(errores[i]) / abs(y_real[i])) * 100 if y_real[i] != 0 else 0 for i in range(len(y_real))]
        
        # Almacenar resultados
        for i in range(len(y_real)):
            resultados.append({
                'valor': valor,
                'prediccion': prediccion[i],
                'valor_real': y_real[i],
                'error': errores[i],
                'error_promedio': error_promedio,
                'desviacion_estandar': desviacion_estandar,
                'error_relativo_porcentual': error_relativo_porcentual[i]
            })

    # Convertir a DataFrame
    df_resultados = pd.DataFrame(resultados)

    # Calcular métricas globales
    error_promedio_global = np.mean(errores_totales)
    desviacion_estandar_global = np.std(errores_totales)
    error_relativo_porcentual_promedio = np.mean(df_resultados['error_relativo_porcentual'])

    # Calcular las métricas de errores menores a 1, 2, 3, 4, 5
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

    # Error máximo
    error_maximo = max(abs(error) for error in errores_totales)


    # Calcular MAPE por columna
    from sklearn.metrics import mean_absolute_percentage_error
    mape_por_columna = [mean_absolute_percentage_error(ytest[:, i], prediccionesTest[:, i]) for i in range(ytest.shape[1])]

    # Calcular R² por columna
    r2_por_columna = [r2_score(ytest[:, i], prediccionesTest[:, i]) for i in range(ytest.shape[1])]

    # Calcular MAE por columna
    mae_por_columna = [mean_absolute_error(ytest[:, i], prediccionesTest[:, i]) for i in range(ytest.shape[1])]
    mae_por_columna = [f"{mae:.2f}" for mae in mae_por_columna]


    r2_por_columna = [f"{r2:.2f}" for r2 in r2_por_columna]
    
    # Promediar los MAPE de todas las columnas para obtener un único valor
    mape_promedio = np.mean(mape_por_columna)
    mape_por_columna = [f"{mape * 100:.2f}%" for mape in mape_por_columna]

    # Calcular desviación estándar por columna
    desviacion_estandar_por_columna = [np.std(prediccionesTest[:, i] - ytest[:, i]) for i in range(ytest.shape[1])]
    desviacion_estandar_por_columna = [f"{desviacion:.2f}" for desviacion in desviacion_estandar_por_columna]

    # Guardar resultados globales en un archivo de texto
    resultados_txt_path = os.path.join(carpeta, 'resultados.txt')
    with open(resultados_txt_path, 'a') as f:
        f.write("\n")
        f.write(f"Error promedio global: {error_promedio_global:.2f}\n")
        f.write(f"Desviación estándar global: {desviacion_estandar_global:.2f}\n")
        f.write(f"Error relativo porcentual promedio global: {error_relativo_porcentual_promedio:.2f}%\n")
        f.write("\n")
        f.write(f"MAPE por columna: {mape_por_columna}\n")
        f.write(f"MAPE promedio: {mape_promedio*100:.2f}\n")
        f.write(f"R² por columna: {r2_por_columna}\n")
        f.write(f"MAE por columna: {mae_por_columna}\n")
        f.write(f"Desviación estándar por columna: {desviacion_estandar_por_columna}\n")
        f.write(f"Cantidad de datos: {total_datos}\n")
        f.write("\n")
        f.write(f"Cantidad de errores menores o iguales a 1: {errores_menores_a_1} ({porcentaje_menores_a_1:.2f}%)\n")
        f.write(f"Cantidad de errores menores o iguales a 2: {errores_menores_a_2} ({porcentaje_menores_a_2:.2f}%)\n")
        f.write(f"Cantidad de errores menores o iguales a 3: {errores_menores_a_3} ({porcentaje_menores_a_3:.2f}%)\n")
        f.write(f"Cantidad de errores menores o iguales a 4: {errores_menores_a_4} ({porcentaje_menores_a_4:.2f}%)\n")
        f.write(f"Cantidad de errores menores o iguales a 5: {errores_menores_a_5} ({porcentaje_menores_a_5:.2f}%)\n")
        f.write(f"El error más grande cometido: {error_maximo:.2f}\n")
        

    # Guardar a un archivo CSV los resultados individuales
    resultados_csv_path = os.path.join(carpeta, 'resultados_predicciones.csv')
    df_resultados.to_csv(resultados_csv_path, index=False)






def crear_carpeta_y_guardar(nombre_modelo):
    carpeta = f"modelos/{nombre_modelo}"
    
    # Verificar si la carpeta ya existe
    if os.path.exists(carpeta):
        respuesta = input(f"La carpeta '{carpeta}' ya existe. ¿Deseas sobrescribirla? (s/n): ").strip().lower()
        if respuesta != 's':
            print("Operación cancelada. No se sobrescribió la carpeta.")
            sys.exit(-1)
            return None, None
    
    # Crear la carpeta con el nombre del modelo
    os.makedirs(carpeta, exist_ok=True)

    # Copiar los scripts a la carpeta
    ruta_script = 'tools/red_principal.py'  # Ruta del script
    destino_script = os.path.join(carpeta, 'red_principal.py')
    shutil.copy(ruta_script, destino_script)

    ruta_script = 'principal.py'  # Ruta del script
    destino_script = os.path.join(carpeta, 'principal.py')
    shutil.copy(ruta_script, destino_script)

    # Crear un archivo de resultados
    resultados_path = os.path.join(carpeta, 'resultados.txt')

    # Si el archivo de resultados ya existe, preguntar si se desea sobrescribir
    if os.path.exists(resultados_path):
        respuesta = input(f"El archivo '{resultados_path}' ya existe. ¿Deseas sobrescribirlo? (s/n): ").strip().lower()
        if respuesta != 's':
            print("Operación cancelada. No se sobrescribió el archivo de resultados.")
            return None, None
    
    # Abrir el archivo de resultados para escribir
    with open(resultados_path, 'w') as f:
        f.write(f"Resultados de la predicción para el {nombre_modelo}:\n")

    print(f"Carpeta '{carpeta}' y archivo de resultados creado exitosamente.")
    return carpeta, resultados_path


def guardar_modelo_y_resultados(carpeta, modelo, scalers):
    # Guardar el modelo
    modelo_path = os.path.join(carpeta, 'modelo')
    modelo.save(modelo_path)

    # Guardar los escaladores
    scalers_path = os.path.join(carpeta, 'scalers.pkl')
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)


    return modelo_path, scalers_path


















def entrenar_modelo(Xtrain, ytrain, Xval, yval, path_guardado='modelo_entrenado.h5'):
    # Asegurar reproducibilidad
    np.random.seed(47)
    tf.random.set_seed(47)
    initializer = GlorotUniform(seed=47)

    # Define los intervalos y los valores de learning rate
    boundaries = [5, 10, 20, 50, 100, 250]  # Los límites de los intervalos (épocas en este caso)
    values = [0.005, 0.002, 0.001, 0.0001, 0.00005, 0.00001, 0.000001]  # Learning rates correspondientes a los intervalos

    # Crea el scheduler de learning rate
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries,
        values=values
    )
    # Crear el modelo LSTM
    model = Sequential()

    model.add(LSTM(128, return_sequences=False, input_shape=(Xtrain.shape[1], Xtrain.shape[2]), kernel_initializer=initializer ) )
    #model.add(Dropout(0.1))
    model.add(BatchNormalization())

    #model.add(Dense(128, activation='relu', kernel_initializer=initializer))
    #model.add(Dropout(0.2)) 
    #model.add(LSTM(20, return_sequences=False, kernel_initializer=initializer ) )
    #model.add(Dropout(0.1))
    #model.add(BatchNormalization())
    #model.add(LSTM(50, return_sequences=False, kernel_initializer=initializer ) )
    #model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Dense(ytrain.shape[1], activation="sigmoid"))

    # Compilar el modelo con el optimizador personalizado
    optimizer = Adam(learning_rate=lr_schedule, clipnorm=1)
    model.compile(optimizer=optimizer, loss='huber_loss')

    # EarlyStopping para evitar sobreajuste
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # ModelCheckpoint para guardar el modelo durante el entrenamiento
    checkpoint = ModelCheckpoint(path_guardado, monitor='val_loss', save_best_only=True, verbose=1)

    try:
        # Entrenar el modelo con datos de validación, EarlyStopping y ModelCheckpoint
        model.fit(Xtrain, ytrain, epochs=300, verbose=1, batch_size=128,
                  validation_data=(Xval, yval), callbacks=[early_stopping, checkpoint])
    except MemoryError as e:
        print("Error de memoria: ", e)
        print("Guardando el modelo hasta el último punto alcanzado...")
        model.save(path_guardado)  # Guarda el modelo al momento de fallo
    except Exception as e:
        print(f"Se produjo un error: {e}")
        print("Guardando el modelo hasta el último punto alcanzado...")
        model.save(path_guardado)  # Guarda el modelo si ocurre otro error

    return model

