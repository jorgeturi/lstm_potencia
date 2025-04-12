import logging
from tools.logger_config import setup_logger
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # Configurar logger
    setup_logger()

    from tools.red_principal import *
    path = "modelos/modelo 0.0.4.0(4entrada)/"
    ventana_entrada = 4
    ventana_prediccion = 4

    modelo = cargar_modelo(path + "modelo")
    if modelo is not None:
        modelo.summary()
        scal = cargar_escaladores(path + "scalers.pkl")
        if scal is not None:
            # Cargar datos
            dias = [0, 1, 2, 3, 4, 5, 6]
            horas = list(range(24))
            df = cargar_datos_especificos('potencias.csv', dias_semanales=dias, horas=horas)

            # Crear ventanas
            X, y = crear_ventana(df[0:200000], ventana_entrada, ventana_prediccion)

            # Separación de datos
            inicio_train = 0
            fin_train = 71500
            inicio_val = fin_train + 1
            fin_val = fin_train + 1 + 18100
            inicio_test = fin_val + 1
            fin_test = inicio_test + 1 + 11615

            Xtest = X[inicio_test:fin_test]
            ytest = y[inicio_test:fin_test]

            # Normalizar test
            Xtest_n = Xtest.copy()
            Xtest_n[:, :, 0] = scal['scaleractiva'].transform(Xtest[:, :, 0])

            logging.info("Inicio predicción")
            prediccionestest_n = modelo.predict(Xtest_n)
            prediccionestest = scal['salidas'].inverse_transform(prediccionestest_n)
            logging.info("Fin predicción")

            # ======================================================
            # Predicción Virtual a partir del último valor de test
            # ======================================================
            pasos_virtuales = 50  # Número de pasos a predecir virtualmente

            # Usamos la última ventana de test normalizada como semilla
            ventana_actual = Xtest_n[-1].copy()
            predicciones_virtuales = []

            logging.info("Inicio de Predicción Virtual")
            for i in range(pasos_virtuales):
                # --- DECODIFICAR el HORARIO de la última medición de la ventana ---
                sin_day = ventana_actual[-1, 1] * 2 - 1  # Desnormalizar seno día
                cos_day = ventana_actual[-1, 2] * 2 - 1  # Desnormalizar coseno día
                angulo_day = np.arctan2(sin_day, cos_day)  # Reconstruir ángulo día
                if angulo_day < 0:
                    angulo_day += 2 * np.pi
                tiempo_actual = angulo_day / (2 * np.pi) * 24  # Convertir a horas

                # Imprimir valores de codificación del ciclo diario
                print(f"Paso {i+1} - Seno día: {sin_day:.4f}, Coseno día: {cos_day:.4f}")
                print(f"Paso {i+1} - Ángulo día reconstruido: {angulo_day:.4f} rad, Tiempo actual: {tiempo_actual:.2f} horas")

                # --- DECODIFICAR el DÍA de la semana de la última medición de la ventana ---
                sin_year = ventana_actual[-1, 3] * 2 - 1  # Desnormalizar seno año
                cos_year = ventana_actual[-1, 4] * 2 - 1  # Desnormalizar coseno año
                angulo_year = np.arctan2(sin_year, cos_year)  # Reconstruir ángulo año
                if angulo_year < 0:
                    angulo_year += 2 * np.pi
                dia_actual = (angulo_year / (2 * np.pi)) * 7  # Convertir a día de la semana

                # Imprimir valores de codificación del ciclo anual
                print(f"Paso {i+1} - Seno año: {sin_year:.4f}, Coseno año: {cos_year:.4f}")
                print(f"Paso {i+1} - Ángulo año reconstruido: {angulo_year:.4f} rad, Día actual: {dia_actual:.2f}")

                # Asegurarse de que dia_actual esté dentro del rango [0, 6] (días de la semana)
                dia_actual = np.round(dia_actual) % 7  # Redondear al valor más cercano y asegurarse que esté en el rango [0, 6]
                print(f"Paso {i+1} - Día actual ajustado: {dia_actual:.2f}")

                # Actualizar el horario: sumar 15 minutos
                nuevo_tiempo = tiempo_actual + 0.25
                if nuevo_tiempo >= 24:
                    nuevo_tiempo -= 24
                    dia_actual += 1  # Cambiar de día solo cuando llegamos a las 00:00 horas
                    if dia_actual >= 7:
                        dia_actual -= 7
                nuevo_angulo_day = nuevo_tiempo / 24 * 2 * np.pi  # Convertir a ángulo
                print(f"Paso {i+1} - Nuevo horario propuesto: {nuevo_tiempo:.2f} horas")

                # Actualizar el día: sumar 1 (nuevo día)
                nuevo_angulo_year = dia_actual / 7 * 2 * np.pi  # Convertir a ángulo
                print(f"Paso {i+1} - Nuevo día propuesto: {dia_actual:.2f}")

                # Realizar la predicción con la ventana actual
                ventana_actual_exp = np.expand_dims(ventana_actual, axis=0)
                pred_n = modelo.predict(ventana_actual_exp)
                if pred_n[0, 0] < 0:
                    pred_n[0, 0] = 0
                pred_inversa = scal['salidas'].inverse_transform(pred_n)[0, 0]
                predicciones_virtuales.append(pred_inversa)
                print(f"Paso {i+1} - Predicción (activa): {pred_inversa:.4f}")
                print(f"Paso {i+1} - Predicción normalizada: {pred_n[0, 0]:.4f}")

                # Actualizar la ventana para el siguiente paso:
                nuevo_registro = ventana_actual[-1].copy()
                nuevo_registro[0] = pred_n[0, 0]  # Actualizar la columna "activa"

                # Actualizar ciclo diario con el nuevo ángulo
                nuevo_registro[1] = (np.sin(nuevo_angulo_day) + 1) / 2  # Normalizar seno día
                nuevo_registro[2] = (np.cos(nuevo_angulo_day) + 1) / 2  # Normalizar coseno día

                # Actualizar ciclo anual con el nuevo ángulo correspondiente al nuevo día
                nuevo_registro[3] = (np.sin(nuevo_angulo_year) + 1) / 2  # Normalizar seno año
                nuevo_registro[4] = (np.cos(nuevo_angulo_year) + 1) / 2  # Normalizar coseno año

                # Imprimir los nuevos valores de seno y coseno
                print(f"Paso {i+1} - Nuevo seno día: {nuevo_registro[1]:.4f}, Nuevo coseno día: {nuevo_registro[2]:.4f}")
                print(f"Paso {i+1} - Nuevo seno año: {nuevo_registro[3]:.4f}, Nuevo coseno año: {nuevo_registro[4]:.4f}")

                # Actualizar la ventana: eliminar el primer registro y agregar el nuevo registro
                ventana_actual = np.concatenate([ventana_actual[1:], [nuevo_registro]], axis=0)
                print("Ventana actualizada:", ventana_actual, "\n")

            logging.info("Fin de Predicción Virtual")

            # ======================================================
            # Graficar Resultados (opcional)
            # ======================================================
            ultimos_reales = 100
            valores_reales = ytest[-ultimos_reales:, 0]
            valores_predichos_test = prediccionestest[-ultimos_reales:-pasos_virtuales, 0]
            x_reales = np.arange(ultimos_reales)
            x_virtual = np.arange(ultimos_reales - pasos_virtuales, ultimos_reales)

            plt.figure(figsize=(12, 5))
            plt.plot(x_reales, valores_reales, label='Valores Reales', color='blue')
            plt.plot(x_reales[:pasos_virtuales], valores_predichos_test, label='Predicción Test', color='green', linestyle='--')
            plt.plot(x_virtual, predicciones_virtuales, label='Predicción Virtual', color='red', linestyle='--')

            plt.title("Comparación: Valores Reales vs. Predicción Test vs. Predicción Virtual")
            plt.xlabel("Paso")
            plt.ylabel("Valor")
            plt.legend()
            plt.grid(True)

            imagen_path = os.path.join(path, "comparacion_virtual.png")
            plt.savefig(imagen_path)
            plt.show()
            logging.info(f"Imagen de comparación guardada en: {imagen_path}")