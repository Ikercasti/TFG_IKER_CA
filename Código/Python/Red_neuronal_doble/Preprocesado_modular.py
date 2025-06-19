import numpy as np
import pandas as pd
import random
from collections import Counter
from scipy.signal import find_peaks

# --- Parámetros globales ---
ventana_tamano = 1000
solapamiento = 300
ventanas_por_secuencia = 4
paso_secuencia = 2
pico_en_ventana = 1
umbral_golpe = 2.5
umbral_caricia = 1.06
tiempo_entre_picos = 100
OFFSETS = [0.77, 0.76]   # Uno para cada sensor
ESCALAS = [1.65, 1.65]   # Uno para cada sensor

def cargar_csv(ruta_csv):
    df = pd.read_csv(ruta_csv, sep=';')
    df = df.dropna()
    df['Voltaje 1 (V)'] = df['Voltaje 1 (V)'].astype(float)
    df['Voltaje 2 (V)'] = df['Voltaje 2 (V)'].astype(float)
    return df

def normalizar_ventanas(X):
    
    X_norm = np.empty_like(X)
    for canal in range(X.shape[-1]):
        X_norm[..., canal] = (X[..., canal] - OFFSETS[canal]) / ESCALAS[canal]
    return X_norm


def detectar_picos(señal, umbral):
    picos, _ = find_peaks(señal, height=umbral, distance=tiempo_entre_picos)
    return picos

def generar_ninguna(df):
    voltajes1 = df['Voltaje 1 (V)'].values
    voltajes2 = df['Voltaje 2 (V)'].values
    etiquetas = df['Etiqueta'].values
    X, y = [], []
    paso = paso_secuencia * ventana_tamano
    secuencia_total = ventana_tamano * ventanas_por_secuencia

    for i in range(0, len(voltajes1) - secuencia_total, paso):
        seg1 = voltajes1[i:i + secuencia_total]
        seg2 = voltajes2[i:i + secuencia_total]
        etiquetas_segmento = etiquetas[i:i + secuencia_total]

        ventanas1 = np.array([seg1[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])
        ventanas2 = np.array([seg2[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])

        if ventanas1.shape != (ventanas_por_secuencia, ventana_tamano):
            continue

        p1 = detectar_picos(seg1, umbral_caricia)
        p2 = detectar_picos(seg2, umbral_caricia)

        if p1.size == 0 and p2.size == 0:
            X.append(np.stack([ventanas1, ventanas2], axis=-1))
            y.append("ninguna")

    return np.array(X), np.array(y)

def generar_caricia_dedo(df):
    voltajes1 = df['Voltaje 1 (V)'].values
    voltajes2 = df['Voltaje 2 (V)'].values
    etiquetas = df['Etiqueta'].values
    X, y = [], []
    paso = paso_secuencia * ventana_tamano
    secuencia_total = ventana_tamano * ventanas_por_secuencia

    for i in range(0, len(voltajes1) - secuencia_total, paso):
        seg1 = voltajes1[i:i + secuencia_total]
        seg2 = voltajes2[i:i + secuencia_total]

        p1 = detectar_picos(seg1, umbral_caricia)
        p2 = detectar_picos(seg2, umbral_caricia)

        if (len(p1) == 0 and len(p2) == 0) or (len(p1) > 0 and len(p2) > 0):
            continue  # Solo uno de los dos debe detectar caricia

        ventanas1 = np.array([seg1[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])
        ventanas2 = np.array([seg2[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])

        if ventanas1.shape != (ventanas_por_secuencia, ventana_tamano):
            continue

        X.append(np.stack([ventanas1, ventanas2], axis=-1))
        y.append("caricia_dedo")

    return np.array(X), np.array(y)

def generar_caricia_mano(df):
    voltajes1 = df['Voltaje 1 (V)'].values
    voltajes2 = df['Voltaje 2 (V)'].values
    etiquetas = df['Etiqueta'].values
    X, y = [], []
    paso = paso_secuencia * ventana_tamano
    secuencia_total = ventana_tamano * ventanas_por_secuencia

    for i in range(0, len(voltajes1) - secuencia_total, paso):
        seg1 = voltajes1[i:i + secuencia_total]
        seg2 = voltajes2[i:i + secuencia_total]

        p1 = detectar_picos(seg1, umbral_caricia)
        p2 = detectar_picos(seg2, umbral_caricia)

        if len(p1) == 0 or len(p2) == 0:
            continue  # Ambos deben detectar caricia

        ventanas1 = np.array([seg1[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])
        ventanas2 = np.array([seg2[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])

        if ventanas1.shape != (ventanas_por_secuencia, ventana_tamano):
            continue

        X.append(np.stack([ventanas1, ventanas2], axis=-1))
        y.append("caricia_mano")

    return np.array(X), np.array(y)

def generar_golpe_dedo(df):
    voltajes1 = df['Voltaje 1 (V)'].values
    voltajes2 = df['Voltaje 2 (V)'].values
    etiquetas = df['Etiqueta'].values
    X, y = [], []
    paso = paso_secuencia * ventana_tamano
    secuencia_total = ventana_tamano * ventanas_por_secuencia

    for i in range(0, len(voltajes1) - secuencia_total, paso):
        seg1 = voltajes1[i:i + secuencia_total]
        seg2 = voltajes2[i:i + secuencia_total]

        p1 = detectar_picos(seg1, umbral_golpe)
        p2 = detectar_picos(seg2, umbral_golpe)

        if (len(p1) == 0 and len(p2) == 0) or (len(p1) > 0 and len(p2) > 0):
            continue  # Solo uno debe detectar golpe

        ventanas1 = np.array([seg1[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])
        ventanas2 = np.array([seg2[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])

        if ventanas1.shape != (ventanas_por_secuencia, ventana_tamano):
            continue

        X.append(np.stack([ventanas1, ventanas2], axis=-1))
        y.append("golpe_dedo")

    return np.array(X), np.array(y)

def generar_golpe_mano(df):
    voltajes1 = df['Voltaje 1 (V)'].values
    voltajes2 = df['Voltaje 2 (V)'].values
    etiquetas = df['Etiqueta'].values
    X, y = [], []
    paso = paso_secuencia * ventana_tamano
    secuencia_total = ventana_tamano * ventanas_por_secuencia

    for i in range(0, len(voltajes1) - secuencia_total, paso):
        seg1 = voltajes1[i:i + secuencia_total]
        seg2 = voltajes2[i:i + secuencia_total]

        p1 = detectar_picos(seg1, umbral_golpe)
        p2 = detectar_picos(seg2, umbral_golpe)

        if len(p1) == 0 or len(p2) == 0:
            continue  # Ambos deben detectar golpe

        ventanas1 = np.array([seg1[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])
        ventanas2 = np.array([seg2[j:j + ventana_tamano] for j in range(0, secuencia_total, ventana_tamano)])

        if ventanas1.shape != (ventanas_por_secuencia, ventana_tamano):
            continue

        X.append(np.stack([ventanas1, ventanas2], axis=-1))
        y.append("golpe_mano")

    return np.array(X), np.array(y)
