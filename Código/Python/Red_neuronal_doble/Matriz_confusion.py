import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from collections import Counter
from Preprocesado_modular import (
    cargar_csv,
    generar_ninguna,
    generar_caricia_dedo,
    generar_caricia_mano,
    generar_golpe_dedo,
    generar_golpe_mano,
    normalizar_ventanas
)


# --- Parámetros de ventaneo ---
ventana_tamaño = 1000
solapamiento = 100
ventanas_por_secuencia = 4
paso_secuencia = 2

# --- Rutas ---
ruta_csv = "C:\\Users\\dcast\\Desktop\\Iker C\\UNIVERSIDAD POLITÉCNICA DE MADRID\\5º AÑO\\2º CUATRIMESTRE\\TFG\\Código\\Python\\Detección de señal\\prueba_doble.csv"
modelo_path = 'modelos/mejor_modelo.keras'
etiquetas_path = 'datos/etiquetas_secuencias.npy'

# --- Cargar modelo entrenado y etiquetas ---
modelo = load_model(modelo_path)
clases = np.load(etiquetas_path)

# --- Cargar CSV completo ---
df = pd.read_csv(ruta_csv, sep=';')

# --- Filtrar por clase ---
df_nada         = df[df['Etiqueta'] == 'ninguna']
df_caricia_dedo = df[df['Etiqueta'] == 'caricia_dedo']
df_caricia_mano = df[df['Etiqueta'] == 'caricia_mano']
df_golpe_dedo   = df[df['Etiqueta'] == 'golpe_dedo']
df_golpe_mano   = df[df['Etiqueta'] == 'golpe_mano']

# --- Generar secuencias válidas ---
X0, y0 = generar_ninguna(df_nada)
X1, y1 = generar_caricia_dedo(df_caricia_dedo)
X2, y2 = generar_caricia_mano(df_caricia_mano)
X3, y3 = generar_golpe_dedo(df_golpe_dedo)
X4, y4 = generar_golpe_mano(df_golpe_mano)

# --- Clases generadas ---
for nombre, x, y_ in [
    ("ninguna", X0, y0),
    ("caricia_dedo", X1, y1),
    ("caricia_mano", X2, y2),
    ("golpe_dedo", X3, y3),
    ("golpe_mano", X4, y4)
]:
    print(f"Clase {nombre}: {len(y_)} muestras")

# --- Concatenar todo ---
X = np.concatenate([X0, X1, X2, X3, X4])
y = np.concatenate([y0, y1, y2, y3, y4])

# --- Normalización ---
X = normalizar_ventanas(X)

# --- Codificar etiquetas ---
le = LabelEncoder()
le.classes_ = clases
y_encoded = le.transform(y)
y_cat = to_categorical(y_encoded)

# --- Predicciones ---
pred_cat = modelo.predict(X)
pred_encoded = np.argmax(pred_cat, axis=1)

# --- Resultados ---
print("\n--- Classification Report ---")
print(classification_report(y_encoded, pred_encoded, target_names=clases))

# --- Matriz de confusión con anotaciones ---
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_encoded, pred_encoded)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)

print("\n--- Métricas Globales ---")
print("Accuracy:", accuracy_score(y_encoded, pred_encoded))
print("F1 macro:", f1_score(y_encoded, pred_encoded, average='macro'))

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap='Blues', ax=ax, colorbar=True)
plt.title("Matriz de Confusi\u00f3n - Multicanal")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
